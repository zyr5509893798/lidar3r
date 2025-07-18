import json
import os
import sys

import einops
import lightning as L
import lpips
import omegaconf
import torch
import wandb

# Add MAST3R and PixelSplat to the sys.path to prevent issues during importing
sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
import data.waymo.waymo as waymo  # 引入我们的waymo数据集处理函数
import src.mast3r_src.mast3r.model as mast3r_model
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.export as export
import utils.geometry as geometry
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils
import workspace


class MAST3RGaussians(L.LightningModule):

    def __init__(self, config):

        super().__init__()

        # Save the config
        self.config = config

        # The encoder which we use to predict the 3D points and Gaussians,
        # trained as a modified MAST3R model. The model's configuration is
        # primarily defined by the pretrained checkpoint that we load, see
        # MASt3R's README.md
        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='gaussian_head',
            output_mode='pts3d+gaussian+desc24',
            depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
            conf_mode=('exp', 1, mast3r_model.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=config.use_offsets,
            sh_degree=config.sh_degree if hasattr(config, 'sh_degree') else 1
        )
        # 冻结整个编码器
        self.encoder.requires_grad_(False)

        # 解冻新增的深度融合模块
        self.encoder.fusion_gate.requires_grad_(True)
        self.encoder.depth_encoder.requires_grad_(True)

        # 解冻原始模型中需要训练的部分
        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)

        # The decoder which we use to render the predicted Gaussians into
        # images, lightly modified from PixelSplat
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        self.benchmarker = benchmarker.Benchmarker()

        # Loss criteria
        if config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

        if config.loss.mast3r_loss_weight is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        self.save_hyperparameters()

    def forward(self, view1, view2):

        # # Freeze the encoder and decoder
        # (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
        #
        # with torch.no_grad():
        #     # (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
        #     dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)
        # 编码
        (shape1, shape2), (tokens1, tokens2), (pos1, pos2), (depth_feat1, depth_feat2) = self.encoder._encode_symmetrized(view1, view2)

        # 特征融合
        tokens1_fused = self.encoder._fuse_features(tokens1, shape1, depth_feat1)
        tokens2_fused = self.encoder._fuse_features(tokens2, shape2, depth_feat2)

        with torch.no_grad():
            # 解码器处理融合特征
            dec1, dec2 = self.encoder._decoder(tokens1_fused, pos1, tokens2_fused, pos2)

        # Train the downstream heads
        pred1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)

        pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])

        # SH残差学习，用于计算球谐函数残差，与3dgs的外观有关
        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1['sh'])
            # new_sh2 = torch.zeros_like(pred2['sh'])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            # new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            # 深度图这部分理论上不需要参与
            #pred2['sh'] = pred2['sh'] + new_sh2

        # Update the keys to make clear that pts3d and means are in view1's frame
        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')

        return pred1, pred2

    def training_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']  # 现在是双图输入了
        # 单图输入，注意这里要取出第一个字典，这里原本是[view1, view2]，现在长这样[view1]，但是我们要把view1取出来，而不是整个列表

        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        # color, _ = self.decoder(batch, pred1, pred2, (h, w))
        # 修改获取decoder输出的部分
        color, depth = self.decoder(batch, pred1, pred2, (h, w))  # 获取深度渲染结果

        # 计算损失（传入深度）
        loss, mse, lpips, depth_loss = self.calculate_loss(
            batch, color, depth  # 添加depth参数
        )

        # Calculate losses
        # loss掩码，确定哪些像素点参与loss计算
        # 计算第一组视图（depth_1）中的哪些像素点，在3D空间中投影到第二组视图（depth_2）时满足三个条件：
        # 1 在第二组视图的视锥（frustum）内
        # 2 第一组视图中的深度值有效（非零）
        # 3 投影后的深度与第二组视图的实际深度匹配（允许微小误差）

        # 如果是单视图，不做匹配，那么就无法使用这个mask
        # mask = loss_mask.calculate_loss_mask(batch)

        # loss也需要改
        # loss, mse, lpips = self.calculate_loss(
        #     batch, color
        # )

        # Log losses
        self.log_metrics('train', loss, mse, lpips, depth_loss)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        # 记录各参数组学习率
        for i, pg in enumerate(self.optimizers().param_groups):
            self.log(f"lr/group_{i}", pg['lr'], prog_bar=(i == 0))

        # 监控新增模块梯度
        for name, param in self.encoder.fusion_gate.named_parameters():
            if param.grad is not None:
                self.log(f"grad_norm/fusion/{name}", param.grad.norm())

    def on_train_epoch_end(self):
        # 检查新增模块权重变化
        for name, param in self.encoder.fusion_gate.named_parameters():
            self.log(f"weight_mean/fusion/{name}", param.data.mean())
            self.log(f"weight_std/fusion/{name}", param.data.std())

    def validation_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']  # 现在是双图输入了

        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        # color, _ = self.decoder(batch, pred1, pred2, (h, w))
        #
        # # Calculate losses
        # # mask = loss_mask.calculate_loss_mask(batch)
        # loss, mse, lpips = self.calculate_loss(
        #     batch, color
        # )
        # 修改获取decoder输出的部分
        color, depth = self.decoder(batch, pred1, pred2, (h, w))  # 获取深度渲染结果

        # 计算损失（传入深度）
        loss, mse, lpips, depth_loss = self.calculate_loss(
            batch, color, depth  # 添加depth参数
        )

        # Log losses
        self.log_metrics('val', loss, mse, lpips, depth_loss)
        return loss

    def test_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']  # 现在是双图输入了
        num_targets = len(batch['target'])

        # Predict using the encoder/decoder and calculate the loss
        with self.benchmarker.time("encoder"):
            pred1, pred2 = self.forward(view1, view2)
        with self.benchmarker.time("decoder", num_calls=num_targets):
            color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        # mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips= self.calculate_loss(
            batch, color
        )

        # Log losses
        self.log_metrics('test', loss, mse, lpips)
        return loss

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(os.path.join(benchmark_file_path))

    def calculate_loss(self, batch, color, depth):  # 添加depth参数

        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color

        # 获取目标深度图和有效掩码
        target_depth = torch.stack([target_view['depthmap'] for target_view in batch['target']], dim=1)
        valid_mask = torch.stack([target_view['valid_mask'] for target_view in batch['target']], dim=1)

        # if apply_mask:
        #     assert mask.sum() > 0, "There are no valid pixels in the mask!"
        #     target_color = target_color * mask[..., None, :, :]
        #     predicted_colo
        #     r = predicted_color * mask[..., None, :, :]

        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        # flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

        # MSE loss
        rgb_l2_loss = (predicted_color - target_color) ** 2
        # if average_over_mask:
        #     mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / mask.sum()
        # else:
        mse_loss = rgb_l2_loss.mean()

        # LPIPS loss
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        # if average_over_mask:
        #     lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        # else:
        lpips_loss = lpips_loss.mean()

        # ===== 新增：深度损失计算 =====
        # 1. 确保深度图在相同设备
        depth = depth.to(valid_mask.device)

        # 2. 计算绝对误差
        depth_l1 = (depth - target_depth).abs()

        # 3. 只计算有效掩码区域
        depth_loss = (depth_l1 * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        # Calculate the total loss
        loss = 0
        loss += self.config.loss.mse_loss_weight * mse_loss
        loss += self.config.loss.lpips_loss_weight * lpips_loss
        # 4. 添加到总损失
        loss += self.config.loss.depth_loss_weight * depth_loss

        # MAST3R Loss，单张图无法使用这个loss，不做匹配，去掉了
        # if self.config.loss.mast3r_loss_weight is not None:
        #     mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
        #     loss += self.config.loss.mast3r_loss_weight * mast3r_loss

        # Masked SSIM
        # if calculate_ssim:
        #     if average_over_mask:
        #         ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
        #         ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        #     else:
        #         ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
        #         ssim_val = ssim_val.mean()
        #     return loss, mse_loss, lpips_loss, ssim_val

        return loss, mse_loss, lpips_loss, depth_loss  # 返回depth_loss

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None):
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
            f'{prefix}/depth_loss': depth_loss,  # 记录深度损失
        }

        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim

        prog_bar = prefix != 'val'
        sync_dist = prefix != 'train'
        self.log_dict(values, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=self.config.data.batch_size)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.opt.lr)
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config.opt.epochs // 2], gamma=0.1)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "epoch",
    #             "frequency": 1,
    #         },
    #     }
    def configure_optimizers(self):
        # 更保守的分组学习率
        param_groups = [
            {
                'params': list(self.encoder.fusion_gate.parameters()),
                'lr': self.config.opt.lr * 30,  # 3e-4
                'name': 'fusion'
            },
            {
                'params': list(self.encoder.depth_encoder.parameters()),
                'lr': self.config.opt.lr * 20,  # 2e-4
                'name': 'depth_enc'
            },
            {
                'params': list(self.encoder.downstream_head1.parameters()) +
                          list(self.encoder.downstream_head2.parameters()),
                'lr': self.config.opt.lr * 3,  # 3e-5
                'name': 'heads'
            }
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.opt.lr,
            weight_decay=self.config.opt.weight_decay,
            betas=(0.9, 0.98),  # 更保守的beta2
            eps=1e-6
        )

        # 使用阶梯式预热策略
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                # 阶段1: 线性预热 (5%的训练步数)
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=int(0.05 * self.trainer.estimated_stepping_batches)
                ),
                # 阶段2: 保持恒定 (45%的训练步数)
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                    total_iters=int(0.45 * self.trainer.estimated_stepping_batches)
                ),
                # 阶段3: 余弦衰减 (50%的训练步数)
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(0.5 * self.trainer.estimated_stepping_batches),
                    eta_min=self.config.opt.lr * 0.01  # 衰减到1e-7
                )
            ],
            milestones=[
                int(0.05 * self.trainer.estimated_stepping_batches),
                int(0.5 * self.trainer.estimated_stepping_batches)
            ]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_after_backward(self):
        # 监控新增模块梯度
        total_norm = 0
        for name, param in self.encoder.fusion_gate.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                self.log(f"grads/{name}_norm", param_norm)

        total_norm = total_norm ** 0.5
        self.log("grads/total_norm", total_norm)

        # 梯度裁剪（动态调整）
        clip_value = max(0.5, 1 / (total_norm + 1e-8))
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)


def run_experiment(config):

    # Set the seed
    L.seed_everything(config.seed, workers=True)

    # Set up loggers
    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []
    if config.loggers.use_csv_logger:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=config.save_dir,
            name=config.name
        )
        loggers.append(csv_logger)
    # if config.loggers.use_wandb:
    #     wandb_logger = L.pytorch.loggers.WandbLogger(
    #         project='splatt3r',
    #         name=config.name,
    #         save_dir=config.save_dir,
    #         config=omegaconf.OmegaConf.to_container(config),
    #     )
    #     if wandb.run is not None:
    #         wandb.run.log_code(".")
    #     loggers.append(wandb_logger)
    #
    # # Set up profiler
    # if config.use_profiler:
    #     profiler = L.pytorch.profilers.PyTorchProfiler(
    #         dirpath=config.save_dir,
    #         filename='trace',
    #         export_to_chrome=True,
    #         schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_dir),
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA
    #         ],
    #         profile_memory=True,
    #         with_stack=True
    #     )
    # else:
    #     profiler = None

    # Model
    print('Loading Model')
    model = MAST3RGaussians(config)
    if config.use_pretrained:
        ckpt = torch.load(config.pretrained_mast3r_path)
        _ = model.encoder.load_state_dict(ckpt['model'], strict=False)
        del ckpt

    # Training Datasets
    print(f'Building Datasets')
    train_dataset = waymo.get_waymo_dataset(
        config.data.root,
        'train',
        config.data.resolution,
        num_epochs_per_epoch=config.data.epochs_per_train_epoch,
    )
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    val_dataset = waymo.get_waymo_test_dataset(
        config.data.root,
        resolution=config.data.resolution,
        use_every_n_sample=100,
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Training
    print('Training')
    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            export.SaveBatchData(save_dir=config.save_dir),
        ],
        check_val_every_n_epoch=1,
        default_root_dir=config.save_dir,
        devices=config.devices,
        gradient_clip_val=config.opt.gradient_clip_val,
        log_every_n_steps=10,
        logger=loggers,
        max_epochs=config.opt.epochs,
        strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
    )
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)

    # Testing
    original_save_dir = config.save_dir
    results = {}
    test_dataset = waymo.get_waymo_test_dataset(
        config.data.root,
        resolution=config.data.resolution,
        use_every_n_sample=10
    )
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    new_save_dir = original_save_dir # 这里没必要搞那些复杂的
    os.makedirs(new_save_dir, exist_ok=True)
    model.config.save_dir = new_save_dir

    L.seed_everything(config.seed, workers=True)

    # Training
    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=[export.SaveBatchData(save_dir=config.save_dir), ],
        default_root_dir=config.save_dir,
        devices=config.devices,
        log_every_n_steps=10,
        strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
    )

    model.lpips_criterion = lpips.LPIPS('vgg')
    model.config.loss.apply_mask = False
    model.config.loss.average_over_mask = False
    res = trainer.test(model, dataloaders=data_loader_test)
    # results[f"alpha: {alpha}, beta: {beta}, apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"] = res
    results["waymo"] = res


    # Save the results
    save_path = os.path.join(original_save_dir, 'results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f)


        # masking_configs = ((True, False), (True, True))
        # for apply_mask, average_over_mask in masking_configs:

            # new_save_dir = os.path.join(
            #     original_save_dir,
            #     f'alpha_{alpha}_beta_{beta}_apply_mask_{apply_mask}_average_over_mask_{average_over_mask}'
            # )
            # os.makedirs(new_save_dir, exist_ok=True)
            # model.config.save_dir = new_save_dir
            #
            # L.seed_everything(config.seed, workers=True)
            #
            # # Training
            # trainer = L.Trainer(
            #     accelerator="gpu",
            #     benchmark=True,
            #     callbacks=[export.SaveBatchData(save_dir=config.save_dir),],
            #     default_root_dir=config.save_dir,
            #     devices=config.devices,
            #     log_every_n_steps=10,
            #     strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
            # )
            #
            # model.lpips_criterion = lpips.LPIPS('vgg', spatial=average_over_mask)
            # model.config.loss.apply_mask = False
            # model.config.loss.average_over_mask = False
            # res = trainer.test(model, dataloaders=data_loader_test)
            # results[f"alpha: {alpha}, beta: {beta}, apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"] = res
            #
            # # Save the results
            # save_path = os.path.join(original_save_dir, 'results.json')
            # with open(save_path, 'w') as f:
            #     json.dump(results, f)


if __name__ == "__main__":

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run training
    run_experiment(config)
