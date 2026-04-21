import os
import sys

import torch
import torchvision
from torch import optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from model.DynamicJSCCR import DynamicJSCCR
from options.config import ConfigParser
from utils import save_checkpoint_with_images

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])


def setup_cuda_and_model(config, model):
    """配置 CUDA 并将模型移至 GPU"""
    if config.use_gpu and torch.cuda.is_available():
        # 解析 GPU ID
        gpu_ids = config.gpu_ids
        # 设置可见 GPU
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using GPU(s): {gpu_ids}")
        print(f"Primary GPU: cuda:{gpu_ids[0]}")
        print(f"GPU count: {len(gpu_ids)}")
        # 将模型移至主 GPU
        model = model.to(device)
        # 如果有多块 GPU，使用 DataParallel
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            print(f"Model wrapped with DataParallel on GPUs: {gpu_ids}")
        return model, device
    else:
        device = torch.device('cpu')
        model = model.to(device)
        print("Using CPU")
        return model, device


class Criterion(torch.nn.Module):
    def __init__(self, beta):
        super(Criterion, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.beta = beta
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, x_recon, label, label_pred, cr):
        batch_size = x.shape[0]
        mse_loss = self.mse(x, x_recon).view(batch_size, -1).mean(dim=1)  # [B]
        ce_loss = self.ce(label_pred, label)  # [B]
        # cr 需要是 [batch_size] 形状，用于逐样本加权
        if cr.dim() == 0:
            cr = cr.unsqueeze(0)

        mse_loss = cr * self.beta * mse_loss
        ce_loss = (1 - cr) * ce_loss
        total_loss = mse_loss + ce_loss

        loss = {
            'mse_loss': mse_loss.mean(),
            'ce_loss': ce_loss.mean(),
            'loss': total_loss.mean()
        }
        return loss


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    加载检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）

    Returns:
        start_epoch: 从哪个 epoch 开始训练
        best_loss: 历史最佳损失
        loaded_metrics: 其他加载的指标
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found, starting from scratch.")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        # 处理 DataParallel 包装的情况
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            # 如果模型被 DataParallel 包装，需要移除 'module.' 前缀
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(state_dict)
        print(f"Loaded model weights from {checkpoint_path}")

    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded optimizer state from {checkpoint_path}")

    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded scheduler state from {checkpoint_path}")

    # 获取训练状态
    start_epoch = checkpoint.get('epoch', -1) + 1
    best_loss = checkpoint.get('best_loss', 1e5)
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch, best_loss


def train_one_epoch(model, train_loader, optimizer, criterion, config, epoch):
    """训练一个 epoch"""
    model.train()

    # 累积统计量
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    total_correct = 0
    total_samples = 0

    # 进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}', ascii=True)
    current_lr = optimizer.param_groups[0]['lr']
    pbar.set_description(f'Epoch {epoch + 1}/{config.epochs} | LR: {current_lr:.2e}')
    for batch_idx, (images, labels) in enumerate(pbar):
        # 数据移至设备
        images = images.to(config.device)
        labels = labels.to(config.device)
        batch_size = images.size(0)
        total_samples += batch_size

        # 随机生成 SNR 和 CR（论文 Algorithm 1 第 6 行）
        snr_db = torch.rand(batch_size, device=config.device) * (
                config.snr_db_max - config.snr_db_min) + config.snr_db_min
        cr = torch.rand(batch_size, device=config.device) * (
                config.cr_max - config.cr_min) + config.cr_min

        # 前向传播
        x_rec, label_pred = model(images, snr_db, cr)

        # 计算损失
        loss_dict = criterion(images, x_rec, labels, label_pred, cr)

        # 反向传播
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        # 累积损失
        total_loss += loss_dict['loss'].item() * batch_size
        total_mse += loss_dict['mse_loss'].item() * batch_size
        total_ce += loss_dict['ce_loss'].item() * batch_size

        # 累积准确率
        _, predicted = torch.max(label_pred, dim=1)
        total_correct += (predicted == labels).sum().item()

        # 计算当前平均指标（用于进度条显示）
        current_avg_loss = total_loss / total_samples
        current_avg_mse = total_mse / total_samples
        current_avg_ce = total_ce / total_samples
        current_accuracy = 100.0 * total_correct / total_samples

        # 更新进度条
        pbar.set_postfix({
            'Loss': f"{current_avg_loss:.4f}",
            'MSE': f"{current_avg_mse:.4f}",
            'CE': f"{current_avg_ce:.4f}",
            'Acc': f"{current_accuracy:.2f}%"
        })

        # ==================== 新增：定期保存图像对 ====================
        if config.save_vis and (batch_idx + 1) % config.vis_interval == 0:
            save_checkpoint_with_images(
                model, optimizer, epoch + 1, batch_idx + 1,
                images.detach(), x_rec.detach(), config
            )
            if config.verbose:
                print(f"\n  Saved visualization at epoch {epoch + 1}, batch {batch_idx + 1}")

    # 计算 epoch 平均指标
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_ce = total_ce / total_samples
    avg_accuracy = 100.0 * total_correct / total_samples
    return avg_loss, avg_mse, avg_ce, avg_accuracy


def validate(model, test_loader, criterion, config, snr_test=None, cr_test=None):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    correct = 0
    total = 0

    # 用于计算 PSNR
    psnr_total = 0.0

    # 创建进度条
    pbar = tqdm(
        test_loader,
        desc='Validation',
        ascii=True,  # 避免乱码
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(config.device)
            labels = labels.to(config.device)
            batch_size = images.size(0)
            total += batch_size

            # 测试时的 SNR 和 CR
            if snr_test is None:
                snr_db = torch.rand(batch_size, device=config.device) * (
                        config.snr_db_max - config.snr_db_min) + config.snr_db_min
            else:
                snr_db = torch.full((batch_size,), snr_test, device=config.device)

            if cr_test is None:
                cr = torch.full((batch_size,), 0.5, device=config.device)
            else:
                cr = torch.full((batch_size,), cr_test, device=config.device)

            # 前向传播
            x_rec, label_pred = model(images, snr_db, cr)

            # 计算损失
            loss_dict = criterion(images, x_rec, labels, label_pred, cr)

            total_loss += loss_dict['loss'].item() * batch_size
            total_mse += loss_dict['mse_loss'].item() * batch_size
            total_ce += loss_dict['ce_loss'].item() * batch_size

            # 计算准确率
            _, predicted = torch.max(label_pred, 1)
            correct += (predicted == labels).sum().item()

            # 计算 PSNR（假设图像范围 [0, 1]）
            mse_per_image = ((images - x_rec) ** 2).mean(dim=[1, 2, 3])
            psnr_per_image = 10 * torch.log10(1.0 / (mse_per_image + 1e-10))
            psnr_total += psnr_per_image.sum().item()

            # 计算当前平均指标
            current_avg_loss = total_loss / total
            current_accuracy = 100.0 * correct / total
            current_avg_psnr = psnr_total / total

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{current_avg_loss:.4f}",
                'Acc': f"{current_accuracy:.2f}%",
                'PSNR': f"{current_avg_psnr:.2f}"
            })

    avg_loss = total_loss / total
    avg_mse = total_mse / total
    avg_ce = total_ce / total
    accuracy = 100.0 * correct / total
    avg_psnr = psnr_total / total

    return avg_loss, avg_mse, avg_ce, accuracy, avg_psnr


def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, config, is_best=False, best_loss=1e5):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }
    weights_dir = os.path.join(config.save_dir, "model_weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 保存最佳模型
    if is_best:
        best_path = os.path.join(config.save_dir, "model_weights", 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved with accuracy: {accuracy:.2f}%,loss :{loss:.3f}")
    else:
        # 保存最新模型
        latest_path = os.path.join(config.save_dir, "model_weights", 'latest.pth')
        torch.save(checkpoint, latest_path)
        # 定期保存
        if (epoch + 1) % config.save_epochs == 0:
            epoch_path = os.path.join(config.save_dir, "model_weights", f'epoch_{epoch + 1}.pth')
            torch.save(checkpoint, epoch_path)


def main(argv):
    config_path = argv[0]  # 获取命令行参数中的配置文件路径
    is_train = argv[1] == 'train'  # 获取命令行参数中的训练/测试标志
    Config = ConfigParser(config_path, is_train=is_train).get_config()  # 解析配置文件并获取配置对象

    # 创建数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=Config.dataroot,  # 数据保存路径
        train=True,  # 训练集
        download=True,  # 如果没下载就下载
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=Config.dataroot,
        train=False,  # 测试集
        download=True,
        transform=transform_test
    )
    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,  # 论文中的 batch_size
        shuffle=Config.shuffle,  # 训练集需要打乱
        num_workers=Config.num_workers,  # 多进程加载
        pin_memory=True  # GPU 训练时加速
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,  # 论文中的 batch_size
        shuffle=False,  # 训练集需要打乱
        num_workers=Config.num_workers,  # 多进程加载
        pin_memory=True
    )

    # 创建模型
    model = DynamicJSCCR(Config.in_channels, Config.out_channels, Config.c, Config.K_max, Config.num_classes)
    model, device = setup_cuda_and_model(Config, model)
    Config.device = device

    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=Config.lr_max)
    # 创建余弦退火调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=Config.epochs,  # 总 epoch 数（一个完整周期）
        eta_min=Config.lr_min  # 最小学习率（论文未明确，常见 1e-5 ~ 1e-6）
    )
    criterion = Criterion(Config.beta)

    start_epoch = 0
    best_loss = 1e5

    if Config.resume_training:
        checkpoint_path = Config.resume
        start_epoch, best_loss = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        print(f"Resumed training from {checkpoint_path}")
        print(f"Start epoch: {start_epoch}")
        # 如果已经完成了所有 epoch
        if start_epoch >= Config.epochs:
            print(f"Training already completed ({start_epoch} >= {Config.epochs})")
            return

    for epoch in range(start_epoch, Config.epochs):
        train_loss, train_mse, train_ce, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion,
                                                                          Config, epoch)
        scheduler.step()  # 更新学习率
        log_path = os.path.join(Config.save_dir, 'training_log.txt')
        if epoch == 0:
            os.makedirs(Config.save_dir, exist_ok=True)
            with open(log_path, 'w') as f:
                f.write(f"{'Epoch':>6} | {'LR':>10} | {'Loss':>10} | {'MSE':>10} | {'CE':>10} | {'Acc':>8}\n")
                f.write("-" * 70 + "\n")
        with open(log_path, 'a') as f:
            current_lr = optimizer.param_groups[0]['lr']
            f.write(f"{epoch + 1:6d} | {current_lr:10.5e} | {train_loss:10.6f} | "
                    f"{train_mse:10.6f} | {train_ce:10.6f} | {train_accuracy:7.2f}%\n")

        if Config.use_eval and (epoch + 1) % Config.val_epochs == 0:
            print("enter validate")
            val_loss, val_mse, val_ce, val_acc, val_psnr = validate(model, test_loader=test_loader, criterion=criterion,
                                                                    config=Config)
            print(f"Validation Results:")
            print(f"  Loss:  {val_loss:.4f}")
            print(f"  MSE:   {val_mse:.6f}")
            print(f"  CE:    {val_ce:.4f}")
            print(f"  Acc:   {val_acc:.2f}%")
            print(f"  PSNR:  {val_psnr:.2f} dB")

            if val_loss < best_loss:
                print("save the best pth")
                save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, loss=val_loss,
                                accuracy=val_acc, config=Config, is_best=True, best_loss=val_loss)
                best_loss = val_loss

        save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, loss=train_loss,
                        accuracy=0, config=Config, is_best=False, best_loss=best_loss)


if __name__ == '__main__':
    main(sys.argv[1:])

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # print(f"\nDevice: {device}")
    #
    # # 创建 Criterion
    # criterion = Criterion(beta=200).to(device)
    # batch_size = 2
    # num_classes = 10
    # # 创建测试数据
    # x = torch.rand(batch_size, 3, 32, 32, device=device)  # [B, 3, 32, 32]
    # x_recon = torch.rand(batch_size, 3, 32, 32, device=device)  # [B, 3, 32, 32]
    # label = torch.randint(0, num_classes, (batch_size,), device=device)  # [B]
    # label_pred = torch.randn(batch_size, num_classes, device=device)  # [B, num_classes]
    # cr_per_sample = torch.tensor([0.2, 0.4], device=device)
    # loss = criterion(x, x_recon, label, label_pred, cr_per_sample)
