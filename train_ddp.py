import os
import json
import math
import time
import argparse
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import ResNet50


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def setup_distributed(backend: str = "nccl"):
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_dataloaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=val_transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_markdown_row(md_path, epoch, metrics):
    header = (
        "| Epoch | Train Loss | Val Loss | Top1 (%) | Top5 (%) | LR | Epoch Time (s) |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    row = f"| {epoch} | {metrics.get('train_loss',''):.4f} | {metrics.get('val_loss',''):.4f} | {metrics.get('val_top1',''):.2f} | {metrics.get('val_top5',''):.2f} | {metrics.get('lr',''):.6f} | {metrics.get('epoch_time',''):.2f} |\n"
    if not os.path.exists(md_path):
        with open(md_path, 'w') as f:
            f.write(header)
            f.write(row)
    else:
        with open(md_path, 'a') as f:
            f.write(row)


def plot_pngs(log_json_path, out_dir):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not os.path.exists(log_json_path):
        return
    with open(log_json_path, 'r') as f:
        hist = json.load(f)
    if not hist:
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(hist)
    # Loss plot
    plt.figure()
    if 'train_loss' in df:
        plt.plot(df['epoch'], df['train_loss'], label='train')
    if 'val_loss' in df:
        plt.plot(df['epoch'], df['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'loss.png'), dpi=200, bbox_inches='tight'); plt.close()
    # Accuracy plot
    plt.figure()
    if 'val_top1' in df:
        plt.plot(df['epoch'], df['val_top1'], label='val@1')
    if 'val_top5' in df:
        plt.plot(df['epoch'], df['val_top5'], label='val@5')
    plt.xlabel('epoch'); plt.ylabel('accuracy (%)'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'accuracy.png'), dpi=200, bbox_inches='tight'); plt.close()
    # LR plot
    if 'lr' in df:
        plt.figure()
        plt.plot(df['epoch'], df['lr'], label='lr')
        plt.xlabel('epoch'); plt.ylabel('learning rate'); plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'lr.png'), dpi=200, bbox_inches='tight'); plt.close()


def main():
    parser = argparse.ArgumentParser(description='ResNet50 ImageNet DDP training (from scratch)')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to ImageNet root containing train/ and val/')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--max-lr', type=float, default=0.4)
    parser.add_argument('--pct-start', type=float, default=0.3)
    parser.add_argument('--div-factor', type=float, default=25.0)
    parser.add_argument('--final-div-factor', type=float, default=1e4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=str, default='checkpoints')
    args = parser.parse_args()

    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])  # for CUDA device
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    device = torch.device('cuda', local_rank)

    run_name = args.run_name or f"resnet50_ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(args.output, run_name)
    logs_dir = os.path.join('logs', run_name)
    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(args)

    num_classes = len(train_loader.dataset.classes)
    model = ResNet50(num_classes=num_classes).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.max_lr / args.div_factor, momentum=args.momentum, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        three_phase=False,
        anneal_strategy='cos'
    )

    scaler = GradScaler(enabled=True)

    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=os.path.join('runs', run_name))

    start_epoch = 0
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    if args.resume and os.path.exists(ckpt_path):
        map_loc = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1

    def train_one_epoch(epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct1 = 0.0
        correct5 = 0.0
        n = 0
        t0 = time.time()
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            with torch.no_grad():
                top1, top5 = accuracy(outputs, targets, topk=(1, 5))
            running_loss += loss.detach().item() * images.size(0)
            correct1 += top1.item() * images.size(0) / 100.0
            correct5 += top5.item() * images.size(0) / 100.0
            n += images.size(0)

        # Reduce across processes
        totals = torch.tensor([running_loss, correct1, correct5, n], dtype=torch.float32, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        running_loss, correct1, correct5, n = totals.tolist()
        epoch_loss = running_loss / n
        train_top1 = 100.0 * (correct1 / n)
        train_top5 = 100.0 * (correct5 / n)
        epoch_time = time.time() - t0
        return epoch_loss, train_top1, train_top5, epoch_time

    def validate(epoch):
        model.eval()
        running_loss = 0.0
        correct1 = 0.0
        correct5 = 0.0
        n = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                top1, top5 = accuracy(outputs, targets, topk=(1, 5))
                running_loss += loss.detach().item() * images.size(0)
                correct1 += top1.item() * images.size(0) / 100.0
                correct5 += top5.item() * images.size(0) / 100.0
                n += images.size(0)

        totals = torch.tensor([running_loss, correct1, correct5, n], dtype=torch.float32, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        running_loss, correct1, correct5, n = totals.tolist()
        val_loss = running_loss / n
        val_top1 = 100.0 * (correct1 / n)
        val_top5 = 100.0 * (correct5 / n)
        return val_loss, val_top1, val_top5

    history = []
    md_path = os.path.join(logs_dir, 'training_log.md') if is_main_process() else None
    json_path = os.path.join(logs_dir, 'training_log.json') if is_main_process() else None

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_top1, train_top5, epoch_time = train_one_epoch(epoch)
        val_loss, val_top1, val_top5 = validate(epoch)

        lr_now = optimizer.param_groups[0]['lr']
        if is_main_process():
            entry = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_top1': float(train_top1),
                'train_top5': float(train_top5),
                'val_loss': float(val_loss),
                'val_top1': float(val_top1),
                'val_top5': float(val_top5),
                'lr': float(lr_now),
                'epoch_time': float(epoch_time),
            }
            history.append(entry)
            with open(json_path, 'w') as f:
                json.dump(history, f, indent=2)
            save_markdown_row(md_path, epoch + 1, {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_top1': val_top1,
                'val_top5': val_top5,
                'lr': lr_now,
                'epoch_time': epoch_time,
            })
            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('train/top1', train_top1, epoch)
                writer.add_scalar('train/top5', train_top5, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/top1', val_top1, epoch)
                writer.add_scalar('val/top5', val_top5, epoch)
                writer.add_scalar('lr', lr_now, epoch)

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'args': vars(args),
            }, os.path.join(out_dir, f'model_{epoch:03d}.pt'))
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'args': vars(args),
            }, ckpt_path)

    if is_main_process():
        plot_pngs(json_path, logs_dir)
        if writer:
            writer.close()

    cleanup_distributed()


if __name__ == '__main__':
    main()
