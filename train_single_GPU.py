import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
from math import sqrt

from model import ResNet50
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import logging
import argparse
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Params:
    def __init__(self):
        self.batch_size = 256
        self.name = "resnet_50_onecycle"
        self.workers = 12
        self.max_lr = 0.175
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 50
        self.pct_start = 0.3
        self.div_factor = 25.0
        self.final_div_factor = 1e4
        self.label_smoothing = 0.1

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class MetricLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []
        self.current_epoch_metrics = {}
        
    def log_metrics(self, epoch_metrics):
        # Update current epoch metrics instead of appending immediately
        self.current_epoch_metrics.update(epoch_metrics)
        
    def finalize_epoch(self):
        # Only append when epoch is complete
        if self.current_epoch_metrics:
            self.metrics.append(self.current_epoch_metrics.copy())
            with open(os.path.join(self.log_dir, 'training_log.json'), 'w') as f:
                json.dump(self.metrics, f, indent=4)
            self.current_epoch_metrics = {}

    def log_markdown_row(self, epoch_metrics):
        md_path = os.path.join(self.log_dir, 'training_log.md')
        header = (
            "| Epoch | Train Loss | Val Loss | Top1 (%) | Top5 (%) | LR | Epoch Time (s) |\n"
            "|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        if not os.path.exists(md_path):
            with open(md_path, 'w') as f:
                f.write(header)
        
        # Format row with proper handling of missing values
        epoch = epoch_metrics.get('epoch', '')
        train_loss = f"{epoch_metrics.get('train_loss', 0):.4f}" if epoch_metrics.get('train_loss') is not None else ''
        test_loss = f"{epoch_metrics.get('test_loss', 0):.4f}" if epoch_metrics.get('test_loss') is not None else ''
        test_acc = f"{epoch_metrics.get('test_accuracy', 0):.2f}" if epoch_metrics.get('test_accuracy') is not None else ''
        test_acc5 = f"{epoch_metrics.get('test_accuracy_top5', 0):.2f}" if epoch_metrics.get('test_accuracy_top5') is not None else ''
        lr = f"{epoch_metrics.get('learning_rate', 0):.6f}" if epoch_metrics.get('learning_rate') is not None else ''
        epoch_time = f"{epoch_metrics.get('epoch_time', 0):.2f}" if epoch_metrics.get('epoch_time') is not None else ''
        
        row = f"| {epoch} | {train_loss} | {test_loss} | {test_acc} | {test_acc5} | {lr} | {epoch_time} |\n"
        with open(md_path, 'a') as f:
            f.write(row)

    def plot_pngs(self):
        import matplotlib
        matplotlib.use('Agg')  # Set backend before importing pyplot
        import matplotlib.pyplot as plt
        
        if not self.metrics:
            return
        
        df = pd.DataFrame(self.metrics)
        
        # Loss plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'train_loss' in df.columns and not df['train_loss'].isna().all():
            ax.plot(df['epoch'], df['train_loss'], label='train', marker='o', linewidth=2)
        if 'test_loss' in df.columns and not df['test_loss'].isna().all():
            ax.plot(df['epoch'], df['test_loss'], label='val', marker='s', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Training and Validation Loss', fontsize=14)
        fig.savefig(os.path.join(self.log_dir, 'loss.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # Accuracy plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'test_accuracy' in df.columns and not df['test_accuracy'].isna().all():
            ax.plot(df['epoch'], df['test_accuracy'], label='val@1', marker='o', linewidth=2)
        if 'test_accuracy_top5' in df.columns and not df['test_accuracy_top5'].isna().all():
            ax.plot(df['epoch'], df['test_accuracy_top5'], label='val@5', marker='s', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Validation Accuracy', fontsize=14)
        fig.savefig(os.path.join(self.log_dir, 'accuracy.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # LR plot
        if 'learning_rate' in df.columns and not df['learning_rate'].isna().all():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['epoch'], df['learning_rate'], marker='o', linewidth=2, color='blue')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title('Learning Rate Schedule', fontsize=14)
            fig.savefig(os.path.join(self.log_dir, 'lr.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)

def train(dataloader, model, loss_fn, optimizer, scheduler, epoch, writer, scaler, metric_logger):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    #print("cuda available:" + str(torch.cuda.is_available()))
    #print("cuda count:" + str(torch.cuda.device_count()))
    #print("cuda device name:" + torch.cuda.get_device_name(0))
    #print("torch cuda version:" + str(torch.version.cuda))
    #print("torch backends cudnn version:" + str(torch.backends.cudnn.version()))

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)

        with autocast(enabled=True):
            pred = model(X)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Step the scheduler here, after each batch
        scheduler.step()

        running_loss += loss.item()
        total += y.size(0)

        # Calculate accuracy
        _, predicted = torch.max(pred.data, 1)
        correct += (predicted == y).sum().item()

        # Calculate top-5 accuracy
        _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
        correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

        batch_size = len(X)
        step = epoch * size + (batch + 1) * batch_size

 
        if batch % 100 == 0:
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            current_acc = 100 * correct / total
            current_acc5 = 100 * correct_top5 / total

            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.2f}%",
                "acc5": f"{current_acc5:.2f}%",
                "lr": f"{current_lr:.6f}"
            })
            if writer is not None:
                writer.add_scalar('training loss', current_loss, step)
                writer.add_scalar('training accuracy', current_acc, step)
                writer.add_scalar('training top5 accuracy', current_acc5, step)
                writer.add_scalar('learning rate', current_lr, step)

    epoch_time = time.time() - start0
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    accuracy_top5 = 100 * correct_top5 / total

    metrics = {
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'train_accuracy': accuracy,
        'train_accuracy_top5': accuracy_top5,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'epoch_time': epoch_time
    }
    metric_logger.log_metrics(metrics)

    logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, Top-5 Acc: {accuracy_top5:.2f}%, Time: {epoch_time:.2f}s")

def test(dataloader, model, loss_fn, epoch, writer, train_dataloader, metric_logger, calc_acc5=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    correct_top5 = 0

    progress_bar = tqdm(dataloader, desc=f"Testing Epoch {epoch+1}")

    with torch.no_grad():
        with autocast(enabled=True):
            for X, y in progress_bar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                
                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == y).sum().item()
                
                if calc_acc5:
                    _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                    correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()
                
                current_acc = 100 * correct / size
                current_acc5 = 100 * correct_top5 / size if calc_acc5 else 0
                current_loss = test_loss / (progress_bar.n + 1)
                
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.2f}%",
                    "acc5": f"{current_acc5:.2f}%"
                })

    test_loss /= num_batches
    accuracy = 100 * correct / size
    accuracy_top5 = 100 * correct_top5 / size if calc_acc5 else None

    metrics = {
        'epoch': epoch + 1,
        'test_loss': test_loss,
        'test_accuracy': accuracy,
        'test_accuracy_top5': accuracy_top5
    }
    metric_logger.log_metrics(metrics)

    step = epoch * len(train_dataloader.dataset)
    if writer is not None:
        writer.add_scalar('test loss', test_loss, step)
        writer.add_scalar('test accuracy', accuracy, step)
        if calc_acc5:
            writer.add_scalar('test accuracy5', accuracy_top5, step)

    logger.info(f"Test Epoch {epoch+1} - Loss: {test_loss:.4f}, Acc: {accuracy:.2f}%, Top-5 Acc: {accuracy_top5:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single-GPU ResNet50 training (from scratch) with OneCycleLR')
    parser.add_argument('--data-dir', type=str, required=False, default=None, help='Root path containing train/ and val/')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max-lr', type=float, default=None)
    parser.add_argument('--pct-start', type=float, default=None)
    parser.add_argument('--div-factor', type=float, default=None)
    parser.add_argument('--final-div-factor', type=float, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--label-smoothing', type=float, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    params = Params()
    if args.batch_size is not None: params.batch_size = args.batch_size
    if args.workers is not None: params.workers = args.workers
    if args.epochs is not None: params.epochs = args.epochs
    if args.max_lr is not None: params.max_lr = args.max_lr
    if args.pct_start is not None: params.pct_start = args.pct_start
    if args.div_factor is not None: params.div_factor = args.div_factor
    if args.final_div_factor is not None: params.final_div_factor = args.final_div_factor
    if args.run_name is not None: params.name = args.run_name
    if args.label_smoothing is not None: params.label_smoothing = args.label_smoothing

    # Create metric logger
    log_dir = os.path.join("logs", params.name)
    metric_logger = MetricLogger(log_dir)

    # dataset paths
    if args.data_dir is not None:
        training_folder_name = os.path.join(args.data_dir, 'train')
        val_folder_name = os.path.join(args.data_dir, 'val')
    else:
        # Fallback example paths; replace or provide --data-dir
        training_folder_name = './data/imagenet-mini/train'
        val_folder_name = './data/imagenet-mini/val'

    train_transformation = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name,
        transform=train_transformation
    )
    train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.workers,
        pin_memory=True,
    )

    val_transformation = transforms.Compose([
        transforms.Resize(size=256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_folder_name,
        transform=val_transformation
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params.batch_size,   
        num_workers=params.workers,
        shuffle=False,
        pin_memory=True
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")

    resume_training = False

    num_classes = len(train_dataset.classes)
    model = ResNet50(num_classes=num_classes)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), 
                               lr=params.max_lr/params.div_factor,
                               momentum=params.momentum,
                               weight_decay=params.weight_decay)

    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=True)

    steps_per_epoch = len(train_loader)
    total_steps = params.epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.max_lr,
        total_steps=total_steps,
        pct_start=params.pct_start,
        div_factor=params.div_factor,
        final_div_factor=params.final_div_factor
    )

    start_epoch = 0
    checkpoint_path = os.path.join("checkpoints", params.name, f"checkpoint.pth")
    
    if resume_training and os.path.exists(checkpoint_path):
        print("Resuming training from checkpoint")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        assert params == checkpoint["params"]

    from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path
    Path(os.path.join("checkpoints", params.name)).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter('runs/' + params.name)
    
    test(val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, 
         metric_logger=metric_logger, calc_acc5=True)
    
    print("Starting training")
    for epoch in range(start_epoch, params.epochs):
        print(f"Epoch {epoch}")
        train(train_loader, model, loss_fn, optimizer, scheduler, epoch=epoch, writer=writer, 
              scaler=scaler, metric_logger=metric_logger)
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "params": params
        }
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"model_{epoch}.pth"))
        torch.save(checkpoint, os.path.join("checkpoints", params.name, f"checkpoint.pth"))
        
        test(val_loader, model, loss_fn, epoch + 1, writer, train_dataloader=train_loader,
             metric_logger=metric_logger, calc_acc5=True)

        # Finalize epoch metrics (combines train + test metrics)
        metric_logger.finalize_epoch()
        
        # Log markdown row and update PNGs after each epoch
        if metric_logger.metrics:
            last_metrics = metric_logger.metrics[-1]
            metric_logger.log_markdown_row(last_metrics)
        
        # Generate plots
        metric_logger.plot_pngs()
