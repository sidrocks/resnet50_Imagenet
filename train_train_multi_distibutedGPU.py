import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
from math import sqrt

from model import ResNet50
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


import signal
import sys

import torch.distributed as dist
# dist.destroy_process_group()

# Add this after the imports
def cleanup_stale_processes():
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass

    # Clear any processes using the port
    os.system('fuser -k 12355/tcp')

def signal_handler(sig, frame):
    print("Cleaning up...")
    dist.destroy_process_group()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class Params:
    def __init__(self):
        self.batch_size = 256
        self.name = "resnet_50_onecycle_distributed"
        self.workers = 12
        self.max_lr = 0.175
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 50
        self.pct_start = 0.3
        self.div_factor = 25.0
        self.final_div_factor = 1e4

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class MetricLogger:
    def __init__(self, log_dir, rank):
        self.log_dir = log_dir
        self.rank = rank
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
        self.metrics = []
        
    def log_metrics(self, epoch_metrics):
        if self.rank == 0:  # Only log metrics on main process
            self.metrics.append(epoch_metrics)
            with open(os.path.join(self.log_dir, 'training_log.json'), 'w') as f:
                json.dump(self.metrics, f, indent=4)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, dataloader, model, loss_fn, optimizer, scheduler, epoch, writer, scaler, metric_logger):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    if rank == 0:
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    else:
        progress_bar = enumerate(dataloader)

    for batch, (X, y) in progress_bar:
        X, y = X.cuda(rank), y.cuda(rank)

        with autocast(device_type='cuda'):
            pred = model(X)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        scheduler.step()

        running_loss += loss.item()
        total += y.size(0)

        _, predicted = torch.max(pred.data, 1)
        correct += (predicted == y).sum().item()

        _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
        correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

        if rank == 0 and batch % 100 == 0:
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            current_acc = 100 * correct / total
            current_acc5 = 100 * correct_top5 / total

            if isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.2f}%",
                    "acc5": f"{current_acc5:.2f}%",
                    "lr": f"{current_lr:.6f}"
                })

            if writer is not None:
                step = epoch * size + (batch + 1) * dataloader.batch_size
                writer.add_scalar('training loss', current_loss, step)
                writer.add_scalar('training accuracy', current_acc, step)
                writer.add_scalar('training top5 accuracy', current_acc5, step)
                writer.add_scalar('learning rate', current_lr, step)

    # Gather metrics from all processes
    world_size = dist.get_world_size()
    running_loss_tensor = torch.tensor([running_loss]).cuda(rank)
    correct_tensor = torch.tensor([correct]).cuda(rank)
    correct_top5_tensor = torch.tensor([correct_top5]).cuda(rank)
    total_tensor = torch.tensor([total]).cuda(rank)

    dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_top5_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    epoch_time = time.time() - start0
    avg_loss = running_loss_tensor.item() / (len(dataloader) * world_size)
    accuracy = 100 * correct_tensor.item() / total_tensor.item()
    accuracy_top5 = 100 * correct_top5_tensor.item() / total_tensor.item()

    if rank == 0:
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

def test(rank, dataloader, model, loss_fn, epoch, writer, train_dataloader, metric_logger, calc_acc5=True):
    model.eval()
    test_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0

    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Testing Epoch {epoch+1}")
    else:
        progress_bar = dataloader

    with torch.no_grad():
        with autocast(device_type='cuda'):
            for X, y in progress_bar:
                X, y = X.cuda(rank), y.cuda(rank)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                total += y.size(0)
                
                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == y).sum().item()
                
                if calc_acc5:
                    _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                    correct_top5 += pred_top5.eq(y.view(-1, 1).expand_as(pred_top5)).sum().item()

    # Gather metrics from all processes
    world_size = dist.get_world_size()
    test_loss_tensor = torch.tensor([test_loss]).cuda(rank)
    correct_tensor = torch.tensor([correct]).cuda(rank)
    correct_top5_tensor = torch.tensor([correct_top5]).cuda(rank)
    total_tensor = torch.tensor([total]).cuda(rank)

    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_top5_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = test_loss_tensor.item() / (len(dataloader) * world_size)
        accuracy = 100 * correct_tensor.item() / total_tensor.item()
        accuracy_top5 = 100 * correct_top5_tensor.item() / total_tensor.item() if calc_acc5 else None

        metrics = {
            'epoch': epoch + 1,
            'test_loss': test_loss,
            'test_accuracy': accuracy,
            'test_accuracy_top5': accuracy_top5
        }
        metric_logger.log_metrics(metrics)

        if writer is not None:
            step = epoch * len(train_dataloader.dataset)
            writer.add_scalar('test loss', test_loss, step)
            writer.add_scalar('test accuracy', accuracy, step)
            if calc_acc5:
                writer.add_scalar('test accuracy5', accuracy_top5, step)

        logger.info(f"Test Epoch {epoch+1} - Loss: {test_loss:.4f}, Acc: {accuracy:.2f}%, Top-5 Acc: {accuracy_top5:.2f}%")

def main_worker(rank, world_size, params):
    try:
        setup(rank, world_size)

        # create required directories at the start (only on rank 0)
        if rank ==0:
            os.makedirs(os.path.join("checkpoints", params.name), exist_ok=True)
            os.makedirs(os.path.join("logs", params.name), exist_ok=True)
            os.makedirs(os.path.join("runs", params.name), exist_ok=True)

        # Wait for rank 0 to create directories
        dist.barrier()

    
        # Create metric logger
        log_dir = os.path.join("logs", params.name)
        metric_logger = MetricLogger(log_dir, rank)

        training_folder_name = '/home/ubuntu/data/Imagenet-dataset/train'
        val_folder_name = '/home/ubuntu/data/Imagenet-dataset/val'

        train_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
            ])

        train_dataset = torchvision.datasets.ImageFolder(
            root=training_folder_name,
            transform=train_transformation
            )
    
        # Use DistributedSampler
        train_sampler = DistributedSampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            sampler=train_sampler,
            num_workers=params.workers,
            pin_memory=True
            )

        val_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
        val_dataset = torchvision.datasets.ImageFolder(
            root=val_folder_name,
            transform=val_transformation
            )

        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
        val_loader = DataLoader(
            val_dataset,
            batch_size=params.batch_size,
            sampler=val_sampler,
            num_workers=params.workers,
            pin_memory=True
            )

        # Wrap your model in DDP after moving to GPU
        torch.cuda.set_device(rank)

        num_classes = len(train_dataset.classes)
        model = ResNet50(num_classes=num_classes).cuda(rank)
        model = DDP(model, device_ids=[rank])

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), 
                               lr=params.max_lr/params.div_factor,
                               momentum=params.momentum,
                               weight_decay=params.weight_decay)

        scaler = GradScaler('cuda')

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
    
        # Modify checkpoint loading to handle device mapping
        if os.path.exists(checkpoint_path):
            print(f"Rank {rank}: Resuming training from checkpoint")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.module.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            assert params == checkpoint["params"]

        writer = SummaryWriter('runs/' + params.name) if rank == 0 else None
    
        test(rank, val_loader, model, loss_fn, epoch=0, writer=writer, train_dataloader=train_loader, 
             metric_logger=metric_logger, calc_acc5=True)
    
        if rank == 0:
            print("Starting training")
    
        for epoch in range(start_epoch, params.epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        
            if rank == 0:
                print(f"Epoch {epoch}")
            
            train(rank, train_loader, model, loss_fn, optimizer, scheduler, epoch=epoch, writer=writer, 
                scaler=scaler, metric_logger=metric_logger)
        
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "params": params
                    }
                torch.save(checkpoint, os.path.join("checkpoints", params.name, f"model_{epoch}.pth"))
                torch.save(checkpoint, os.path.join("checkpoints", params.name, f"checkpoint.pth"))
        
            test(rank, val_loader, model, loss_fn, epoch + 1, writer, train_dataloader=train_loader,
                 metric_logger=metric_logger, calc_acc5=True)
    
    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        raise e
    finally:
        cleanup()

if __name__ == "__main__":
    cleanup_stale_processes()  # Delete existing stale processes
    params = Params()
    
    world_size = torch.cuda.device_count()
    try:
        mp.spawn(main_worker,
                args=(world_size, params),
                nprocs=world_size,
                join=True)
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        # Ensure all processes are terminated
        cleanup_stale_processes()
        import sys
        sys.exit(1)
