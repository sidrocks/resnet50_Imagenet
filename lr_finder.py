from torch_lr_finder import LRFinder
import torch
import torchvision
import torchvision.transforms as transforms
from model import ResNet50
import torch.nn as nn
import torch.optim as optim
from train import Params
import fire
import matplotlib.pyplot as plt
from datetime import datetime
import os

def find_lr(start_lr=1e-7, end_lr=10, num_iter=100, output_dir='lr_finder_plots'):
    params = Params()
    print(f"Find LR with params: Start_lr: {start_lr}, End_lr: {end_lr}, Num_iter: {num_iter}")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    
    training_folder_name = '/mnt/imagenet/ILSVRC/Data/CLS-LOC/train'
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.workers,
        pin_memory=True
    )

    model = ResNet50(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=params.momentum, weight_decay=params.weight_decay)

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp and parameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'lr_finder_{timestamp}_start{start_lr}_end{end_lr}_iter{num_iter}.png'
    filepath = os.path.join(output_dir, filename)
    
    # Plot and save
    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)
    plt.title(f'Learning Rate Finder (iter: {num_iter})')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {filepath}")
    lr_finder.reset()

if __name__ == "__main__":
    fire.Fire(find_lr)
