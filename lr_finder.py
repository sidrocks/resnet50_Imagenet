from torch_lr_finder import LRFinder
import torch
import torchvision
import torchvision.transforms as transforms
from model import ResNet50
import torch.nn as nn
import torch.optim as optim
import fire
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

def find_lr(
    data_dir: str = '/mnt/imagenet/ILSVRC/Data/CLS-LOC/train',
    batch_size: int = 256,
    workers: int = 8,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_iter: int = 200,
    output_dir: str = 'lr_finder_plots'
):
    print(f"Find LR: data_dir={data_dir}, start_lr={start_lr}, end_lr={end_lr}, num_iter={num_iter}")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    
    training_folder_name = data_dir
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    model = ResNet50(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=1e-4)

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

    # Suggest a max_lr based on min-loss/10 heuristic
    history = lr_finder.history
    lrs = history.get('lr', [])
    losses = history.get('loss', [])
    suggested = None
    if isinstance(lrs, list) and isinstance(losses, list) and len(losses) > 0:
        min_idx = min(range(len(losses)), key=lambda i: losses[i])
        if min_idx is not None and min_idx < len(lrs):
            suggested = max(lrs[min_idx] / 10.0, start_lr)
    suggestion_path = os.path.join(output_dir, f'lr_suggestion_{timestamp}.json')
    with open(suggestion_path, 'w') as f:
        json.dump({
            'suggested_max_lr': suggested,
            'min_loss': losses[min_idx] if suggested is not None else None,
            'min_loss_lr': lrs[min_idx] if suggested is not None else None,
            'start_lr': start_lr,
            'end_lr': end_lr,
            'num_iter': num_iter
        }, f, indent=2)

    print(f"Plot saved to: {filepath}")
    if suggested is not None:
        print(f"Suggested max_lr: {suggested:.6f} (saved to {suggestion_path})")
    lr_finder.reset()

if __name__ == "__main__":
    fire.Fire(find_lr)
