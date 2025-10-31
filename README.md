# ResNet50 ImageNet from Scratch (OneCycle + DDP)

Train ResNet-50 on ImageNet-1K from scratch on EC2, targeting 75%+ Top-1 (78% with strong regularization and sufficient schedule). Includes:
- LR Finder and OneCycleLR policy
- Mixed precision (AMP)
- Multi-GPU via torchrun DistributedDataParallel (DDP)
- Single-GPU script for local runs
- Per-epoch Markdown log and PNG plots (loss, accuracy, LR)

## Requirements
- Python 3.8+
- CUDA 11.7+ / CUDA 12.x
- PyTorch 2.0+
- 500GB+ storage for ImageNet dataset
- GPU with 16GB+ VRAM recommended

## Model Architecture
This implementation uses PyTorch's torchvision ResNet50 architecture:
- **Architecture**: ResNet-50 (Deep Residual Learning)
- **Parameters**: ~25.6M trainable parameters
- **Input Size**: 224x224 RGB images
- **Training**: From scratch (no pretrained weights)
- **Output**: 1000 classes (ImageNet-1K)

## 1) EC2 setup

- Recommended instances:
  - 8x A100 40GB: p4d.24xlarge (fastest)
  - 4x A10G 24GB: g5.12xlarge (budget)
  - 4x V100 16GB: p3.8xlarge
- Storage: 500GB+ EBS gp3 or attach S3 bucket with ImageNet.
- AMI: Deep Learning AMI (Ubuntu) GPU PyTorch latest.

### System prep (on the EC2 instance over SSH)
```bash
# Update and install basics
sudo apt-get update -y && sudo apt-get install -y git tmux htop

# Optional: mount EBS at /mnt and create dataset path
sudo mkdir -p /mnt/imagenet && sudo chown $USER:$USER /mnt/imagenet
```

### Clone and environment
```bash
# Original repository
git clone https://github.com/sidrocks/resnet50_Imagenet.git
cd resnet50_Imagenet

# OR use the fork
git clone https://github.com/yasirreshi/Resnet50_Imagenet_Fork.git
cd Resnet50_Imagenet_Fork

# Setup environment
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies** (from `requirements.txt`):
- `torch`, `torchvision` - PyTorch deep learning framework
- `matplotlib`, `pandas` - For plotting and logging
- `tqdm` - Progress bars
- `tensorboard` - Training visualization
- `torch-lr-finder` - Learning rate finder utility
- `numpy` - Numerical computing

## 2) Prepare ImageNet-1K

- Layout expected:
```
<DATA_DIR>/
  train/
    n01440764/
      *.JPEG
    ... (1000 classes)
  val/
    n01440764/
      *.JPEG
    ...
```
- If your validation images are flat with XML annotations, use `organise_val_folder.py` (edit paths inside) to structure `val/`.

## 3) LR Finder

Use a subset (e.g., 10–20k images) to find a stable `max_lr`.
```bash
# Example: data under /mnt/imagenet/ILSVRC/Data/CLS-LOC
python lr_finder.py \
  --data_dir /mnt/imagenet/ILSVRC/Data/CLS-LOC/train \
  --batch_size 128 \
  --workers 2 \
  --start_lr 1e-6 \
  --end_lr 1 \
  --num_iter 500
```
Outputs under `lr_finder_plots/`: PNG plot and `lr_suggestion_*.json` with a heuristic `suggested_max_lr` for OneCycle.

## 4) Single-GPU quick run

```bash
# Minimal run (provide your data root)
python train_single_GPU.py --data-dir /mnt/imagenet/ILSVRC/Data/CLS-LOC --run-name resnet50_single

# Optional overrides (use suggested max_lr from LR finder)
python train_single_GPU.py \
  --data-dir /mnt/imagenet/ILSVRC/Data/CLS-LOC \
  --batch-size 128 \
  --workers 2 \
  --epochs 90 \
  --max-lr 0.097 \
  --pct-start 0.3 \
  --div-factor 25 \
  --final-div-factor 10000 \ 
  --label-smoothing 0.1 \
  --run-name resnet50_single
```
- Edit dataset paths in the script (`training_folder_name` and `val_folder_name`).
- Logs: `logs/<run-name>/training_log.md` and PNGs.

## 5) Multi-GPU DDP training (recommended)

Use torchrun with all GPUs. Example for 8 GPUs (Linux bash):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 train_ddp.py \
  --data-dir /mnt/imagenet/ILSVRC/Data/CLS-LOC \
  --batch-size 256 \
  --workers 8 \
  --epochs 120 \
  --max-lr 0.4 \
  --pct-start 0.3 \
  --div-factor 25 \
  --final-div-factor 10000 \
  --label-smoothing 0.1 \
  --run-name resnet50_from_scratch
```

PowerShell (Windows) example syntax:
```powershell
$env:CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
torchrun --nproc_per_node 8 train_ddp.py --data-dir D:\imagenet --batch-size 256 --workers 8 --epochs 120 --max-lr 0.4 --pct-start 0.3 --div-factor 25 --final-div-factor 10000 --label-smoothing 0.1 --run-name resnet50_from_scratch
```
- Checkpoints: `checkpoints/<run-name>/`
- Logs/plots: `logs/<run-name>/` includes `training_log.md`, `loss.png`, `accuracy.png`, `lr.png`.

### Hyperparameters to reach 75–78% Top-1
- Epochs: 120–150 with cosine anneal via OneCycle tail; warmup `pct_start=0.3`.
- Batch size: 256 per GPU if memory allows; otherwise 128 with gradient accumulation (not included by default).
- Regularization: weight_decay=1e-4, label_smoothing=0.1.
- Augmentations: AutoAugment(IMAGENET). Consider Mixup/CutMix for +0.5–1.0% (not enabled by default here).
- EMA: model EMA can add +0.2–0.4% (not included to keep code lean).

## 6) Resume training
```bash
# Add --resume to continue from last checkpoint
torchrun --nproc_per_node 8 train_ddp.py --data-dir <DATA_DIR> --epochs 120 --max-lr <LR> --resume --run-name resnet50_from_scratch
```

## 7) Monitoring Training Progress

Training progress is automatically logged and visualized in multiple formats:

### Log Files
- **Markdown log**: `logs/<run-name>/training_log.md` - Human-readable table with all metrics
- **JSON log**: `logs/<run-name>/training_log.json` - Machine-readable format for programmatic access
- **TensorBoard logs**: `runs/<run-name>/` - Real-time metrics visualization

### Plots (Auto-generated after each epoch)
The training scripts automatically generate and update PNG plots after each epoch:

- **`loss.png`** - Training and validation loss curves over epochs
  - Shows both train loss (solid line with circles) and validation loss (solid line with squares)
  - Helps identify overfitting (when val loss increases while train loss decreases)

- **`accuracy.png`** - Validation accuracy metrics
  - Top-1 accuracy (most likely prediction is correct)
  - Top-5 accuracy (correct answer in top 5 predictions)
  - Target: 75-78% Top-1, 92-94% Top-5

- **`lr.png`** - Learning rate schedule over epochs
  - Visualizes the OneCycleLR policy
  - Shows warmup phase, peak LR, and annealing phase

**Plots are updated in real-time** as training progresses, allowing you to monitor convergence without waiting for training to complete.

### Using TensorBoard
For real-time monitoring during training:
```bash
# In a separate terminal
tensorboard --logdir runs/<run-name>
# Then open http://localhost:6006 in your browser
```

### Checkpoints
- Saved in: `checkpoints/<run-name>/`
- `checkpoint.pt` - Latest checkpoint (overwritten each epoch)
- `model_000.pt`, `model_001.pt`, ... - Per-epoch checkpoints
- Each checkpoint contains: model state, optimizer state, scheduler state, scaler state, epoch number

## 8) Tips and troubleshooting
- Ensure NCCL env for multi-node or different topologies; for single node usually default works. If issues:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
```
- Data throughput: use `--workers 8` or higher, enable EBS gp3 > 3000 IOPS, consider pre-caching on instance NVMe.
- If OOM: reduce `--batch-size`, or try `--max-lr` lowered proportionally.
- Accuracy below 75%: extend to 150 epochs, add Mixup/CutMix and EMA (see Next Steps section).

## 9) Next steps (optional improvements)
- Add Mixup/CutMix and random erasing.
- Add EMA of weights.
- Add gradient accumulation.
- Add multi-node launch script.

```
Example targets (with 8x A100, 120 epochs, AutoAugment + label smoothing):
- Top-1: 76–78%
- Top-5: 92–94%
Actual results depend on hardware, data throughput, and tuning.
```
## 10) Results on the Mini-Imagenet 100 classes
![Training and Val Loss](https://github.com/user-attachments/assets/4f28946f-eaef-4d8d-9e56-28dbadbcbe58)
![Validation Accuracy](https://github.com/user-attachments/assets/007b5215-3462-4169-87b9-81f9041f4c99)
![LR Schedule](https://github.com/user-attachments/assets/e67f32e9-720b-41df-8227-7afb617dd8f2)

## 11) Results on the Imagenet 1000 classes

Infrastructure: **AWS g4dn.2xlarge EC2**
GPU: **Single Nvidia T4 Tensor core**
Training Time: ~80mins per epoch
Total Training Time: ~ 67 hours for 50 Epochs

**Training Optimizations:**

1. Mixed Precision Training (FP16)
2. One Cycle Learning Rate Policy
3. Learning Rate found using LR Finder
4. Checkpointing
5. Data Loading Optimizations (Pin Memory and multi-worker data loading)
6. Data Augmentations

**Metrics:**
After 50 Epochs Testing - Top 1 Accuracy: 75.27%✅
                          Top 5 Accuracy: 92.56%✅

**Monitoring (Tensor Board Logs)**

<img width="1461" height="585" alt="image" src="https://github.com/user-attachments/assets/033a59dc-12fb-41a8-b8a9-0ef8b8624670" />

<img width="1455" height="553" alt="image" src="https://github.com/user-attachments/assets/25df1935-967c-4b49-b56b-3ac7cd284102" />

<img width="1451" height="536" alt="image" src="https://github.com/user-attachments/assets/30cb9405-e93a-4b7b-8756-f4319ac61d23" />

<img width="1490" height="538" alt="image" src="https://github.com/user-attachments/assets/59d98de8-5671-4e2f-b134-cb59177681cc" />

**Training Logs**

        Using cuda device
        Resuming training from checkpoint
        Testing Epoch 1: 100%|██████████████████████| 295/295 [01:13<00:00,  3.99it/s, loss=2.4969, acc=61.86%, acc5=85.00%]
        Starting training
        Epoch 37
        Epoch 38: 100%|████████████| 7537/7537 [1:18:41<00:00,  1.60it/s, loss=2.6135, acc=57.77%, acc5=80.43%, lr=0.026325]
        Testing Epoch 39: 100%|█████████████████████| 295/295 [01:23<00:00,  3.51it/s, loss=2.4980, acc=61.73%, acc5=84.82%]
        Epoch 38
        Epoch 39: 100%|████████████| 7537/7537 [1:18:41<00:00,  1.60it/s, loss=2.6617, acc=58.62%, acc5=80.95%, lr=0.022473]
        Testing Epoch 40: 100%|█████████████████████| 295/295 [01:23<00:00,  3.54it/s, loss=2.4724, acc=62.23%, acc5=85.21%]
        Epoch 39
        Epoch 40: 100%|████████████| 7537/7537 [1:18:41<00:00,  1.60it/s, loss=2.5698, acc=59.53%, acc5=81.63%, lr=0.018842]
        Testing Epoch 41: 100%|█████████████████████| 295/295 [01:23<00:00,  3.55it/s, loss=2.4198, acc=63.76%, acc5=86.09%]
        Epoch 40
        Epoch 41: 100%|████████████| 7537/7537 [1:18:40<00:00,  1.60it/s, loss=2.5732, acc=60.64%, acc5=82.34%, lr=0.015462]
        Testing Epoch 42: 100%|█████████████████████| 295/295 [01:23<00:00,  3.53it/s, loss=2.3390, acc=65.83%, acc5=87.50%]
        Epoch 41
        Epoch 42: 100%|████████████| 7537/7537 [1:18:40<00:00,  1.60it/s, loss=2.5303, acc=61.89%, acc5=83.20%, lr=0.012360]
        Testing Epoch 43: 100%|█████████████████████| 295/295 [01:23<00:00,  3.52it/s, loss=2.2936, acc=66.57%, acc5=87.81%]
        Epoch 42
        Epoch 43: 100%|████████████| 7537/7537 [1:18:40<00:00,  1.60it/s, loss=2.4994, acc=63.30%, acc5=84.09%, lr=0.009562]
        Testing Epoch 44: 100%|█████████████████████| 295/295 [01:23<00:00,  3.54it/s, loss=2.2563, acc=67.93%, acc5=88.49%]
        Epoch 43
        Epoch 44: 100%|████████████| 7537/7537 [1:18:43<00:00,  1.60it/s, loss=2.4748, acc=64.91%, acc5=85.12%, lr=0.007089]
        Testing Epoch 45: 100%|█████████████████████| 295/295 [01:27<00:00,  3.39it/s, loss=2.1831, acc=69.27%, acc5=89.60%]
        Epoch 44
        Epoch 45: 100%|████████████| 7537/7537 [1:18:43<00:00,  1.60it/s, loss=2.4158, acc=66.65%, acc5=86.22%, lr=0.004961]
        Testing Epoch 46: 100%|█████████████████████| 295/295 [01:27<00:00,  3.36it/s, loss=2.1240, acc=71.05%, acc5=90.40%]
        Epoch 45
        Epoch 46: 100%|████████████| 7537/7537 [1:18:43<00:00,  1.60it/s, loss=1.9967, acc=68.53%, acc5=87.31%, lr=0.003196]
        Testing Epoch 47: 100%|█████████████████████| 295/295 [01:26<00:00,  3.40it/s, loss=2.0639, acc=72.38%, acc5=91.16%]
        Epoch 46
        Epoch 47: 100%|████████████| 7537/7537 [1:18:43<00:00,  1.60it/s, loss=2.2941, acc=70.40%, acc5=88.39%, lr=0.001808]
        Testing Epoch 48: 100%|█████████████████████| 295/295 [01:27<00:00,  3.38it/s, loss=2.0126, acc=73.58%, acc5=91.81%]
        Epoch 47
        Epoch 48: 100%|████████████| 7537/7537 [1:18:43<00:00,  1.60it/s, loss=1.9884, acc=72.13%, acc5=89.36%, lr=0.000808]
        Testing Epoch 49: 100%|█████████████████████| 295/295 [01:28<00:00,  3.33it/s, loss=1.9729, acc=74.65%, acc5=92.31%]
        Epoch 48
        Epoch 49: 100%|████████████| 7537/7537 [1:18:43<00:00,  1.60it/s, loss=1.9985, acc=73.54%, acc5=90.09%, lr=0.000204]
        Testing Epoch 50: 100%|█████████████████████| 295/295 [01:27<00:00,  3.38it/s, loss=1.9535, acc=75.19%, acc5=92.52%]
        Epoch 49
        Epoch 50: 100%|████████████| 7537/7537 [1:18:43<00:00,  1.60it/s, loss=2.1103, acc=74.23%, acc5=90.41%, lr=0.000000]
        Testing Epoch 51: 100%|█████████████████████| 295/295 [01:28<00:00,  3.35it/s, loss=1.9487, acc=75.27%, acc5=92.56%]
        Epoch 50

| Epoch | Train Loss | Val Loss | Top1 (%) | Top5 (%) | LR | Epoch Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5.9304 | 5.1480 | 11.83 | 28.86 | 0.005049 | 4731.90 |
| 3 | 4.8121 | 4.3392 | 23.70 | 47.83 | 0.008150 | 4730.63 |
| 4 | 4.1925 | 4.9917 | 30.03 | 55.09 | 0.013167 | 4731.29 |
| 5 | 3.8249 | 3.6392 | 39.83 | 66.59 | 0.019882 | 4731.14 |
| 6 | 3.5956 | 3.4444 | 40.36 | 67.18 | 0.028000 | 4730.66 |
| 7 | 3.4488 | 3.4023 | 41.38 | 68.25 | 0.037168 | 4731.23 |
| 8 | 3.3504 | 3.1728 | 46.09 | 73.02 | 0.046983 | 4730.95 |
| 9 | 3.2880 | 3.1855 | 46.15 | 72.52 | 0.057018 | 4730.42 |
| 10 | 3.2411 | 3.0102 | 49.91 | 76.08 | 0.066834 | 4728.43 |
| 11 | 3.2048 | 2.9701 | 50.63 | 76.89 | 0.076001 | 4727.40 |
| 12 | 3.1713 | 2.9947 | 50.08 | 76.25 | 0.084119 | 4727.50 |
| 13 | 3.1469 | 2.9866 | 50.22 | 76.58 | 0.090833 | 4726.25 |
| 14 | 3.1251 | 2.9585 | 50.60 | 76.80 | 0.095851 | 4725.51 |
| 15 | 3.1052 | 3.0242 | 49.32 | 75.62 | 0.098951 | 4724.07 |
| 16 | 3.0849 | 2.8874 | 52.76 | 78.57 | 0.100000 | 4724.22 |
| 17 | 3.0705 | 2.8533 | 53.20 | 78.83 | 0.099799 | 4722.69 |
| 18 | 3.0540 | 2.8739 | 52.91 | 78.64 | 0.099196 | 4722.69 |
| 19 | 3.0417 | 3.0003 | 50.06 | 75.94 | 0.098198 | 4722.69 |
| 20 | 3.0267 | 2.8948 | 52.58 | 78.28 | 0.096812 | 4722.03 |
| 21 | 3.0176 | 2.7812 | 54.97 | 80.53 | 0.095048 | 4720.54 |
| 22 | 3.0045 | 2.9654 | 51.30 | 76.77 | 0.092922 | 4721.01 |
| 23 | 2.9928 | 2.7860 | 54.59 | 80.05 | 0.090451 | 10793.27 |
| 24 | 2.9816 | 2.8391 | 53.75 | 78.85 | 0.087653 | 4723.05 |
| 25 | 2.9705 | 2.7419 | 55.77 | 81.15 | 0.084553 | 4720.35 |
| 26 | 2.9603 | 2.7860 | 54.96 | 79.84 | 0.081174 | 4721.55 |
| 27 | 2.9502 | 2.7179 | 56.68 | 81.10 | 0.077544 | 4722.01 |
| 28 | 2.9348 | 2.7254 | 56.45 | 81.10 | 0.073693 | 4722.36 |
| 29 | 2.9242 | 2.6967 | 57.12 | 81.59 | 0.069651 | 4722.90 |
| 30 | 2.9092 | 2.6912 | 57.09 | 82.01 | 0.065450 | 4721.16 |
| 31 | 2.8947 | 2.6462 | 58.14 | 82.40 | 0.061126 | 4720.72 |
| 32 | 2.8791 | 2.6926 | 57.05 | 81.66 | 0.056711 | 5559.07 |
| 33 | 2.8616 | 2.6183 | 58.60 | 83.01 | 0.052243 | 7762.10 |
| 34 | 2.8422 | 2.5799 | 59.98 | 83.52 | 0.047756 | 4720.52 |
| 35 | 2.8205 | 2.6183 | 58.86 | 82.99 | 0.043288 | 4719.00 |
| 36 | 2.7980 | 2.5721 | 59.98 | 83.55 | 0.038874 | 4719.22 |
| 37 | 2.7717 | 2.5382 | 60.74 | 84.36 | 0.034549 | 4689.59 |
| 38 | 2.7442 | 2.4969 | 61.86 | 85.00 | 0.030348 | 4689.74 |
| 39 | 2.7096 | 2.4980 | 61.73 | 84.82 | 0.026306 | 4721.79 |
| 40 | 2.6758 | 2.4724 | 62.23 | 85.21 | 0.022455 | 4721.88 |
| 41 | 2.6340 | 2.4198 | 63.76 | 86.09 | 0.018825 | 4721.40 |
| 42 | 2.5905 | 2.3390 | 65.83 | 87.50 | 0.015447 | 4720.90 |
| 43 | 2.5364 | 2.2936 | 66.57 | 87.81 | 0.012346 | 4721.13 |
| 44 | 2.4780 | 2.2563 | 67.93 | 88.49 | 0.009549 | 4721.21 |
| 45 | 2.4110 | 2.1831 | 69.27 | 89.60 | 0.007078 | 4723.30 |
| 46 | 2.3401 | 2.1240 | 71.05 | 90.40 | 0.004952 | 4723.96 |
| 47 | 2.2648 | 2.0639 | 72.38 | 91.16 | 0.003188 | 4724.12 |
| 48 | 2.1890 | 2.0126 | 73.58 | 91.81 | 0.001802 | 4723.51 |
| 49 | 2.1189 | 1.9729 | 74.65 | 92.31 | 0.000804 | 4724.14 |
| 50 | 2.0647 | 1.9535 | 75.19 | 92.52 | 0.000202 | 4724.03 |
| 51 | 2.0399 | 1.9487 | 75.27 | 92.56 | 0.000000 | 4723.75 
