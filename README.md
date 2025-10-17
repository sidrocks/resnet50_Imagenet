# ResNet50 ImageNet from Scratch (OneCycle + DDP)

Train ResNet-50 on ImageNet-1K from scratch on EC2, targeting 75%+ Top-1 (78% with strong regularization and sufficient schedule). Includes:
- LR Finder and OneCycleLR policy
- Mixed precision (AMP)
- Multi-GPU via torchrun DistributedDataParallel (DDP)
- Single-GPU script for local runs
- Per-epoch Markdown log and PNG plots (loss, accuracy, LR)

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
git clone https://github.com/sidrocks/resnet50_Imagenet.git
cd resnet50_Imagenet
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

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
  --batch_size 512 \
  --workers 8 \
  --start_lr 1e-6 \
  --end_lr 1 \
  --num_iter 300
```
Outputs under `lr_finder_plots/`: PNG plot and `lr_suggestion_*.json` with a heuristic `suggested_max_lr` for OneCycle.

## 4) Single-GPU quick run

```bash
# Minimal run (provide your data root)
python train_single_GPU.py --data-dir /mnt/imagenet/ILSVRC/Data/CLS-LOC --run-name resnet50_single

# Optional overrides (use suggested max_lr from LR finder)
python train_single_GPU.py \
  --data-dir /mnt/imagenet/ILSVRC/Data/CLS-LOC \
  --batch-size 256 \
  --workers 8 \
  --epochs 90 \
  --max-lr 0.3 \
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

## 7) Deliverables
- Markdown log: `logs/<run-name>/training_log.md` contains every epoch from 1..N.
- PNG plots: `loss.png`, `accuracy.png`, `lr.png` saved alongside.
- Checkpoints: under `checkpoints/<run-name>/`.

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