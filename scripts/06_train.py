import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset
from pathlib import Path

# add BWTreeNet to path
sys.path.insert(0, os.path.expanduser("~/BWTreeNet/GuiTest"))

from model.BWTreeNet import BWTreeNet
from loss.lossfunction import IoULoss
from util.Forest_dataset import Forest_dataset
from util.augmentation import (RandomFlip, RandomBrightness, RandomNoise,
                                RandomScratch, RandomBlur, RandomDistortion,
                                RandomRoll)

# ── config ─────────────────────────────────────────────────────────────────
IMAGES_DIRS = [
    os.path.expanduser("~/bw_treenet/data/processed/malmo/tiles/images/"),
    os.path.expanduser("~/bw_treenet/data/processed/gtb/tiles/images/"),
]
LABELS_DIRS = [
    os.path.expanduser("~/bw_treenet/data/processed/malmo/tiles/labels/"),
    os.path.expanduser("~/bw_treenet/data/processed/gtb/tiles/labels/"),
]
WEIGHTS_DIR = os.path.expanduser("~/bw_treenet/models/")
LOG_FILE    = os.path.expanduser("~/bw_treenet/results/training_log_v2.csv")
PRETRAINED  = os.path.expanduser(
    "~/bw_treenet/models/BWTreeNet_SwissHistorical_1980s.pth")

Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.expanduser("~/bw_treenet/results")).mkdir(parents=True, exist_ok=True)

EPOCHS      = 100
BATCH_SIZE  = 1
LR          = 0.001
VAL_SPLIT   = 0.2
N_CLASSES   = 2
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_EVERY  = 10

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(f"Using device: {DEVICE}")

# ── augmentation ───────────────────────────────────────────────────────────
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomBrightness(bright_range=0.15, prob=0.5),
    RandomNoise(noise_range=10, prob=0.3),
    RandomBlur(prob=0.3),
    RandomDistortion(prob=0.3),
    RandomScratch(prob=0.3),
    RandomRoll(prob=0.3),
]

# ── dataset ────────────────────────────────────────────────────────────────
datasets = []
for img_dir, lbl_dir in zip(IMAGES_DIRS, LABELS_DIRS):
    ds = Forest_dataset(
        map_dir=img_dir,
        map_seffix=".tif",
        label_dir=lbl_dir,
        label_seffix=".tif",
        have_label=True,
        class_num=N_CLASSES,
        input_h=1000,
        input_w=1000,
        transform=augmentation_methods
    )
    datasets.append(ds)

full_dataset = ConcatDataset(datasets)
print(f"Total dataset size: {len(full_dataset)}")

n_val   = int(len(full_dataset) * VAL_SPLIT)
n_train = len(full_dataset) - n_val
train_set, val_set = random_split(full_dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {n_train}  Val: {n_val}")

# ── model ──────────────────────────────────────────────────────────────────
model = BWTreeNet(n_class=N_CLASSES).to(DEVICE)

if os.path.exists(PRETRAINED):
    state = torch.load(PRETRAINED, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained weights from {PRETRAINED}")
else:
    print("No pretrained weights found, training from scratch")

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS)

# ── logging ────────────────────────────────────────────────────────────────
with open(LOG_FILE, "w") as f:
    f.write("epoch,train_loss,val_loss,val_iou\n")

# ── training loop ──────────────────────────────────────────────────────────
best_val_iou = 0.0

for epoch in range(1, EPOCHS + 1):
    # ── train ──────────────────────────────────────────────────────────────
    model.train()
    train_losses = []

    for images, labels, _ in train_loader:
        images = images.to(DEVICE, dtype=torch.float32)
        labels = labels.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)

        loss_fn = IoULoss(outputs, labels)
        loss = loss_fn.loss_function()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    scheduler.step()
    avg_train_loss = np.mean(train_losses)

    # ── validate ───────────────────────────────────────────────────────────
    model.eval()
    val_losses  = []
    iou_scores  = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.long)

            outputs = model(images)

            loss_fn = IoULoss(outputs, labels)
            val_losses.append(loss_fn.loss_function().item())

            preds = outputs.argmax(dim=1)
            inter = ((preds == 1) & (labels == 1)).sum().item()
            union = ((preds == 1) | (labels == 1)).sum().item()
            if union > 0:
                iou_scores.append(inter / union)

    avg_val_loss = np.mean(val_losses)
    avg_val_iou  = np.mean(iou_scores) if iou_scores else 0.0

    print(f"Epoch {epoch:03d}/{EPOCHS}  "
          f"train_loss={avg_train_loss:.4f}  "
          f"val_loss={avg_val_loss:.4f}  "
          f"val_iou={avg_val_iou:.4f}")

    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_train_loss:.4f},"
                f"{avg_val_loss:.4f},{avg_val_iou:.4f}\n")

    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save(model.state_dict(),
                   os.path.join(WEIGHTS_DIR, "bwtreenet_malmo_gtb_best.pt"))
        print(f"  --> saved best model (val_iou={best_val_iou:.4f})")

    if epoch % SAVE_EVERY == 0:
        torch.save(model.state_dict(),
                   os.path.join(WEIGHTS_DIR,
                                f"bwtreenet_malmo_gtb_epoch{epoch}.pt"))

print(f"\nTraining complete. Best val IoU: {best_val_iou:.4f}")
print(f"Log saved to {LOG_FILE}")