# %%
import os
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from tqdm import tqdm, trange
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
from timm.models import create_model
from dataset_cache import NpyDatasetCache, NpyDataset
from swin.Transformer.SwinTransformer.swin_lzh import MySwinFormer
from matplotlib import pyplot as plt
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--tr_npy_path',
                    type=str,
                    default='data/npy',
                    help='Path to training npy files; two subfolders: gts and imgs')
parser.add_argument('-task_name',
                    type=str,
                    default='MedSAM-Lite-RepViT')
parser.add_argument(
                    "-pretrained_checkpoint",
                    type=str,
                    default="lite_medsam.pth",
                    help="Path to the pretrained Lite-MedSAM checkpoint."
)
parser.add_argument(
                    '-tch_pretrained_checkpoint',
                    type=str,
                    default='',
                    help="Teacher checkpoint"
)
parser.add_argument(
                    "-work_dir",
                    type=str,
                    default="./workdir/",
                    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument('--data_aug',
                    action='store_true',
                    default=False,
                    help='use data augmentation during training')
parser.add_argument(
                    "-num_epochs",
                    type=int,
                    default=100,
                    help="Number of epochs to train."
)
parser.add_argument(
                    "-batch_size",
                    type=int,
                    default=4,
                    help="Batch size."
)
parser.add_argument(
                    "-num_workers",
                    type=int,
                    default=4,
                    help="Number of workers for dataloader."
)
parser.add_argument(
                    "-device",
                    type=str,
                    efault="cuda:0",
                    help="Device to train on."
)
parser.add_argument(
                    '-weight_decay',
                    type=float,
                    default=0.01,
                    help='weight decay (default: 0.01)'
)
parser.add_argument(
                    '-lr',
                    type=float,
                    default=0.0001,
                    metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument(
                    "-bbox_shift",
                    type=int,
                    default=5,
                    help="Perturbation to bounding box coordinates during training."
)
parser.add_argument(
                    "-weight_decay",
                    type=float,
                    default=0.01,
                    help="Weight decay."
)
parser.add_argument(
                    "-iou_loss_weight",
                    type=float,
                    default=1.0,
                    help="Weight of IoU loss."
)
parser.add_argument(
                    "-seg_loss_weight",
                    type=float,
                    default=1.0,
                    help="Weight of segmentation loss."
)
parser.add_argument(
                    "-ce_loss_weight",
                    type=float,
                    default=1.0,
                    help="Weight of cross entropy loss."
)


args = parser.parse_args()
# %%
work_dir = args.work_dir
tr_npy_path = args.tr_npy_path
medsam_lite_checkpoint = args.pretrained_checkpoint
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
device = args.device
bbox_shift = args.bbox_shift
lr = args.lr
weight_decay = args.weight_decay
iou_loss_weight = args.iou_loss_weight
seg_loss_weight = args.seg_loss_weight
ce_loss_weight = args.ce_loss_weight
tch_pretrained_checkpoint = args.tch_pretrained_checkpoint
makedirs(work_dir, exist_ok=True)

# %%
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


def load_partial_state_dict(model, checkpoint_path):
    # Load the checkpoint from the specified path
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get the current model's state dict
    current_state_dict = model.state_dict()

    # Filter the loaded checkpoint state dict to only include keys that exist in the current model
    filtered_state_dict = {k: v for k, v in checkpoint.items() if k in current_state_dict}

    # Update the current model's state dict with the filtered checkpoint values
    current_state_dict.update(filtered_state_dict)

    # Load the updated state dict back into the model
    model.load_state_dict(current_state_dict)

    # Print the keys that were loaded
    # print("Warning: Only part of models are loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Loaded parts:", list(filtered_state_dict.keys()))

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

swin_encoder = MySwinFormer(pretrain_image_size=(256,256),
                            patch_size=(1, 1), in_chans=3, embed_dim=64,
                            norm_layer=nn.LayerNorm,
                            patch_norm=True,
                            if_absolute_embedding=True).cuda()

medsam_lite_image_encoder = create_model('repvit')

if isfile(args.tch_pretrained_checkpoint):
    # With partial loading, the medsam_lite_model and swin_encoder are both loaded with the available parts in checkpoint.
    load_partial_state_dict(swin_encoder, args.tch_pretrained_checkpoint)

medsam_lite_image_encoder = medsam_lite_image_encoder.to(device)
medsam_lite_image_encoder.train()
swin_encoder = swin_encoder.to(device)
swin_encoder.eval()
freeze_model(swin_encoder)

# %%
print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_image_encoder.parameters())}")
# %%
optimizer = optim.AdamW(
    medsam_lite_image_encoder.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.9,
    patience=5,
    cooldown=0
)
distill_mse_loss = nn.MSELoss()
# %%
dataset_cache = NpyDatasetCache(img_data_root=str(Path(args.tr_npy_path) / "imgs"), gt_data_root=str(Path(args.tr_npy_path) / "gts"))
tr_cache, val_cache = dataset_cache.divide()
tr_dataset = NpyDataset(tr_cache, data_aug=args.data_aug)
val_dataset = NpyDataset(val_cache, data_aug=args.data_aug)
train_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

start_epoch = 0
best_loss = 1e10
# %%
train_losses = []
val_losses = []
epoch_times = []
for epoch in range(start_epoch + 1, num_epochs):
    epoch_train_loss = [1e10 for _ in range(len(train_loader))]
    epoch_val_loss = [1e10 for _ in range(len(val_loader))]
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        image = batch["image"]
        gt2D = batch["gt2D"]
        boxes = batch["bboxes"]
        optimizer.zero_grad()
        image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)

        # current model inference
        image_embedding = medsam_lite_image_encoder(image)
        # teacher model inference
        with torch.no_grad():
            teacher_embedding = swin_encoder(image)[-1]

        # Add the distill loss
        l_distill = distill_mse_loss(image_embedding, teacher_embedding)
        loss = l_distill

        epoch_train_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"Training Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

    medsam_lite_image_encoder.eval()
    pbar = tqdm(val_loader)
    for step, batch in enumerate(pbar):
        with torch.no_grad():
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)

            # current model inference
            image_embedding = medsam_lite_image_encoder(image)
            # teacher model inference
            with torch.no_grad():
                teacher_embedding = swin_encoder(image)[-1]

            # Add the distill loss
            l_distill = distill_mse_loss(image_embedding, teacher_embedding)
            loss = l_distill

            epoch_val_loss[step] = loss.item()
            pbar.set_description(f"Validating Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

    epoch_end_time = time()
    epoch_train_loss_reduced = sum(epoch_train_loss) / len(epoch_train_loss)
    epoch_val_loss_reduced = sum(epoch_val_loss) / len(epoch_val_loss)
    train_losses.append(epoch_train_loss_reduced)
    val_losses.append(epoch_val_loss_reduced)
    lr_scheduler.step(epoch_val_loss_reduced)
    model_weights = medsam_lite_image_encoder.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_train_loss_reduced,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, join(work_dir, "medsam_lite_latest.pth"))
    if epoch_val_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_val_loss_reduced:.4f}")
        best_loss = epoch_val_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "medsam_lite_best.pth"))

    # %% plot loss
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].title.set_text("Dice + Binary Cross Entropy + IoU Loss")
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses, label="Val")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epoch_times)
    axes[1].title.set_text("Epoch Duration")
    axes[1].set_ylabel("Duration (s)")
    axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(join(work_dir, "log.png"))
    plt.close()
