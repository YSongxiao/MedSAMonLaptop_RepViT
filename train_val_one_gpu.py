# %%
import os
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pickle
from pathlib import Path


from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from dataset_cache import NpyDatasetCache, NpyDataset
import cv2
import torch.nn.functional as F
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
                    "-resume",
                    type=str,
                    default='workdir/medsam_lite_latest.pth',
                    help="Path to the checkpoint to continue training."
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
checkpoint = args.resume
makedirs(work_dir, exist_ok=True)


# %%
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))


def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    iou = intersection.float() / union.float()
    
    return iou.unsqueeze(1)


class MedSAM_Lite(nn.Module):
    def __init__(self,
                image_encoder, 
                mask_decoder,
                prompt_encoder
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)[-1]  # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

medsam_lite_image_encoder = MySwinFormer(pretrain_image_size=(256,256),
                                         patch_size=(1, 1), in_chans=3, embed_dim=64,
                                         norm_layer=nn.LayerNorm,
                                         patch_norm=True,
                                         if_absolute_embedding=True).cuda()

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
)

medsam_lite_model = MedSAM_Lite(
    image_encoder=medsam_lite_image_encoder,
    mask_decoder=medsam_lite_mask_decoder,
    prompt_encoder=medsam_lite_prompt_encoder
)

if medsam_lite_checkpoint is not None:
    if isfile(medsam_lite_checkpoint):
        print(f"Finetuning with pretrained weights {medsam_lite_checkpoint}")
        medsam_lite_ckpt = torch.load(
            medsam_lite_checkpoint,
            map_location="cpu"
        )
        medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
    else:
        print(f"Pretained weights {medsam_lite_checkpoint} not found, training from scratch")

medsam_lite_model = medsam_lite_model.to(device)
medsam_lite_model.train()

# %%
print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
# %%
optimizer = optim.AdamW(
    medsam_lite_model.parameters(),
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
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
iou_loss = nn.MSELoss(reduction='mean')
# %%
# train_dataset = NpyDataset(data_root=data_root, data_aug=True)
dataset_cache = NpyDatasetCache(img_data_root=str(Path(args.tr_npy_path) / "imgs"), gt_data_root=str(Path(args.tr_npy_path) / "gts"))
tr_cache, val_cache = dataset_cache.divide()
tr_dataset = NpyDataset(tr_cache, data_aug=args.data_aug)
val_dataset = NpyDataset(val_cache, data_aug=args.data_aug)
train_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if checkpoint and isfile(checkpoint):
    print(f"Resuming from checkpoint {checkpoint}")
    checkpoint = torch.load(checkpoint)
    medsam_lite_model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
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
        logits_pred, iou_pred = medsam_lite_model(image, boxes)
        l_seg = seg_loss(logits_pred, gt2D)
        l_ce = ce_loss(logits_pred, gt2D.float())
        #mask_loss = l_seg + l_ce
        mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
        iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
        l_iou = iou_loss(iou_pred, iou_gt)
        #loss = mask_loss + l_iou
        loss = mask_loss + iou_loss_weight * l_iou
        epoch_train_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"Training Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

    medsam_lite_model.eval()
    pbar = tqdm(val_loader)
    for step, batch in enumerate(pbar):
        with torch.no_grad():
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
            logits_pred, iou_pred = medsam_lite_model(image, boxes)
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            mask_loss = l_seg + l_ce
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            loss = mask_loss + l_iou
            epoch_val_loss[step] = loss.item()
            pbar.set_description(f"Validating Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

    epoch_end_time = time()
    epoch_train_loss_reduced = sum(epoch_train_loss) / len(epoch_train_loss)
    epoch_val_loss_reduced = sum(epoch_val_loss) / len(epoch_val_loss)
    train_losses.append(epoch_train_loss_reduced)
    val_losses.append(epoch_val_loss_reduced)
    lr_scheduler.step(epoch_val_loss_reduced)
    model_weights = medsam_lite_model.state_dict()
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
