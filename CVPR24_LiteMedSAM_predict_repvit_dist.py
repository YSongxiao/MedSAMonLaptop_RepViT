from os import makedirs
from os.path import join, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.fx import symbolic_trace
import cv2
import argparse
from collections import OrderedDict
from datetime import datetime
# import repvit

# %% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='/mnt/data2/datasx/MedSAM/validation/time_eval_input/',
    # required=True,
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='/mnt/data2/datasx/MedSAM/validation/time_eval_output/',
    help='directory to save the prediction',
)
parser.add_argument(
    '-lite_medsam_ckpt_path',
    type=str,
    default="/mnt/data1/songxiao/MedSAM-LiteMedSAM/work_dir/20240327/medsam_lite_best_extracted.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-encoder_ckpt_path',
    type=str,
    default="/mnt/data1/songxiao/MedSAM-LiteMedSAM/work_dir/20240508_dist_mini_repvit_tinyswin/medsam_lite_minirepvit_encoder_best_19.pth",
    help='path to the checkpoint of Image Encoder',
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)

args = parser.parse_args()
data_root = args.input_dir
pred_save_dir = args.output_dir
makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256


def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3:  ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:  ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded


class MedSAM_Lite_NoEncoder(nn.Module):
    def __init__(
            self,
            mask_decoder,
            prompt_encoder
    ):
        super().__init__()
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image_embedding, box_np):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        low_res_pred = torch.sigmoid(low_res_masks)

        return low_res_pred


def postprocess_masks(masks, new_size, original_size):
    """
    Do cropping and resizing

    Parameters
    ----------
    masks : torch.Tensor
        masks predicted by the model
    new_size : tuple
        the shape of the image after resizing to the longest side of 256
    original_size : tuple
        the original shape of the image

    Returns
    -------
    torch.Tensor
        the upsampled mask to the original size
    """
    # Crop
    masks = crop_mask_fx(masks, new_size)
    # Resize
    masks = F.interpolate(
        masks,
        size=(original_size[0], original_size[1]),
        mode="bilinear",
        align_corners=False,
    )

    return masks

@torch.fx.wrap
def crop_mask_fx(masks, new_size):
    masks = masks[..., :new_size[0], :new_size[1]]
    return masks


def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256


def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box


@torch.no_grad()
def medsam_inference_with_embed(medsam_model, img_embed, box_256, new_size, original_size):
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)

    low_res_pred = medsam_model(img_embed, box_torch)

    low_res_pred = postprocess_masks(low_res_pred, new_size, original_size)

    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg


def load_states_split(medsam_encoder, medsam_lite_model, repvit_ckpt_path, decoder_ckpt_path):
    repvit_ckpt = torch.load(repvit_ckpt_path, map_location='cpu')
    decoder_ckpt = torch.load(decoder_ckpt_path, map_location='cpu')
    medsam_encoder.load_state_dict(repvit_ckpt["model"])

    prompt_state_dict = medsam_lite_model.prompt_encoder.state_dict()
    filtered_prompt_state_dict = {k.replace("prompt_encoder.", ""): v for k, v in decoder_ckpt.items() if
                                  k.replace("prompt_encoder.", "") in prompt_state_dict}
    prompt_state_dict.update(filtered_prompt_state_dict)
    medsam_lite_model.prompt_encoder.load_state_dict(prompt_state_dict)

    mask_state_dict = medsam_lite_model.mask_decoder.state_dict()
    filtered_mask_state_dict = {k.replace("mask_decoder.", ""): v for k, v in decoder_ckpt.items() if
                                k.replace("mask_decoder.", "") in mask_state_dict}
    mask_state_dict.update(filtered_mask_state_dict)
    medsam_lite_model.mask_decoder.load_state_dict(mask_state_dict)
    return medsam_encoder, medsam_lite_model


def fuse_model(model):
    graph_module = torch.fx.symbolic_trace(model)
    for node in list(graph_module.graph.nodes):
        if node.op == "call_module":
            target_module = graph_module.get_submodule(node.target)
            if isinstance(target_module, nn.BatchNorm2d):
                prev_node = node.args[0]
                if prev_node.op == "call_module":
                    prev_module = graph_module.get_submodule(prev_node.target)
                    if isinstance(prev_module, nn.Conv2d):
                        # Fuse Conv and BN
                        fused_conv = fuse_conv_bn_eval(prev_module, target_module)
                        setattr(graph_module, prev_node.target, fused_conv)
                        node.replace_all_uses_with(prev_node)
                        graph_module.graph.erase_node(node)
    graph_module.recompile()
    return graph_module


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
medsam_lite_model = MedSAM_Lite_NoEncoder(
    mask_decoder=medsam_lite_mask_decoder,
    prompt_encoder=medsam_lite_prompt_encoder
)

medsam_lite_image_encoder = torch.jit.load(args.encoder_ckpt_path)
medsam_lite_model.load_state_dict(torch.load(args.lite_medsam_ckpt_path))
medsam_lite_image_encoder.to(device)
medsam_lite_image_encoder.eval()
medsam_lite_model.to(device)
medsam_lite_model.eval()


def MedSAM_infer_npz_2D(medsam_lite_model, img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)  # (H, W, 3)
    img_3c = npz_data['imgs']  # (H, W, 3)
    assert np.max(img_3c) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    img_embedding = medsam_lite_image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes, start=1):
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...]  # (1, 4)
        medsam_mask = medsam_inference_with_embed(medsam_lite_model, img_embedding, box256, (newh, neww), (H, W))
        segs[medsam_mask > 0] = idx
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')

    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )


def MedSAM_infer_npz_3D(medsam_lite_model, img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs']  # (D, H, W)
    spacing = npz_data['spacing']  # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8)
    boxes_3D = npz_data['boxes']  # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min) / 2 + z_min)

        # infer from middle slice to the z_max
        z_max = min(z_max+1, img_3D.shape[0])
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # Pad image to 256x256
            img_256 = pad_image(img_256)

            # convert the shape to (3, H, W)
            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                img_embedding = medsam_lite_image_encoder(img_256_tensor)
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                pre_seg256 = resize_longest_side(pre_seg)
                if np.max(pre_seg256) > 0:
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg = medsam_inference_with_embed(medsam_lite_model, img_embedding, box_256, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg > 0] = idx

        # infer from middle slice to the z_max
        z_min = max(0, z_min - 1)
        for z in range(z_middle - 1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # Pad image to 256x256
            img_256 = pad_image(img_256)

            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                img_embedding = medsam_lite_image_encoder(img_256_tensor)

            pre_seg = segs_3d_temp[z + 1, :, :]
            pre_seg256 = resize_longest_side(pre_seg)
            if np.max(pre_seg256) > 0:
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg = medsam_inference_with_embed(medsam_lite_model, img_embedding, box_256, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg > 0] = idx
        segs[segs_3d_temp > 0] = idx
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )


if __name__ == '__main__':
    img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
    infered_npz_files = sorted(glob(join(pred_save_dir, '*.npz'), recursive=True))
    infered_npz_files = [basename(i) for i in infered_npz_files]
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    dummy_img_tensor = torch.ones((1, 1, 256, 256))
    dummy_box256 = np.array([0, 0, 255, 255])
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if basename(img_npz_file).startswith('3D'):
            MedSAM_infer_npz_3D(medsam_lite_model, img_npz_file)
        else:
            MedSAM_infer_npz_2D(medsam_lite_model, img_npz_file)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    # efficiency_df = pd.DataFrame(efficiency)
    # efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
