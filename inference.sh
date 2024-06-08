#!/bin/bash
python3 CVPR24_LiteMedSAM_infer_repvit.py -i validation/imgs/ -o validation/preds/ -device cpu -lite_medsam_ckpt_path path/to/teacher_model -encoder_ckpt_path path/to/distilled_encoder
