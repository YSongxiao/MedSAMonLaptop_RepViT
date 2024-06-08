#!/bin/bash
#/mnt/data1/datasx/MedSAM/validation/imgs
python3 CVPR24_LiteMedSAM_predict_repvit_dist.py -i /workspace/inputs/ -o /workspace/outputs/ -lite_medsam_ckpt_path ./work_dir/other_part.pth -encoder_ckpt_path ./work_dir/mini_repvit_dist.jit
