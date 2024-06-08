python train_val_one_gpu.py \
    -tch_pretrained_checkpoint "work_dir/pretrain_weights/medsam_lite_best_extracted_swin.pth" \
    -work_dir "work_dir" \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 100