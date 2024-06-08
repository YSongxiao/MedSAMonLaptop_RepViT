python train_val_one_gpu.py \
    -pretrained_checkpoint "lite_medsam.pth" \
    -work_dir "work_dir" \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 100