# A Light-weight Universal Medical Segmentation Network for Laptops Based on Knowledge Distillation

This repository is the official implementation of [A Light-weight Universal Medical Segmentation Network for Laptops Based on Knowledge Distillation](TBA). 


![Model](./figs/fig_network.jpg)

## Environments and Requirements

- Ubuntu 22.04.4 LTS
- CPU: Intel(R) Core(TM) i9-13900KF RAM: 4x16GB GPU: 1 x NVIDIA RTX 4090 24G
- CUDA 12.1
- Python 3.10

To install requirements:

Enter MedSAMonLaptop_RepViT folder `cd MedSAMonLaptop_RepViT` and run

```bash
pip install -e .
```

## Preprocessing

Running the data preprocessing code:

```bash
python npz_to_npy.py --input_path <path_to_input_data> --output_path <path_to_output_data>
```

## Training

To train the teacher model in the paper, run this command:

```bash
sh train_val_one_gpu.sh
```

To distill the repvit encoder from the trained teacher model, run this command:

```bash
sh train_val_one_gpu_distill.sh
```

## Trained Models

You can download trained models here:

- [Teacher Swin-T based MedSAM](https://github.com/YSongxiao/MedSAMonLaptop_RepViT/blob/main/work_dir/pretrain_weights/medsam_lite_best_extracted_swin.pth) trained on the above dataset with the above code. 
- [Distilled RepViT Encoder](https://github.com/YSongxiao/MedSAMonLaptop_RepViT/blob/main/work_dir/pretrain_weights/medsam_lite_repvit_encoder_best.pth) trained on the above dataset with the above code.


## Inference

To infer the testing cases, run this command:

```bash
sh inference.sh 
```


## Evaluation

To compute the evaluation metrics, run:

```bash
python evaluation/compute_metrics.py -s test_demo/litemedsam-seg -g test_demo/gts -csv_dir ./metrics.csv
```



## Results

The results will be released after CVPR2024.

[//]: # (| Model name       |  DICE  | 95% Hausdorff Distance |)

[//]: # (| ---------------- | :----: | :--------------------: |)

[//]: # (| My awesome model | 90.68% |         32.71          |)

## Acknowledgement

> We thank the contributors of public datasets and the authors of [RepViT](https://github.com/THU-MIG/RepViT). 
