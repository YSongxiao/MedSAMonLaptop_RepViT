# A Light-weight Universal Medical Segmentation Network for Laptops Based on Knowledge Distillation

This repository is the official implementation of [A Light-weight Universal Medical Segmentation Network for Laptops Based on Knowledge Distillation](TBA). 


## Environments and Requirements

- Ubuntu 22.04.4 LTS
- CPU: Intel(R) Core(TM) i9-13900KF RAM: 4x16GB GPU: 1 x NVIDIA RTX 4090 24G
- CUDA 12.1
- Python 3.10

To install requirements:

Enter MedSAMonLaptop_RepViT folder
```bash
cd MedSAMonLaptop_RepViT
``` 
and run

```bash
pip install -e .
```

## Build Docker
```bash
docker build -f Dockerfile -t litemedsam_repvit .
```

> Note: don't forget the `.` in the end

Run the docker on the testing demo images

```bash
docker container run -m 8G --name litemedsam_repvit --rm -v $PWD/test_demo/imgs/:/workspace/inputs/ -v $PWD/test_demo/litemedsam-seg/:/workspace/outputs/ litemedsam_repvit:latest /bin/bash -c "sh predict.sh"
```

> Note: please run `chmod -R 777 ./*` if you run into `Permission denied` error.

Save docker 

```bash
docker save litemedsam_repvit | gzip -c > litemedsam_repvit.tar.gz
```


## Acknowledgement

> We thank the contributors of public datasets and the authors of [RepViT](https://github.com/THU-MIG/RepViT). 
