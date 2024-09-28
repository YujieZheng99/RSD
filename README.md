# Restructuring the Teacher and Student in Self-Distillation

## Installation

This repo was tested with Ubuntu 20.04.4 LTS, Python 3.7.13, PyTorch 1.8.0, and CUDA 11.1.
| Package | Version (CIFAR) | Version (ImageNet) |
| ------ | ------ | ------ |
| h5py | 3.11.0 | 3.11.0 |
| lmdb | 1.4.1 | 1.4.1 |
| matplotlib | 3.5.3 | 3.9.0 |
| msgpack\_python | 0.5.6 | 0.5.6 |
| numpy | 1.21.6 | 1.21.6 |
| Pillow | 9.4.0 | 9.3.0 |
| pyarrow | 12.0.1 | 12.0.1 |
| scikit\_learn | 1.5.0 | 1.5.0 |
| seaborn | 0.13.2 | 0.13.2 |
| six | 1.16.0 | 1.16.0 |
| tensorboard\_logger | 0.1.0 | 0.1.0 |
| torch | 1.8.0+cu111 | 1.8.0+cu111 |
| torchvision | 0.9.0+cu111 | 0.9.0+cu111 |
| tqdm | 4.65.0 | 4.66.1 | 

```
pip install -r requirements.txt
```

## Training
For CIFAR-100:
```
python RSD_cifar100.py --model repResNet32 --blocktype AMBB --dataset cifar100 --kd_T 3 -a 1.0 -b 1.0 -r 0.2
```

## Equivalent structural transformation:
```
python convert.py --arch repResNet32 --blocktype AMBB --load `save/student_model/repResNet32_last.pth` --save `save/student_model/repResNet32_deploy.pth`
```

## Validation:
The accuracy of the model can be validated based on the provided checkpoint files. 
Our checkpoints are at https://github.com/YujieZheng99/RSD/releases/tag/checkpoints
```
python validation.py --model repResNet32 --model_path `save/student_model/repResNet32_deploy.pth` --blocktype AMBB --deploy_flag True
```
