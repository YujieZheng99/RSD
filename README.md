# Restructuring the Teacher and Student in Self-Distillation

## Installation

This repo was tested with Ubuntu 20.04.4 LTS, Python 3.7.13, PyTorch 1.8.0, and CUDA 11.1.

## Training

Due to the paper being under review, the training code will be provided later.
Now the accuracy of the model can be validated based on the provided checkpoint files. 

## Validation:
```
python validation.py --model repResNet32 --model_path `save/student_model/repResNet32_deploy.pth` --blocktype AMBB --deploy_flag True
```
