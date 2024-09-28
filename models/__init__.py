from .repvgg import repvgg16, repvgg19
from .represnetv2 import repResNet32, repResNet110, repWRN20_8, repResNet50
from .repdensenet import repDenseNetd40k12
from .convnet_utils import switch_deploy_flag, switch_conv_bn_impl

model_dict = {
    'repvgg16': repvgg16,
    'repvgg19': repvgg19,
    'repResNet32': repResNet32,
    'repResNet110': repResNet110,
    'repWRN20_8': repWRN20_8,
    'repResNet50': repResNet50,
    'repDenseNet40_12': repDenseNetd40k12
}
