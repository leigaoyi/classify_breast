__version__ = "0.6.1"
from .model import EfficientNet
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

def efficientnet_b0(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b0",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b0", kwargs)

def efficientnet_b1(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b1",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b1", kwargs)

def efficientnet_b2(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b2",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b2", kwargs)

def efficientnet_b3(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b3",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b3", kwargs)

def efficientnet_b4(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b4",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b4", kwargs)




