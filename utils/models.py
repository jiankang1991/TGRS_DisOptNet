
import torch
from torch import nn

from typing import Optional, Union, List
import segmentation_models_pytorch as smp

def _load_model_weights(model, path):
    """Backend for loading the model."""
    if torch.cuda.is_available():
        try:
            loaded = torch.load(path)
        except FileNotFoundError:
            raise FileNotFoundError("{} doesn't exist.".format(path))
    else:
        try:
            loaded = torch.load(path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError("{} doesn't exist.".format(path))

    # if isinstance(loaded, torch.nn.Module):  # if it's a full model already
    #     model.load_state_dict(loaded.state_dict())
    # else:
    #     model.load_state_dict(loaded)
    model.load_state_dict(loaded['state_dict'])
    print(f"model {path} loading finished")
    return model

model_dict = {
    'deeplabv3plus':smp.DeepLabV3Plus,
    'unet':smp.Unet,
    'unetplusplus':smp.UnetPlusPlus,
    'fpn': smp.FPN,
    'psp': smp.PSPNet
}

