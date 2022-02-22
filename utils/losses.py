import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

class TVLoss_SML1(nn.Module):
    """ 
    TV loss
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, embs):
        
        # b,c,h,w = embs.shape
        grad_w_l = F.smooth_l1_loss(embs[:,:,:,:-1], embs[:,:,:,1:])
        grad_h_l = F.smooth_l1_loss(embs[:,:,:-1,:], embs[:,:,1:,:])

        # print(grad_w.shape)
        tv_loss = grad_w_l + grad_h_l

        return tv_loss

class CriterionPixelWise(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ignore_index = ignore_index
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        # if not reduce:
        #     print("disabled the reduce.")

    def forward(self, preds_S, preds_T):
        # preds_T[0] is the seg logit
        preds_T[0].detach()
        assert preds_S[0].shape == preds_T[0].shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S[0].shape
        softmax_pred_T = F.softmax(preds_T[0].permute(0,2,3,1).contiguous().view(-1,C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum( - softmax_pred_T * logsoftmax(preds_S[0].permute(0,2,3,1).contiguous().view(-1,C))))/W/H
        return loss

torch_losses = {
    'crossentropyloss': nn.CrossEntropyLoss,
    'softcrossentropyloss': smp.losses.SoftCrossEntropyLoss,
    'focalloss': smp.losses.FocalLoss,
    'jaccardloss': smp.losses.JaccardLoss,
    'diceloss': smp.losses.DiceLoss,
    'logitTV': TVLoss_SML1
}

def get_loss(loss, loss_weights=None, custom_losses=None):
    """Load a loss function based on a config file for the specified framework.

    Arguments
    ---------
    loss : dict
        Dictionary of loss functions to use.  Each key is a loss function name,
        and each entry is a (possibly-empty) dictionary of hyperparameter-value
        pairs.
    loss_weights : dict, optional
        Optional dictionary of weights for loss functions.  Each key is a loss
        function name (same as in the ``loss`` argument), and the corresponding
        entry is its weight.
    custom_losses : dict, optional
        Optional dictionary of Pytorch classes of any
        user-defined loss functions.  Each key is a loss function name, and the
        corresponding entry is the Python object implementing that loss.
    """
    if not isinstance(loss, dict):
        raise TypeError('The loss description is formatted improperly.'
                        ' See the docs for details.')
    if len(loss) > 1:
        # get the weights for each loss within the composite
        if loss_weights is None:
            # weight all losses equally
            weights = {k: 1 for k in loss.keys()}
        else:
            weights = loss_weights

        # check if sublosses dict and weights dict have the same keys
        if list(loss.keys()).sort() != list(weights.keys()).sort():
            raise ValueError(
                'The losses and weights must have the same name keys.')

        return TorchCompositeLoss(loss, weights, custom_losses)
    
    else:  # parse individual loss functions
        loss_name, loss_dict = list(loss.items())[0]
        return get_single_loss(loss_name, loss_dict, custom_losses)


def get_single_loss(loss_name, params_dict, custom_losses=None):

    if params_dict is None:
        if custom_losses is not None and loss_name in custom_losses:
            return custom_losses.get(loss_name)()
        else:
            return torch_losses.get(loss_name.lower())()

    else:
        if custom_losses is not None and loss_name in custom_losses:
            return custom_losses.get(loss_name)(**params_dict)
        else:
            return torch_losses.get(loss_name.lower())(**params_dict)

class TorchCompositeLoss(nn.Module):
    """Composite loss function."""
    def __init__(self, loss_dict, weight_dict=None, custom_losses=None):
        """Create a composite loss function from a set of pytorch losses."""
        super().__init__()
        self.weights = weight_dict
        self.losses = {loss_name: get_single_loss(loss_name,
                                                  loss_params,
                                                  custom_losses)
                       for loss_name, loss_params in loss_dict.items()}
        self.values = {}  # values from the individual loss functions

    def forward(self, outputs, targets):
        loss = 0
        for func_name, weight in self.weights.items():
            self.values[func_name] = self.losses[func_name](outputs, targets)
            loss += weight*self.values[func_name]
        return loss



