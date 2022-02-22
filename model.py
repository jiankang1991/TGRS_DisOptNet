
import torch
from torch import nn
from typing import Optional, Union, List
import torch.nn.functional as F
from torch.nn import Parameter

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder

class DeepLabV3plus(smp.DeepLabV3Plus):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, x):
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        decoder_output = self.decoder(*features)
        # print(decoder_output.shape) # .x256x128x128
        logit = self.segmentation_head(decoder_output)
        # multi_features = features[-2:] + [decoder_output]
        # multi_features = features[-3:] + [decoder_output]
        # multi_features = features[-4:] + [decoder_output]
        # multi_features = features[-5:] + [decoder_output]
        # multi_features = features[-1:] + [decoder_output]
        multi_features = [decoder_output]
        return logit, multi_features

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                
class FeatureFusionModuleSCSE_V2(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super().__init__()
        
        self.scse_1 = md.SCSEModule(in_chan)
        self.scse_2 = md.SCSEModule(in_chan)
        self.convblk = ConvBNReLU(in_chan*2, out_chan, ks=1, stride=1, padding=0)
        self.scse = md.SCSEModule(out_chan)
        self.init_weight()

    def forward(self, fsp, fcp):
        fsp = self.scse_1(fsp)
        fcp = self.scse_2(fcp)
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        feat_out = self.scse(feat)
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiDeepLabV3p_Dist(SegmentationModel):
    def __init__(self,
        encoder_name: str = "efficientnet-b3",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4) -> None:
        
        super().__init__()

        self.classes = classes
        self.encoder_sar = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.block_num = len(self.encoder_sar._blocks)
        self.drop_connect_rate = self.encoder_sar._global_params.drop_connect_rate
        # print(self.drop_connect_rate)
        # print(self.encoder_main._stage_idxs)
        self.encoder_sar.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )
        self.decoder_sar = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder_sar.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )
        
        # Distill branch
        encoder_rgb = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        encoder_rgb.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )
        self.decoder_rgb = DeepLabV3PlusDecoder(
            encoder_channels=encoder_rgb.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.encoder_rgb_stages = nn.ModuleList(encoder_rgb.get_stages()[-2:])

        self.seg_head_rgb = SegmentationHead(decoder_channels, classes, activation=activation, upsampling=upsampling)
        self.seg_head_sar = SegmentationHead(decoder_channels, classes, activation=activation, upsampling=upsampling)
        
        self.ffm = FeatureFusionModuleSCSE_V2(decoder_channels, decoder_channels)
        self.ffm_seg_head = SegmentationHead(decoder_channels, classes, activation=activation, upsampling=upsampling)
        
    def forward(self, x):
        features = self.encoder_sar(x)
        aux_features = [*features[:4]]
        block_number = 8.
        x = features[3]
        for stage in self.encoder_rgb_stages:
        # for stage in self.encoder_rgb_stages[1:]:   
            for module in stage:
                drop_connect = self.drop_connect_rate * block_number / self.block_num
                block_number += 1.
                x = module(x, drop_connect)
            aux_features.append(x)
        decode_sar = self.decoder_sar(*features)
        decode_rgb = self.decoder_rgb(*aux_features)

        logit_sar = self.seg_head_sar(decode_sar)
        logit_rgb = self.seg_head_rgb(decode_rgb)

        fused_features = self.ffm(decode_sar, decode_rgb)
        fused_logit = self.ffm_seg_head(fused_features)

        return logit_sar, logit_rgb, fused_logit, aux_features[-2:] + [decode_rgb]
