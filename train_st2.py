
import os
import torch
import torch.nn.functional as F
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorboardX import SummaryWriter
from utils.models import _load_model_weights, model_dict
from utils.dataGen import make_SAR_RGB_data_generator
from utils.losses import get_loss, CriterionPixelWise
from utils.metrics import dice_coeff, MetricTracker
from utils.lr_scheduler import LR_Scheduler

from model import BiDeepLabV3p_Dist, DeepLabV3plus

class Trainer:
    """Object for training `solaris` models using PyTorch. """
    def __init__(self, config, custom_losses=None):
        self.sv_name = config['sv_name']
        self.checkpoint_dir = config['checkpoint_dir']
        self.logs_dir = config['logs_dir']
        self.config = config
        self.batch_size = self.config['batch_size']
        self.model_name = self.config['model_name']
        self.model_path = self.config.get('model_path', None)

        self.rgb_model = DeepLabV3plus(**self.config['model_specs'])
        if self.model_path:
            self.rgb_model = _load_model_weights(self.rgb_model, self.model_path)
        self.sar_model = BiDeepLabV3p_Dist(**self.config['model_specs'])

        self.sar_train_df, self.sar_val_df = pd.read_csv(config['sar_training_data_csv']), pd.read_csv(config['sar_validation_data_csv'])
        self.rgb_train_df, self.rgb_val_df = pd.read_csv(config['rgb_training_data_csv']), pd.read_csv(config['rgb_validation_data_csv'])

        self.train_datagen = make_SAR_RGB_data_generator(self.config, self.sar_train_df, self.rgb_train_df, stage='train')
        self.val_datagen = make_SAR_RGB_data_generator(self.config, self.sar_val_df, self.rgb_val_df, stage='validate')
        self.epochs = self.config['training']['epochs']
        self.lr = self.config['training']['lr']

        self.seg_loss = get_loss(self.config['training'].get('loss'),
                             self.config['training'].get('loss_weights'),
                             custom_losses)
        self.dist_f_loss = torch.nn.MSELoss()
        self.dist_pix_loss = CriterionPixelWise()
        self.metrics = dice_coeff
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
        else:
            self.gpu_count = 0

        self.train_writer = SummaryWriter(os.path.join(self.logs_dir, 'runs', self.sv_name, 'training'))
        self.val_writer = SummaryWriter(os.path.join(self.logs_dir, 'runs', self.sv_name, 'val'))

        self.initialize_model()

    def initialize_model(self):
        if self.gpu_available:
            self.sar_model = self.sar_model.cuda()
            self.rgb_model = self.rgb_model.cuda()

            if self.gpu_count > 1:
                self.sar_model = torch.nn.DataParallel(self.sar_model)
                self.rgb_model = torch.nn.DataParallel(self.rgb_model)

        self.optimizer_sar = torch.optim.SGD(
                    self.sar_model.parameters(), lr=self.lr,
                    momentum=0.9, weight_decay=1e-4, nesterov=True
                )
        self.lr_scheduler = LR_Scheduler('poly', self.lr, self.epochs + 1, len(self.train_datagen))

    def run(self):
        """
        the main function to run
        """
        best_metric = 0
        for epoch in range(1, self.epochs+1):
            print('Epoch {}/{}'.format(epoch, self.epochs))
            print('-' * 10)
            self.train(epoch, best_metric)
            metric_v = self.val(epoch)

            is_best_metric = metric_v > best_metric
            best_metric = max(metric_v, best_metric)

            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.sar_model.module.state_dict() if isinstance(self.sar_model, torch.nn.DataParallel) else self.sar_model.state_dict(),
                'best_metric': best_metric,
                # 'optimizer': self.optimizer.state_dict()
            }, None)

    def train(self, epoch, best_metric):
        Losses = MetricTracker()
        self.sar_model.train(), self.rgb_model.eval()
        for idx, batch in enumerate(tqdm(self.train_datagen, desc="training", ascii=True, ncols=60)):
            sar_img = batch['image'].cuda() 
            rgb_img = batch['imageRGB'].cuda()
            target = batch['mask'].cuda().long()

            self.optimizer_sar.zero_grad()

            with torch.no_grad():
                rgb_seg, rgb_features = self.rgb_model(rgb_img)
                rgb_tgt = torch.argmax(rgb_seg, dim=1)

            logit_sar, logit_rgb, logit_fused, sar_features = self.sar_model(sar_img)

            sar_loss = self.seg_loss(logit_sar, target)
            rgb_loss = self.seg_loss(logit_rgb, rgb_tgt.detach()) + self.dist_pix_loss([logit_rgb], [rgb_seg.detach()])
            fused_loss = self.seg_loss(logit_fused, target)

            dist_rgb_loss = sum([self.dist_f_loss(sar_f, rgb_f) for sar_f, rgb_f in zip(sar_features, rgb_features)])
            loss = sar_loss + rgb_loss + fused_loss + dist_rgb_loss / len(sar_features)

            loss.backward()

            Losses.update(loss.item(), sar_img.size(0))
            self.optimizer_sar.step()
            self.lr_scheduler(self.optimizer_sar, idx, epoch, best_metric)

        info = {
                "Loss": Losses.avg
        }
        for tag, value in info.items():
            self.train_writer.add_scalar(tag, value, epoch)
        
        print('Train Loss: {:.6f}'.format(
                Losses.avg
                ))
        return None
    
    def val(self, epoch):
        self.sar_model.eval()
        torch.cuda.empty_cache()
        
        val_Metric = MetricTracker()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.val_datagen, desc="val", ascii=True, ncols=60)):
                if torch.cuda.is_available():
                    data = batch['image'].cuda()
                    target = batch['mask'].cuda().float()
                
                logits = self.sar_model(data)
                outputs = torch.argmax(logits[2], dim=1).float()
                val_Metric.update(self.metrics(outputs, target), outputs.size(0))

        info = {
            "Dice": val_Metric.avg
        }
        for tag, value in info.items():
            self.val_writer.add_scalar(tag, value, epoch)
        
        print('Val Dice: {:.6f}'.format(
                val_Metric.avg
                ))

        return val_Metric.avg
        
    def save_checkpoint(self, state, is_best):
        filename = os.path.join(self.checkpoint_dir, self.sv_name + '_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.checkpoint_dir, self.sv_name + '_model_best.pth.tar'))

