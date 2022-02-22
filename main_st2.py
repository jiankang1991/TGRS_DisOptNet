import os
import argparse
from datetime import datetime

# import sys
# sys.path.append('../')

from utils.config import parse
from train_st2 import Trainer

def defineyaml(args):
    #YAML
    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    yamlcontents = f"""
sv_name: '{sv_name}'
model_name: deeplabv3plus
model_path: ./20210908_095909/checkpoints/20210908_095909_checkpoint.pth.tar
train: true
model_specs:
    encoder_name: efficientnet-b3
    in_channels: 4
    classes: 2
    upsampling: 4
batch_size: 8
data_specs:
    width: 512
    height: 512
    dtype:
    image_type: 32bit
    rescale: false
    rescale_minima: auto
    rescale_maxima: auto
    label_type: mask
    is_categorical: false
    mask_channels: 1
    val_holdout_frac:
    data_workers: 4
sar_training_data_csv: {args.sar_traincsv}
sar_validation_data_csv: {args.sar_validcsv}
rgb_training_data_csv: {args.rgb_traincsv}
rgb_validation_data_csv: {args.rgb_validcsv}
training_augmentation:
    augmentations:
        HorizontalFlip:
            p: 0.5
        RandomCrop:
            height: 512
            width: 512
            p: 1.0
        Normalize:
            mean:
                - 0.5
            std:
                - 0.125
            max_pixel_value: 255.0
            p: 1.0
    p: 1.0
    shuffle: true
validation_augmentation:
    augmentations:
        CenterCrop:
            height: 512
            width: 512
            p: 1.0
        Normalize:
            mean:
                - 0.5
            std:
                - 0.125
            max_pixel_value: 255.0
            p: 1.0
    p: 1.0
training:
    epochs: 200
    lr: 5e-3
    loss:
        diceloss:
            mode: multiclass
            from_logits: True
        crossentropyloss:
    loss_weights:
        crossentropyloss: 1.0
        diceloss: 1.0
"""
    print('saving file name is ', sv_name)
    checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
    logs_dir = os.path.join('./', sv_name, 'logs')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    with open(os.path.join('./', sv_name, f'{sv_name}.yaml'), 'w') as f:
        f.write(yamlcontents)

    return sv_name, checkpoint_dir, logs_dir

def main(args):

    sv_name, checkpoint_dir, logs_dir = defineyaml(args)
    config = parse(os.path.join('./', sv_name, f'{sv_name}.yaml'))
    config['checkpoint_dir'] = checkpoint_dir
    config['logs_dir'] = logs_dir
    trainer = Trainer(config)
    trainer.run()
    
    return None

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='SpaceNet 6 Algorithm')
    parser.add_argument('--sar-traincsv', default='/home/zkgy/Data/SpaceNet6/proc_train_test/train.csv',
                        help='Where to save reference CSV of training data')
    parser.add_argument('--sar-validcsv', default='/home/zkgy/Data/SpaceNet6/proc_train_test/valid.csv',
                        help='Where to save reference CSV of validation data')
    parser.add_argument('--rgb-traincsv', default='/home/zkgy/Data/SpaceNet6/proc_train_test/train.csv',
                        help='Where to save reference CSV of training data')
    parser.add_argument('--rgb-validcsv', default='/home/zkgy/Data/SpaceNet6/proc_train_test/valid.csv',
                        help='Where to save reference CSV of validation data')
    args = parser.parse_args()

    main(args)


