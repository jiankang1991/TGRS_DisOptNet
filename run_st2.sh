export CUDA_VISIBLE_DEVICES=0

python main_st2.py \
--sar-traincsv /home/zkgy/Documents/SpaceNet6_base/pretrain/train.csv \
--sar-validcsv /home/zkgy/Documents/SpaceNet6_base/pretrain/valid.csv \
--rgb-traincsv /home/zkgy/Documents/SpaceNet6_base/pretrain/opticaltrain.csv \
--rgb-validcsv /home/zkgy/Documents/SpaceNet6_base/pretrain/opticalvalid.csv \
