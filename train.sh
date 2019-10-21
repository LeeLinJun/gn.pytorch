#!/bin/bash
suffix=$1
echo "Generating training data..."
python gen_data_sst_acrobat.py --max_episodes 1800 --train_data data/acrobot$suffix.npy --time_step $suffix
echo "Generating testing data..."
python gen_data_sst_acrobat.py --max_episodes 200 --train_data data/acrobot_test$suffix.npy   --time_step $suffix
python normalizer_sst.py data/acrobot$suffix.npy normalized/acrobot$suffix.pth
python train_gn_sst.py --model '' --train_data data/acrobot$suffix.npy --test_data data/acrobot_test$suffix.npy --normalizer "normalized/acrobot$suffix.pth" --tstep $suffix
