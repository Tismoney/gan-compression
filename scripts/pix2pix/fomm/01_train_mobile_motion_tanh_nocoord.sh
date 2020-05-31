#!/usr/bin/env bash
python3 train.py --dataroot database/fomm_62k \
  --model pix2pix \
  --dataset_mode triplet\
  --log_dir logs/pix2pix/fomm/01_mobile_motion_tanh_nocoord_62k \
  --real_stat_path real_stat/fomm_B.npz \
  --batch_size 8 \
  --ngf 96 \
  --lambda_recon 10 \
  --lambda_gan 1 \
  --nepochs 200 \
  --nepochs_decay 200 \
  --save_epoch_freq 50 \
  --save_latest_freq 10000 \
  --eval_batch_size 16 \
  --num_threads 0 \
  --use_wandb \
  --use_motion \
  --use_motion_tanh
