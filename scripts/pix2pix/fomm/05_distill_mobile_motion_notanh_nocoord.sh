#!/usr/bin/env bash
python distill.py --dataroot database/fomm_88k \
  --dataset_mode triplet\
  --distiller resnet \
  --log_dir logs/pix2pix/fomm/05_distill_motion_notanh_nocoord_88k/ \
  --batch_size 8 \
  --input_nc 6 \
  --output_nc 3 \
  --teacher_ngf 96 \
  --pretrained_ngf 96 \
  --restore_teacher_G_path logs/pix2pix/fomm/05_mobile_motion_notanh_nocoord_88k/checkpoints/latest_net_G.pth \
  --restore_pretrained_G_path logs/pix2pix/fomm/05_mobile_motion_notanh_nocoord_88k/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/fomm/05_mobile_motion_notanh_nocoord_88k/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/fomm_B_88k.npz \
  --num_threads 0 \
  --use_motion \
  --use_wandb
