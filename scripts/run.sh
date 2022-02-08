# train; ImageNet path is set in the config file
python moco_spawn_ada_warm.py configs/IN200/moco/wm_up10.py \
  --work-dir ./checkpoints/IN200_up10

# linear
python DDP_linear.py configs/linear/imagenet/Res50_IN200_wm.py \
  --load ./checkpoints/IN200_up10/last.pth
