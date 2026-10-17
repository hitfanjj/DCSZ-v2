CUDA_VISIBLE_DEVICES=1 python tools/test_syn.py --cfg config/eval/kitti-S.json \
                                            --model ../SEA-RAFT-main/pretrained/Tartan-C-T-TSKH-kitti432x960-S.pth \
                                            --log_dir ./xsmall/xiaomi06/small4_lr2_noload \
                                            --dataset_dir dataset/test