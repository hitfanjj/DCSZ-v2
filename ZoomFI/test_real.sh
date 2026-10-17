
# EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI
CUDA_VISIBLE_DEVICES=4 python ./test_real.py --cfg config/eval/kitti-S.json \
                                            --model ../SEA-RAFT-main/pretrained/Tartan-C-T-TSKH-kitti432x960-S.pth \
                                            --log_dir ./xiaorong_color/huawei85_num/addf_color_random_gaussian_180 \
                                            --dataset_dir ./dataset/real_world