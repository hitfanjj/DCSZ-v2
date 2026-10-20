CUDA_VISIBLE_DEVICES=5 python colorloss_img.py \
                                --cfg ../config/eval/kitti-S.json \
                                --model ../../SEA-RAFT-main/pretrained/Tartan-C-T-TSKH-kitti432x960-S.pth \
                                --dataroot "test data path" \
                                --save_dir "save crop imgs path" \
                                --UW_size 0.6 