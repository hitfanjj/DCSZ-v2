CUDA_VISIBLE_DEVICES=6 python ./evaluate_real.py --cfg config/eval/kitti-M.json \
                                            --model ../SEA-RAFT-main/pretrained/Tartan-C-T-TSKH-kitti432x960-M.pth \
                                            --log_dir ./ckpt_251015_huawei_06to10_1/addforward_lr1 \
                                            --dataset_dir ../