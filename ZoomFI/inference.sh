CUDA_VISIBLE_DEVICES=0 python ./inference.py --cfg config/eval/kitti-S.json \
                                            --model ./pretrained/searaft/Tartan-C-T-TSKH-kitti432x960-S.pth \
                                            --log_dir "your load path" \
                                            --dataset_dir ../  \
                                            --suffix 4 \
                                            --UW_size 0.6 