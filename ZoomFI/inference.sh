CUDA_VISIBLE_DEVICES=0 python ./inference.py --cfg model/searaft/config/kitti-S.json \
                                            --model ./pretrained/searaft/Tartan-C-T-TSKH-kitti432x960-S.pth \
                                            --log_dir "your load path" \
                                            --dataset_dir ../  \
                                            --suffix xiaomi_06_flownet \
                                            --UW_size 0.6 