CUDA_VISIBLE_DEVICES=5 python crop_img.py \
                                --cfg ../model/searaft/config/kitti-S.json \
                                --dataroot "test data path" \
                                --save_dir "save crop imgs path" \
                                --UW_size 0.6 