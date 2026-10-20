CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29558 train_color.py \
                                --cfg config/eval/kitti-S.json \
                                --model ./pretrained/Tartan-C-T-TSKH-kitti432x960-S.pth \
                                --log_dir "your output path" \
                                --dataset_dir "your dataset path" \
                                --load_dir "your pretrained nocolor model path" \
                                --UW_size 0.6 \
                                --color_model "your color model path" \
                                --epoch 5 \
                                --world_size=2