CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29558 train.py \
                                --cfg config/eval/kitti-S.json \
                                --model ./pretrained/searaft/Tartan-C-T-TSKH-kitti432x960-S.pth \
                                --log_dir "your output path" \
                                --dataset_dir "your dataset path" \
                                --UW_size 0.6 \
                                --model_size 'full' \
                                --epoch 10 \
                                --world_size=2