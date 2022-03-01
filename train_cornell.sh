python train_net.py --description gpnn_r50_cor_0.0-0.2_ow \
    --batch-size 8 \
    --network lgpnet \
    --dataset cornell \
    --dataset-path ./dataset/dataset_cornell/ \
    --layers 50 \
    --gpu-idx 7 \
    --start-split 0.0 \
    --end-split 0.2 