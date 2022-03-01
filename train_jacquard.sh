python train_net.py --description gpnn_r18_jacssptea_0.8-1.0 \
    --batch-size 3 \
    --network lgpnet \
    --dataset jacquard \
    --dataset-path ./dataset/dataset_jacquard/ \
    --layers 18 \
    --gpu-idx 7 \
    --start-split 0.8 \
    --end-split 1.0 
