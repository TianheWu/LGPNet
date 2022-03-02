# Lightweight Grasping Pyramid Network

## Abstract
We propose a new semi-supervised learning method, focusing on solving the problems of a few labeled data in natural scenes. Moreover, we propose a novel **Lightweight Grasping Pyramid Network (LGPNet)** with knowledge distillation to solve the large-scale network with high calculating speed. We conduct experiments in the Cornell dataset and Jacquard dataset with **25%** labeled data and **75%** unlabeled data, obtaining state-of-the-art performance with depth images by object-wise data split. The accuracy score reaches 97.2% in the Cornell dataset and 92.4% in the Jacquard dataset.

## Installation
Python requirements can installed by:
```shell
pip install -r requirements.txt
```

## Cornell Dataset & Jacquard Dataset
- [Cornell Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp)
- [Jacquard Dataset](https://jacquard.liris.cnrs.fr/)

## Training
```shell
# You can train LGPNet by this command.
python train_net.py --description <Description of training> \
    --batch-size 8 \
    --network lgpnet \
    --dataset <cornell or jacquard> \
    --dataset-path <Path to your dataset> \
    --layers 50 \
    --gpu-idx 7 \
    --start-split 0.0 \
    --end-split 0.2 

# --start-split is a float number between 0.0-1.0, it means the start postion of split dataset.
# --end-split is a float number between 0.0-1.0, it means the end postion of split dataset.

# You can run the shell files to train Cornell dataset and Jacquard dataset.
```

## Contact Information
Email: tianhe_wu@emails.bjut.edu.cn
