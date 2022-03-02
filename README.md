# LGPNet: Alleviating a few Labels and Large-Scale Network Dilemmas in Grasping Detection

## Abstract
Training deep neural networks with numerous labeled data is the mainstream approach for grasping detection tasks. However, most of the natural labeled data is challenging to obtain. Meanwhile, the mature grasping networks are large-scale with many parameters, but hardware conditions in realistic scenes limit robot grasping. Thus, lightweight models are essential for a robot to respond quickly. In this paper, we propose a new semi-supervised learning method, focusing on solving the problems of a few labeled data in natural scenes. Moreover, we propose a novel Lightweight Grasping Pyramid Network (LGPNet) with knowledge distillation to solve the large-scale network with high calculating speed. We conduct experiments in the Cornell dataset and Jacquard dataset with 25% labeled data and 75% unlabeled data, obtaining state-of-the-art performance with depth images by object-wise data split. The accuracy score reaches 97.2% in the Cornell dataset and 92.4% in the Jacquard dataset.

## Installation
Python requirements can installed by:
```shell
pip install -r requirements.txt
```

## Dataset
### Cornell Dataset
Training datasets: [Cornell Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp). 
### Jacquard Dataset
Testing dataset: [Jacquard Dataset](https://jacquard.liris.cnrs.fr/)

## Network
![result](./images/LGPNet.png)

### Training

```shell
# Train LGPNet on Cornell Dataset
sh train_cornell.sh

# Train LGPNet on Cornell Dataset
sh train_jacquard.sh
```

## Contact Information

Email: tianhe_wu@emails.bjut.edu.cn




