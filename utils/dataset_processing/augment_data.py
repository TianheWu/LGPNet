import torch
import numpy as np
import random
import copy

from random import shuffle


class Augment():
    """
    :param data: size [300, 300], type ndarray
    :return: size [300, 300] type numpy, except none_label function
    """
    @staticmethod
    def rotate_90(data):
        return np.rot90(data, -1)
    
    @staticmethod
    def rotate_180(data):
        return np.rot90(data, 2)
    
    @staticmethod
    def rotate_270(data):
        return np.rot90(data, 1)

    @staticmethod
    def ver_mirror(data):
        return data[::-1]
    
    @staticmethod
    def hor_mirror(data):
        data = Augment.rotate_180(data)
        return Augment.ver_mirror(data)
    
    @staticmethod
    def ver_hor_mirror(data):
        data = Augment.ver_mirror(data)
        return Augment.hor_mirror(data)


def augment(data, method='rotate_90'):
    """
    :param data: size [300, 300], type ndarray
    :param method: rotate and mirror
    :return: size [300, 300], type ndarray
    """
    aug = Augment()
    return getattr(aug, method)(data)


def pred_model(model, x):
    """
    :param model: The net you will train
    :param x: The dataset, [1, 300, 300]
    :return: Soft label, [pos, cos, sin, width]
    """
    model.eval()
    x = x.unsqueeze(0)
    ret = model(x)
    return ret


def aug(l_train_data, input_label, ul_train_data, net, device, num_aug=6):
    """
    :param l_train_data: [2, 1, 300, 300], type tensor 
    :param input_label: tuple(pos, cos, sin, width), each [2, 1, 300, 300] 
    :param ul_train_data: [6, 1, 300, 300], type tensor 
    :param net: The net you will train
    :param num_aug: the number of data 
    :return: type tuple, (data includes augment label data and augment unlabel data, augment label, guess label)
    """
    random_aug_method = ['rotate_90', 'rotate_180', 'rotate_270', 'ver_mirror', 'hor_mirror', 'ver_hor_mirror']
    aug_l_train_data = []; aug_l_label = []; aug_ul_train_data = []; aug_ul_label = []

    # Augment labeled data
    for i in range(len(l_train_data)):
        idx = random.randint(0, 5)
        aug_l_train_data.append(l_train_data[i])
        aug_l_label.append([input_label[0][i], input_label[1][i], input_label[2][i], input_label[3][i]])
        x = l_train_data[i].cpu().squeeze(0).numpy()
        y_pos = input_label[0][i].cpu().squeeze(0).numpy()
        y_cos = input_label[1][i].cpu().squeeze(0).numpy()
        y_sin = input_label[2][i].cpu().squeeze(0).numpy()
        y_width = input_label[3][i].cpu().squeeze(0).numpy()

        aug_l_train_data.append(torch.from_numpy(augment(x, method=random_aug_method[idx]).copy()).unsqueeze(0).to(device))
        aug_label_pos = torch.from_numpy(augment(y_pos, method=random_aug_method[idx]).copy()).unsqueeze(0).to(device)
        aug_label_cos = torch.from_numpy(augment(y_cos, method=random_aug_method[idx]).copy()).unsqueeze(0).to(device)
        aug_label_sin = torch.from_numpy(augment(y_sin, method=random_aug_method[idx]).copy()).unsqueeze(0).to(device)
        aug_label_width = torch.from_numpy(augment(y_width, method=random_aug_method[idx]).copy()).unsqueeze(0).to(device)
        aug_l_label.append([aug_label_pos, aug_label_cos, aug_label_sin, aug_label_width])

    # Augment unlabeled data
    for i in range(len(ul_train_data)):
        x = ul_train_data[i].cpu().squeeze(0).numpy()
        aug_ul_train_data.append(ul_train_data[i])
        tmp_pred = pred_model(net, ul_train_data[i])
        tmp_aug_pos, tmp_aug_cos, tmp_aug_sin, tmp_aug_width = tmp_pred
        aug_ul_label.append([tmp_aug_pos.squeeze(0), tmp_aug_cos.squeeze(0), tmp_aug_sin.squeeze(0), tmp_aug_width.squeeze(0)])

        ul_data_dict = {}
        for _ in range(num_aug):
            idx = 99
            while True:
                idx = random.randint(0, 5)
                if idx not in ul_data_dict:
                    ul_data_dict[idx] = 1
                    break
            # [1, 300, 300]
            tmp_data = torch.from_numpy(augment(x, random_aug_method[idx]).copy()).unsqueeze(0).to(device)
            aug_ul_train_data.append(tmp_data)
            tmp_pred = pred_model(net, tmp_data)
            tmp_aug_pos, tmp_aug_cos, tmp_aug_sin, tmp_aug_width = tmp_pred
            aug_ul_label.append([tmp_aug_pos.squeeze(0), tmp_aug_cos.squeeze(0), tmp_aug_sin.squeeze(0), tmp_aug_width.squeeze(0)])

    aug_data = torch.tensor([]).to(device)
    aug_label = []
    # [1, 300, 300]
    for x in aug_l_train_data:
        x = x.to(device)
        aug_data = torch.cat((aug_data, x.unsqueeze(0)), 0)
    for x in aug_ul_train_data:
        x = x.to(device)
        aug_data = torch.cat((aug_data, x.unsqueeze(0)), 0)
    for x in aug_l_label:
        aug_label.append(x)
    for x in aug_ul_label:
        aug_label.append(x)

    w = list(zip(aug_data, aug_label))
    shuffle(w)
    w_data, w_label = zip(*w)
    w_label = list(w_label)
    w_data = list(w_data)

    ret_x = torch.tensor([]).to(device)
    pos = torch.tensor([]).to(device)
    cos = torch.tensor([]).to(device)
    sin = torch.tensor([]).to(device)
    width = torch.tensor([]).to(device)

    for x in w_data:
        ret_x = torch.cat((ret_x, x.unsqueeze(0)), 0)
    for x in w_label:
        _pos, _cos, _sin, _width = x
        # one = torch.ones(1, 300, 300).to(device)
        # zero = torch.zeros(1, 300, 300).to(device)
        # _pos = torch.where(_pos > 0.2, one, _pos)
        # _pos = torch.where(_pos <= 0.2, zero, _pos)

        pos = torch.cat((pos, _pos.unsqueeze(0)), 0)
        cos = torch.cat((cos, _cos.unsqueeze(0)), 0)
        sin = torch.cat((sin, _sin.unsqueeze(0)), 0)
        width = torch.cat((width, _width.unsqueeze(0)), 0)

    return ret_x, [pos, cos, sin, width]
