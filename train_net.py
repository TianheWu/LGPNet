import datetime
import os
import sys
import argparse
import logging
import cv2
import torch
import torch.utils.data
import torch.optim as optim

import tensorboardX

from random import shuffle
from torchsummary import summary
from utils.visualisation.gridshow import gridshow
from utils.dataset_processing import evaluation
from utils.dataset_processing.augment_data import aug
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train LGPNet')

    # Network
    parser.add_argument('--network', type=str, default='ramnet', help='Network Name in .models')
    parser.add_argument('--layers', type=int, default=50, help='Layers number')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--start-split', type=float, default=0.0, help='The start of the split for train dataset')
    parser.add_argument('--end-split', type=float, default=0.2, help='The end of the split for train dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataset workers')
    parser.add_argument('--ds-shuffle', action='store_true', help='Turning OW to IW')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')
    parser.add_argument('--use-ssp', action='store_true', help='Use semi-supervised learning')
    parser.add_argument('--use-tea', action='store_true', help='Use knowledge distillation')
    parser.add_argument('--trained-net-path', type=str, default='', help='Path to trained LGPNet')
    parser.add_argument('--gpu-idx', type=str, default='0', help='GPU index')
    
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')

    args = parser.parse_args()
    return args


def validate(net, device, val_data):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out, ang_out,
                                                val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                no_grasps=1,
                                                grasp_width=w_out,
                                                )
            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False, use_tea=False, use_ssp=False, trained_net=None):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch: Data batches to train on
    :param vis: Visualise training progress
    :param use_tea: Use knowledge distillation
    :param use_ssp: Use semi-supervised learning
    :param trained_net: Teacher model
    :return: Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()
    if use_tea:
        trained_net.eval()
        lossd = {
            'loss': 0,
            'losses': {
                'pos_loss': 0,
                'cos_loss': 0,
                'sin_loss': 0,
                'width_loss': 0
            },
            'pred': {
                'pos': 0,
                'cos': 0,
                'sin': 0,
                'width': 0
            }
        }

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            # xc = x.to(f'cuda:{net.device_ids[0]}')
            # yc = [yy.to(f'cuda:{net.device_ids[0]}') for yy in y]

            xc = x.to(device)
            yc = [yy.to(device) for yy in y]

            pos, cos, sin, width = yc
            l_pos = pos[:2]
            l_cos = cos[:2]
            l_sin = sin[:2]
            l_width = width[:2]

            if use_ssp:
                trained_net.eval()
                l_train_data = xc[:2]
                ul_train_data = xc[2:]
                input_label = tuple([l_pos, l_cos, l_sin, l_width])
                xc, yc = aug(l_train_data, input_label, ul_train_data, trained_net, device, num_aug=3)
            else:
                xc = xc[:2]
                yc = [l_pos, l_cos, l_sin, l_width]
            
            if use_tea:
                t_label = trained_net(xc)
                s_loss = net.compute_loss(xc, yc)
                st_loss = net.compute_loss(xc, t_label)
                for key, val in lossd.items():
                    if isinstance(val, dict):
                        for _key, _val in val.items():
                            lossd[key][_key] = s_loss[key][_key] + st_loss[key][_key]
                    else:
                        lossd[key] = s_loss[key] + st_loss[key]
            else:
                lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()
    print(args)

    if args.vis:
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    train_dataset = Dataset(
        args.dataset_path, 
        start_split=args.start_split,
        end_split=args.end_split,
        ds_rotate=args.ds_rotate,
        is_val=False,
        ds_shuffle=args.ds_shuffle,
        random_rotate=True,
        random_zoom=True,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb
    )
    val_dataset = Dataset(
        args.dataset_path,
        start_split=args.start_split,
        end_split=args.end_split,
        ds_rotate=args.ds_rotate,
        is_val=True,
        ds_shuffle=args.ds_shuffle,
        random_rotate=False,
        random_zoom=False,
        include_depth=args.use_depth,
        include_rgb=args.use_rgb
    )
    
    logging.info('Training size: {}, start to end split is {}-{}'.format(train_dataset.length, args.start_split, args.end_split))
    logging.info('Validation size: {}, start to end split is {}-{}'.format(val_dataset.length, args.start_split, args.end_split))
    
    split_dataset_method = 'OW'
    if args.ds_shuffle:
        split_dataset_method = 'IW'
    logging.info('Split dataset with {} method'.format(split_dataset_method))

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    logging.info('Done')
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    LGPNet = get_network(args.network)

    device = torch.device("cuda:{}".format(args.gpu_idx))
    logging.info('Using Backbone Resnet-{}'.format(args.layers))
    net = LGPNet(layers=args.layers, dropout=0.1, classes=32, zoom_factor=8, pretrained=True, bin=6, is_stu=True)
    net = net.to(device)
    logging.info('Using semi-supervised learning: {}'.format(args.use_ssp))
    logging.info('Using knowledge distillation: {}'.format(args.use_tea))

    trained_net = None
    if args.use_ssp or args.use_tea:
        logging.info('Loading trained network...')
        trained_net = LGPNet(layers=50, dropout=0.1, classes=32, zoom_factor=8, pretrained=True, bin=6, is_stu=True)
        trained_net.load_state_dict(torch.load(args.trained_net_path), strict=False)
        trained_net = trained_net.to(device)
        logging.info('trained network done')

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    logging.info('Network Done')

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis, 
                                use_tea=args.use_tea, use_ssp=args.use_ssp, trained_net=trained_net)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_acc_%0.4f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_acc_%0.4f_statedict.pt' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()