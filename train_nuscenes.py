import argparse
import os
import time
import warnings

import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data as data
from nuscenes.nuscenes import NuScenes

from config.config import config
from data.nuscenes import TrainDataset
from layer.matching_loss import Matching_Loss
from layer.matching_network import build_network
from layer.pose_network import PoseNet
from utils.augmentations import collate_fn
from utils.augmentations import SSJAugmentation
from utils.operation import show_batch_circle_image
warnings.filterwarnings('ignore')


# import posee


def str2bool(v): return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(
    description='Joint Pose and Association Train',
)
parser.add_argument(
    '--basenet', default=config['base_net_folder'], help='pretrained base model',
)
parser.add_argument(
    '--batch_size', default=config['batch_size'], type=int, help='Batch size for training',
)
parser.add_argument(
    '--resume', default=config['resume'], type=str, help='Resume from checkpoint',
)
parser.add_argument(
    '--num_workers', default=config['num_workers'], type=int, help='Number of workers used in dataloading',
)
parser.add_argument(
    '--iterations', default=config['iterations'], type=int, help='Number of training iterations',
)
parser.add_argument(
    '--start_iter', default=config['start_iter'], type=int,
    help='Begin counting iterations starting from this value (used with resume)',
)
parser.add_argument(
    '--cuda', default=config['cuda'], type=str2bool, help='Use cuda to train model',
)
parser.add_argument(
    '--lr', '--learning-rate',
    default=config['learning_rate'], type=float, help='initial learning rate',
)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--weight_decay', default=5e-4,
    type=float, help='Weight decay for SGD',
)
parser.add_argument(
    '--gamma', default=0.1, type=float,
    help='Gamma update for SGD',
)

parser.add_argument(
    '--save_folder', default=config['save_folder'], help='Location to save checkpoint models',
)
parser.add_argument(
    '--nuscenes_data_root',
    default=config['nuscenes_data_root'], help='Location of nuscenes TL pose and tracking dataset',
)
parser.add_argument(
    '--nuscenes_root', type=str,
    default='/s/red/a/nobackup/vision/nuScenes/data/sets/nuscenes', help='path to nuscenes directory ',
)
parser.add_argument(
    '--Joint', type=bool, default=True,
    help='True if joint pose and matching, False if matching only ',
)

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if 'save_images_folder' in config and not os.path.exists(config['save_images_folder']):
    os.mkdir(config['save_images_folder'])

means = config['mean_pixel']
batch_size = args.batch_size
max_iter = args.iterations
weight_decay = args.weight_decay
max_objects = config['max_object']

if 'learning_rate_decay_by_epoch' in config:
    stepvalues = list(
        config['epoch_size'] * i for i in config['learning_rate_decay_by_epoch']
    )
    save_weights_iteration = config['save_weight_every_epoch_num'] * \
        config['epoch_size']
else:
    print('stepvalues ')
    stepvalues = (20000, 40000, 60000, 80000, 100000)
    save_weights_iteration = 5000

gamma = args.gamma
momentum = args.momentum


estimator = PoseNet()
net = build_network(
    'train', pose_estimator=estimator,
    max_objects=max_objects, batch_size=batch_size,
)


if args.resume:
    print(f'Resuming training, loading {args.resume}...')
    net.load_state_dict(torch.load(args.resume))
else:
    vgg_weights = torch.load(args.basenet)

    print('Loading the base network...')
    net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    net.selector.apply(weights_init)
    net.final_net.apply(weights_init)

lr = args.lr
weight_decay = args.weight_decay
optimizer = optim.Adam(net.parameters(), lr=lr)


criterion = Matching_Loss(args.cuda)


def train():

    net.train()
    current_lr = config['learning_rate']
    print('Loading Dataset...')

    dataset = TrainDataset(
        args.nuscenes_data_root,
        SSJAugmentation(
            config['image_size'], means,
        ),
    )
    print('length = ', len(dataset))
    epoch_size = len(dataset) // args.batch_size
    step_index = 0

    batch_iterator = None
    batch_size = 1
    data_loader = data.DataLoader(
        dataset, batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    if args.Joint:
        nusc = NuScenes(
            version='v1.0-trainval', verbose=True,
            dataroot=args.nuscenes_root,
        )
    for iteration in range(0, 380000):
        print(iteration)
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            all_epoch_loss = []
        print('pass 0')
        if iteration in stepvalues:
            step_index += 1
            current_lr = current_lr*0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # load train data
        img_pre, img_next, boxes_pre, boxes_next, labels, valid_pre, valid_next, current_poses, next_poses, pre_bbox, next_bbox, img_org_pre, img_org_next, current_tokens, next_tokens =\
            next(batch_iterator)

        if args.Joint:
            current_token = current_tokens[0]
            next_token = next_tokens[0]
            first_token = current_tokens[1]

            current_cam_record = nusc.get('sample_data', current_token)
            current_cam_path = nusc.get_sample_data_path(current_token)
            current_cam_path, boxes, current_camera_intrinsic = nusc.get_sample_data(
                current_cam_record['token'],
            )
            current_cs_record = nusc.get(
                'calibrated_sensor', current_cam_record['calibrated_sensor_token'],
            )
            current_poserecord = nusc.get(
                'ego_pose', current_cam_record['ego_pose_token'],
            )

            next_cam_record = nusc.get('sample_data', next_token)
            next_cam_path = nusc.get_sample_data_path(next_token)
            next_cam_path, boxes, next_camera_intrinsic = nusc.get_sample_data(
                next_cam_record['token'],
            )
            next_cs_record = nusc.get(
                'calibrated_sensor', next_cam_record['calibrated_sensor_token'],
            )
            next_poserecord = nusc.get(
                'ego_pose', next_cam_record['ego_pose_token'],
            )

            first_cam_record = nusc.get('sample_data', first_token)
            first_cam_path = nusc.get_sample_data_path(first_token)
            first_cam_path, boxes, first_camera_intrinsic = nusc.get_sample_data(
                first_cam_record['token'],
            )
            first_cs_record = nusc.get(
                'calibrated_sensor', first_cam_record['calibrated_sensor_token'],
            )
            first_poserecord = nusc.get(
                'ego_pose', first_cam_record['ego_pose_token'],
            )

        if args.cuda:
            img_pre = (img_pre.cuda())
            img_next = (img_next.cuda())
            boxes_pre = (boxes_pre.cuda())
            boxes_next = (boxes_next.cuda())
            valid_pre = (valid_pre.cuda())
            valid_next = (valid_next.cuda())
            labels = (labels.cuda())
            current_poses = (current_poses.cuda())
            next_poses = (next_poses.cuda())
            pre_bbox = (pre_bbox.cuda())
            next_bbox = (next_bbox.cuda())
            img_org_pre = (img_org_pre.cuda())
            img_org_next = (img_org_next.cuda())

        # forward
        if args.Joint:
            out, pose_loss = net(
                args.Joint, img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next, current_poses, next_poses, pre_bbox, next_bbox, img_org_pre,
                img_org_next, current_cs_record, current_poserecord, next_cs_record, next_poserecord, first_cs_record, first_poserecord, first_camera_intrinsic,
            )

        else:
            out = net(
                img_pre, img_next, boxes_pre,
                boxes_next, valid_pre, valid_next,
            )

        optimizer.zero_grad()
        loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = criterion(
            out, labels, valid_pre, valid_next,
        )
        total_loss = loss+0.1*pose_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        all_epoch_loss += [loss.data.cpu()]

        print(
            'iter ' + repr(iteration) + ', ' + repr(epoch_size) + ' || epoch: %.4f ' %
            (iteration/(float)(epoch_size)) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ',
        )

        if iteration % 1000 == 0:

            result_image = show_batch_circle_image(
                img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next, predict_indexes, iteration,
            )

        if iteration % 1000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(
                net.state_dict(),
                os.path.join(
                    args.save_folder,
                    'model_' + repr(iteration) + '.pth',
                ),
            )


if __name__ == '__main__':
    train()
