import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from pyquaternion import Quaternion
from torch.nn.modules.loss import _Loss

from config.config import config


def loss_calculation(pred_r, pred_t, target):
    bs, num_p, _ = pred_r.size()
    target_t = target[:, 0]
    target_r = target[:, 1:]
    rz = target_r[0][2].item()
    dis_t = torch.mean(torch.norm((target_t - pred_t), dim=2), dim=1)
    dis_r = torch.mean(
        torch.norm(
            (target_r.view(bs, num_p) - pred_r.view(bs, num_p)), dim=0,
        ), dim=0,
    )
    if rz > 0.95:
        loss = 10*dis_r+0.008*dis_t
    elif rz > -0.9:

        loss = 5*dis_r+0.008*dis_t

    else:
        loss = dis_r+0.002*dis_t

    return loss, dis_r, dis_t


def cosh(y_t, y_prime_t):
    ey_t = y_t - y_prime_t
    return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class Loss(_Loss):

    def __init__(self):
        super().__init__(True)

    def forward(self, pred_r, pred_t, target):

        return loss_calculation(pred_r, pred_t, target)


def convert_to_ref_frame(Cx, Cy, pred_t, pred_r, cs_record, poserecord, first_rec, first_poserecord, camera_intrinsic):
    points_T = np.array([[Cx], [Cy], [pred_t[0][0].item()]])
    points_T[:2] = points_T[:2]*points_T[2].item()
    view = camera_intrinsic
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points_T.shape[1]

    # Do operation in homogenous coordinates.
    points_T = np.concatenate((points_T, np.ones((1, nbr_points))))
    points_T = np.dot(np.linalg.inv(viewpad), points_T)
    points_T = points_T[:3, :]

    points_T = np.dot(
        np.linalg.inv(
            Quaternion(
                cs_record['rotation'],
            ).rotation_matrix.T,
        ), points_T,
    )
    points_T = points_T + np.array(cs_record['translation']).reshape((-1, 1))

    points_T = np.dot(
        np.linalg.inv(
            Quaternion(
                poserecord['rotation'],
            ).rotation_matrix.T,
        ), points_T,
    )
    points_T = points_T + np.array(poserecord['translation']).reshape((-1, 1))

    # Transform into the ego vehicle frame for the timestamp of the image.
    points_T = points_T - \
        np.array(first_poserecord['translation']).reshape((-1, 1))
    points_T = np.dot(
        Quaternion(
            first_poserecord['rotation'],
        ).rotation_matrix.T, points_T,
    )

    # Transform into the camera.
    points_T = points_T - np.array(first_rec['translation']).reshape((-1, 1))
    points_T = np.dot(
        Quaternion(
            first_rec['rotation'],
        ).rotation_matrix.T, points_T,
    )

    points_R = np.array(
        [[pred_r[0][0].item()], [pred_r[0][1].item()], [pred_r[0][2].item()]],
    )

    points_R = np.dot(
        np.linalg.inv(
            Quaternion(
                cs_record['rotation'],
            ).rotation_matrix.T,
        ), points_R,
    )
    points_R = points_R + np.array(cs_record['translation']).reshape((-1, 1))

    points_R = np.dot(
        np.linalg.inv(
            Quaternion(
                poserecord['rotation'],
            ).rotation_matrix.T,
        ), points_R,
    )
    points_R = points_R + np.array(poserecord['translation']).reshape((-1, 1))

    # Transform into the ego vehicle frame for the timestamp of the image.
    points_R = points_R - \
        np.array(first_poserecord['translation']).reshape((-1, 1))
    points_R = np.dot(
        Quaternion(
            first_poserecord['rotation'],
        ).rotation_matrix.T, points_R,
    )

    # Transform into the camera.
    points_R = points_R - np.array(first_rec['translation']).reshape((-1, 1))
    points_R = np.dot(
        Quaternion(
            first_rec['rotation'],
        ).rotation_matrix.T, points_R,
    )
    points_R = points_R-points_T

    pose_global_frame = np.concatenate((points_T, points_R), axis=0)
    pose_global_frame = torch.from_numpy(pose_global_frame).float().cuda()
    return pose_global_frame


class network(nn.Module):
    def __init__(self, phase, base, extras, selector, final_net, use_gpu=config['cuda'], pose_estimator=None, max_objects=100, batch_size=1):
        super().__init__()
        self.phase = phase

        # vgg network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.selector = nn.ModuleList(selector)

        self.stacker2_bn = nn.BatchNorm2d(int(config['final_net'][0]/2))
        self.final_dp = nn.Dropout(0.5)
        self.final_net = nn.ModuleList(final_net)

        self.image_size = config['image_size']
        self.max_object = config['max_object']
        self.selector_channel = config['selector_channel']

        self.false_objects_column = None
        self.false_objects_row = None
        self.false_constant = config['false_constant']
        self.use_gpu = use_gpu
        self.max_objects = max_objects
        self.batch_size = batch_size
        self.criterion = Loss()
        self.estimator = pose_estimator
        self.estimator.cuda()

    def forward(self, joint_model, x_pre, x_next, l_pre, l_next, valid_pre=None, valid_next=None, current_features=None, next_features=None, pre_bbox=None, next_bbox=None, img_org_pre=None, img_org_next=None, current_cs_record=None, current_poserecord=None, next_cs_record=None, next_poserecord=None, first_cs_record=None, first_poserecord=None, camera_intrinsic=None):

        idx_pre = (valid_pre[0][0] == 1).nonzero()
        idx_next = (valid_next[0][0] == 1).nonzero()

        img_org_pre = img_org_pre[0].cpu().data.numpy()
        img_org_next = img_org_next[0].cpu().data.numpy()

        height, width, channels = img_org_pre.shape
        img_org_pre = Image.fromarray(np.uint8(img_org_pre))
        img_org_next = Image.fromarray(np.uint8(img_org_next))

        current_poses_features = torch.zeros(
            [self.batch_size, self.max_objects, 6], dtype=torch.float,
        ).cuda()
        next_poses_features = torch.zeros(
            [self.batch_size, self.max_objects, 6], dtype=torch.float,
        ).cuda()
        total_objects = 0
        total_pose_loss = 0
        if joint_model:
            for i in range(idx_pre.shape[0]-1):
                total_objects = total_objects+1
                box_index = idx_pre[i][0].item()

                x1 = pre_bbox[0][box_index][0].item()*width
                y1 = pre_bbox[0][box_index][1].item()*height
                x2 = pre_bbox[0][box_index][2].item()*width
                y2 = pre_bbox[0][box_index][3].item()*height
                Cx = 0.5*(x1+x2)
                Cy = 0.5*(y1+y2)
                box = (int(x1-10), int(y1-10), int(x2+10), int(y2+10))
                crop = img_org_pre.crop(box).convert('RGB')
                crop = np.array(crop)
                crop = np.transpose(crop, (2, 0, 1))
                nnorm = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                )
                crop = nnorm(torch.from_numpy(crop.astype(np.float32)))
                crop.unsqueeze_(dim=0)
                crop = crop.cuda()
                pred_r, pred_t = self.estimator(crop)
                pose_pred_glob = 0.001*convert_to_ref_frame(
                    Cx, Cy, pred_t, pred_r, current_cs_record,
                    current_poserecord, first_cs_record, first_poserecord, camera_intrinsic,
                )
                current_poses_features[0][box_index] = pose_pred_glob.view(6)
                target_pose = current_features[0][box_index][2:].cuda()
                target_pose = target_pose.unsqueeze(0).unsqueeze(2)
                loss, dis_r, dis_t = self.criterion(
                    pred_r, pred_t, target_pose,
                )
                total_pose_loss = total_pose_loss+loss

            for i in range(idx_next.shape[0]-1):
                total_objects = total_objects+1
                box_index = idx_next[i][0].item()
                x1 = next_bbox[0][box_index][0].item()*width
                y1 = next_bbox[0][box_index][1].item()*height
                x2 = next_bbox[0][box_index][2].item()*width
                y2 = next_bbox[0][box_index][3].item()*height

                box = (int(x1-10), int(y1-10), int(x2+10), int(y2+10))
                crop = img_org_next.crop(box).convert('RGB')
                crop = np.array(crop)
                crop = np.transpose(crop, (2, 0, 1))
                nnorm = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                )
                crop = nnorm(torch.from_numpy(crop.astype(np.float32)))
                crop.unsqueeze_(dim=0)
                crop = crop.cuda()
                pred_r, pred_t = self.estimator(crop)
                pose_pred_glob = 0.001*convert_to_ref_frame(
                    Cx, Cy, pred_t, pred_r, next_cs_record, next_poserecord, first_cs_record, first_poserecord, camera_intrinsic,
                )
                next_poses_features[0][box_index] = pose_pred_glob.view(6)
                target_pose = next_features[0][box_index][2:].cuda()
                target_pose = target_pose.unsqueeze(0).unsqueeze(2)
                loss, dis_r, dis_t = self.criterion(
                    pred_r, pred_t, target_pose,
                )
                total_pose_loss = total_pose_loss+loss
            total_pose_loss = total_pose_loss/total_objects

        sources_pre = list()
        sources_next = list()
        x_pre = self.forward_vgg(x_pre, self.vgg, sources_pre)
        x_next = self.forward_vgg(x_next, self.vgg, sources_next)
        x_pre = self.forward_extras(
            x_pre, self.extras,
            sources_pre,
        )
        x_next = self.forward_extras(
            x_next, self.extras,
            sources_next,
        )
        x_pre = self.forward_selector_stacker1(
            sources_pre, l_pre, self.selector, current_features,
        )

        x_next = self.forward_selector_stacker1(
            sources_next, l_next, self.selector, next_features,
        )
        x_pre = torch.cat((x_pre, current_poses_features), 2)
        x_next = torch.cat((x_next, next_poses_features), 2)
        # [B, N, N, C]
        x = self.forward_stacker2(
            x_pre, x_next,
        )
        x = self.final_dp(x)
        # [B, N, N, 1]
        x = self.forward_final(x, self.final_net)

        # add false unmatched row and column
        x = self.add_unmatched_dim(x)
        return x, total_pose_loss

    def forward_feature_extracter(self, joint_model, x, l, poses, image_org, detection_org, current_cs_record=None, current_poserecord=None, first_cs_record=None, first_poserecord=None, camera_intrinsic=None):
        '''
        extract features from the vgg layers and extra net
        :param x:
        :param l:
        :return: the features
        '''

        poses_features = torch.zeros(
            [self.batch_size, detection_org.shape[0], 6], dtype=torch.float,
        ).cuda()
        if joint_model:

            height, width, channels = image_org.shape
            image_org = Image.fromarray(np.uint8(image_org))

            for i in range(detection_org.shape[0]):

                x1 = detection_org[i][0]*width
                y1 = detection_org[i][1]*height
                x2 = detection_org[i][2]*width
                y2 = detection_org[i][3]*height
                x2 = x1+x2
                y2 = y1+y2
                Cx = 0.5*(x1+x2)
                Cy = 0.5*(y1+y2)
                box = (int(x1-10), int(y1-10), int(x2+10), int(y2+10))
                crop = image_org.crop(box).convert('RGB')

                crop = np.array(crop)
                crop = np.transpose(crop, (2, 0, 1))
                nnorm = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                )
                crop = nnorm(torch.from_numpy(crop.astype(np.float32)))
                crop.unsqueeze_(dim=0)
                crop = crop.cuda()
                pred_r, pred_t = self.estimator(crop)

                pose_pred_glob = 0.001*convert_to_ref_frame(
                    Cx, Cy, pred_t, pred_r, current_cs_record,
                    current_poserecord, first_cs_record, first_poserecord, camera_intrinsic,
                )
                poses_features[0][i] = pose_pred_glob.view(6)

        s = list()

        x = self.forward_vgg(x, self.vgg, s)
        x = self.forward_extras(x, self.extras, s)
        x = self.forward_selector_stacker1(s, l, self.selector, poses)
        x = torch.cat((x, poses_features), 2)

        return x, poses_features

    def get_similarity(self, image1, detection1, image2, detection2):
        feature1 = self.forward_feature_extracter(image1, detection1)
        feature2 = self.forward_feature_extracter(image2, detection2)
        return self.forward_stacker_features(feature1, feature2, False)

    def resize_dim(self, x, added_size, dim=1, constant=0):
        if added_size <= 0:
            return x
        shape = list(x.shape)
        shape[dim] = added_size
        if self.use_gpu:
            new_data = (torch.ones(shape)*constant).cuda()
        else:
            new_data = (torch.ones(shape) * constant)
        return torch.cat([x, new_data], dim=dim)

    def forward_stacker_features(self, xp, xn, fill_up_column=True):
        pre_rest_num = self.max_object - xp.shape[1]
        next_rest_num = self.max_object - xn.shape[1]
        pre_num = xp.shape[1]
        next_num = xn.shape[1]
        x = self.forward_stacker2(
            self.resize_dim(xp, pre_rest_num, dim=1),
            self.resize_dim(xn, next_rest_num, dim=1),
        )

        x = self.final_dp(x)
        # [B, N, N, 1]
        x = self.forward_final(x, self.final_net)
        x = x.contiguous()
        # add zero
        if next_num < self.max_object:
            x[0, 0, :, next_num:] = 0
        if pre_num < self.max_object:
            x[0, 0, pre_num:, :] = 0
        x = x[0, 0, :]
        # add false unmatched row and column
        x = self.resize_dim(x, 1, dim=0, constant=self.false_constant)
        x = self.resize_dim(x, 1, dim=1, constant=self.false_constant)

        x_f = F.softmax(x, dim=1)
        x_t = F.softmax(x, dim=0)
        # slice
        last_row, last_col = x_f.shape
        row_slice = list(range(pre_num)) + [last_row-1]
        col_slice = list(range(next_num)) + [last_col-1]
        x_f = x_f[row_slice, :]
        x_f = x_f[:, col_slice]
        x_t = x_t[row_slice, :]
        x_t = x_t[:, col_slice]

        x = (torch.zeros(pre_num, next_num+1))
#         x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        x[0:pre_num, 0:next_num] = (
            x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]
        ) / 2.0
        x[:, next_num:next_num+1] = x_f[:pre_num, next_num:next_num+1]
        if fill_up_column and pre_num > 1:
            x = torch.cat(
                [x, x[:, next_num:next_num+1].repeat(1, pre_num-1)], dim=1,
            )

        if self.use_gpu:
            y = x.data.cpu().numpy()
            # del x, x_f, x_t
            # torch.cuda.empty_cache()
        else:
            y = x.data.numpy()

        return y

    def forward_vgg(self, x, vgg, sources):
        for k in range(16):
            x = vgg[k](x)
        sources.append(x)

        for k in range(16, 23):
            x = vgg[k](x)
        sources.append(x)

        for k in range(23, 35):
            x = vgg[k](x)
        sources.append(x)
        return x

    def forward_extras(self, x, extras, sources):
        for k, v in enumerate(extras):
            # x = F.relu(v(x), inplace=True)        #done: relu is unnecessary.
            x = v(x)
            # done: should select the output of BatchNormalization (-> k%6==2)
            if k % 6 == 3:
                sources.append(x)
        return x

    def forward_selector_stacker1(self, sources, labels, selector, features):
        '''
        :param sources: [B, C, H, W]
        :param labels: [B, N, 1, 1, 2]
        :return: the connected feature
        '''
        sources = [
            F.relu(net(x), inplace=True) for net, x in zip(selector, sources)
        ]

        res = list()
        for label_index in range(labels.size(1)):
            label_res = list()
            for source_index in range(len(sources)):
                # [N, B, C, 1, 1]
                label_res.append(
                    # [B, C, 1, 1]
                    F.grid_sample(
                        sources[source_index],  # [B, C, H, W]
                        labels[:, label_index, :],  # [B, 1, 1, 2
                    ).squeeze(2).squeeze(2),
                )
            res.append(torch.cat(label_res, 1))

        return torch.stack(res, 1)

    def forward_stacker2(self, stacker1_pre_output, stacker1_next_output):

        stacker1_pre_output = stacker1_pre_output.unsqueeze(
            2,
        ).repeat(1, 1, self.max_object, 1).permute(0, 3, 1, 2)
        stacker1_next_output = stacker1_next_output.unsqueeze(
            1,
        ).repeat(1, self.max_object, 1, 1).permute(0, 3, 1, 2)
#         print('stacker1_pre_output.shape ',stacker1_pre_output.shape)
        stacker1_pre_output = self.stacker2_bn(
            stacker1_pre_output.contiguous(),
        )
        stacker1_next_output = self.stacker2_bn(
            stacker1_next_output.contiguous(),
        )

        output = torch.cat(
            [stacker1_pre_output, stacker1_next_output],
            1,
        )

        return output

    def forward_final(self, x, final_net):
        x = x.contiguous()
        for f in final_net:
            x = f(x)
        return x

    def add_unmatched_dim(self, x):
        if self.false_objects_column is None:
            self.false_objects_column = (
                torch.ones(
                    x.shape[0], x.shape[1], x.shape[2], 1,
                )
            ) * self.false_constant
            if self.use_gpu:
                self.false_objects_column = self.false_objects_column.cuda()
        x = torch.cat([x, self.false_objects_column], 3)

        if self.false_objects_row is None:
            self.false_objects_row = (
                torch.ones(
                    x.shape[0], x.shape[1], 1, x.shape[3],
                )
            ) * self.false_constant
            if self.use_gpu:
                self.false_objects_row = self.false_objects_row.cuda()
        x = torch.cat([x, self.false_objects_row], 2)
        return x

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(
                    base_file,
                    map_location=lambda storage, loc: storage,
                ),
            )
            print('Finished')
        else:
            print('Sorry only .pth and .pkl files supported.')


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5, conv6,
        nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True),
    ]
    return layers


def add_extras(cfg, i, batch_norm=True):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                conv2d = nn.Conv2d(
                    in_channels, cfg[k+1],
                    kernel_size=(1, 3)[flag],
                    stride=2,
                    padding=1,
                )
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(cfg[k+1]), nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(
                    in_channels, v,
                    kernel_size=(1, 3)[flag],
                )
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v), nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


def add_final(cfg, batch_normal=True):
    layers = []
    in_channels = int(cfg[0])
    layers += []
    # 1. add the 1:-2 layer with BatchNorm
    for v in cfg[1:-2]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        if batch_normal:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    # 2. add the -2: layer without BatchNorm for BatchNorm would make the output value normal distribution.
    for v in cfg[-2:]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return layers


def selector(vgg, extra_layers, batch_normal=True):
    '''
    batch_normal must be same to add_extras batch_normal
    '''
    selector_layers = []
    vgg_source = config['vgg_source']

    for k, v in enumerate(vgg_source):
        selector_layers += [
            nn.Conv2d(
                vgg[v-1].out_channels,
                config['selector_channel'][k],
                kernel_size=3,
                padding=1,
            ),
        ]
    if batch_normal:
        for k, v in enumerate(extra_layers[3::6], 3):
            selector_layers += [
                nn.Conv2d(
                    v.out_channels,
                    config['selector_channel'][k],
                    kernel_size=3,
                    padding=1,
                ),
            ]
    else:
        for k, v in enumerate(extra_layers[3::4], 3):
            selector_layers += [
                nn.Conv2d(
                    v.out_channels,
                    config['selector_channel'][k],
                    kernel_size=3,
                    padding=1,
                ),
            ]

    return vgg, extra_layers, selector_layers


def build_network(phase, use_gpu=config['cuda'], pose_estimator=None, max_objects=100, batch_size=1):
    '''
    create the SSJ Tracker Object
    :return: ssj tracker object
    '''
    if phase != 'test' and phase != 'train':
        print('Error: Phase not recognized')
        return

    base = config['base_net']
    extras = config['extra_net']
    final = config['final_net']

    return network(
        phase,
        *selector(
            vgg(base, 3),
            add_extras(extras, 1024),
        ),
        add_final(final),
        use_gpu, pose_estimator, max_objects, batch_size,
    )
