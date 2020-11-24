import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from config.config import config


def show_circle(img, boxes, valid):
    batch_size = img.shape[0]
    images = list()
    for i in range(batch_size):
        img1 = img[i, :].permute(1, 2, 0).data
        img1 = img1.cpu().numpy() + config['mean_pixel']
        valid1 = valid[i, 0, :-1].data.cpu().numpy()
        boxes1 = boxes[i, :, 0, 0, :].data.cpu().numpy()
        boxes1 = boxes1[valid1 == 1]
        img1 = np.clip(img1, 0, 255).astype(np.uint8).copy()
        for b in boxes1:
            img1 = cv2.circle(
                img1, tuple(
                    ((b + 1) / 2.0 * config['image_size']).astype(int),
                ), 20, [255, 0, 0], thickness=3,
            )

        img1 = torch.from_numpy(
            img1.astype(
                np.float,
            ) - config['mean_pixel'],
        ).permute(2, 0, 1)
        images.append(img1)
    return torch.stack(images, dim=0)


def getProperbility(input, target, mask0, mask1):
    mask_pre = mask0[:, :, :]

    mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, config['max_object'] + 1)
    mask1 = mask1.unsqueeze(2).repeat(1, 1, config['max_object'] + 1, 1)
    mask0 = (mask0.data)
    mask1 = (mask1.data)
    target = (target.byte().data)

    if config['cuda']:
        mask0 = mask0.cuda()
        mask1 = mask1.cuda()

    # add false object to the input matrix

    mask_region = (mask0 * mask1).float()  # the valid position mask

    mask_region_pre = mask_region.clone()  # note: should use clone (fix bug)
    mask_region_pre[:, :, config['max_object'], :] = 0
    # note: should use clone (fix bug) finnally
    mask_region_next = mask_region.clone()
    mask_region_next[:, :, :, config['max_object']] = 0
    mask_region_union = mask_region_pre * mask_region_next

    input_pre = nn.Softmax(dim=3)(mask_region_pre * input)
    input_next = nn.Softmax(dim=2)(mask_region_next * input)

    # input = torch.max(torch.stack([input_pre, input_next], 4), 4)[0]
    input = input_pre.clone()
    input[:, :, :-1, :-1] = (
        input_pre[:, :, :-1, :-1] +
        input_next[:, :, :-1, :-1]
    ) / 2.0
    # input = (input_pre + input_next) / 2.0
    # target = (mask_region * target).float()
    target = target.float()
    target_pre = mask_region_pre * target
    target_next = mask_region_next * target
    target_union = mask_region_union * target
    target_num = target.sum()
    target_num_pre = target_pre.sum()
    target_num_next = target_next.sum()
    target_num_union = target_union.sum()
    if int(target_num_pre.data[0]):
        loss_pre = - (target_pre * torch.log(input_pre)).sum() / target_num_pre
    else:
        loss_pre = - (target_pre * torch.log(input_pre)).sum()
    if int(target_num_next.data[0]):
        loss_next = - (
            target_next * torch.log(input_next)
        ).sum() / target_num_next
    else:
        loss_next = - (target_next * torch.log(input_next)).sum()
    if int(target_num_union.data[0]):
        loss_similarity = (
            0.5 * target_union * (
                torch.abs((1 - input_pre) ** 2 - (1 - input_next) ** 2)
            )
        ).sum() / target_num
    else:
        loss_similarity = (
            0.5 * target_union *
            (torch.abs((1 - input_pre) ** 2 - (1 - input_next) ** 2))
        ).sum()

    loss = (loss_pre + loss_next)/2.0 + loss_similarity

    _, indexes = input.max(3)
    _, indexes_ = target_pre.max(3)
    mask_pre_num = mask_pre.sum().data[0] - 1
    if mask_pre_num:
        accuracy = (
            indexes[mask_pre][:-1] == indexes_[mask_pre]
            [:-1]
        ).float().sum() / mask_pre_num
    else:
        accuracy = (
            indexes[mask_pre][:-1] ==
            indexes_[mask_pre][:-1]
        ).float().sum() + 1

    indexes_pre = mask_pre[0, 0, :-1].nonzero()[:, 0]
    indexes_next = indexes[mask_pre][:-1]

    print('accuracy:' + str(accuracy))
    return loss, accuracy, input, input_pre, input_next, indexes_pre, indexes_next


def show_matching_rectangle(img_pre, img_next, boxes_pre, boxes_next, labels, show_matching=True):
    img_p = img_pre.copy()
    img_n = img_next.copy()
    for box in boxes_pre[:, 0:4]:
        img_p = cv2.rectangle(
            img_p, tuple(box[:2].astype(int)), tuple(
                (box[2:4]).astype(int),
            ), (255, 0, 0), 2,
        )

    for box in boxes_next[:, 0:4]:
        img_n = cv2.rectangle(
            img_n, tuple(box[:2].astype(int)), tuple(
                (box[2:4]).astype(int),
            ), (255, 0, 0), 2,
        )

    h, w, c = img_p.shape
    img = np.concatenate([img_p, img_n], axis=0)
    if show_matching:
        rows, cols = np.nonzero(labels)
        for r, c in zip(rows, cols):
            box_p = boxes_pre[r, 0:4]
            box_n = boxes_next[c, 0:4]
            center_p = (box_p[:2] + box_p[2:4]) / 2.0
            center_n = (box_n[:2] + box_n[2:4]) / 2.0 + np.array([0, h])
            cv2.line(
                img, tuple(center_p.astype(int)), tuple(center_n.astype(int)),
                (
                    (int)(np.random.randn() * 255), (int)
                    (np.random.randn() * 255), (int)(np.random.randn() * 255),
                ),
                2,
            )

    return img


def show_batch_circle_image(img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next, indexes, iteration=-1, img_path=config['log_folder']):
    batch_size = img_pre.shape[0]
    images = list()
    h = config['image_size']
    gap = 20 / config['image_size']
    for i in range(batch_size):
        img1 = img_pre[i, :].permute(1, 2, 0).data
        img1 = img1.cpu().numpy() + config['mean_pixel']
        valid1 = valid_pre[i, 0, :-1].data.cpu().numpy()
        boxes1 = boxes_pre[i, :, 0, 0, :].data.cpu().numpy()
        boxes1 = boxes1[valid1 == 1]
        img1 = np.clip(img1, 0, 255).astype(np.uint8).copy()

        index = indexes[i, 0, :].data.cpu().numpy()[valid1 == 1]

        img2 = img_next[i, :].permute(1, 2, 0).data
        img2 = img2.cpu().numpy() + config['mean_pixel']
        valid2 = valid_next[i, 0, :-1].data.cpu().numpy()
        boxes2 = boxes_next[i, :, 0, 0, :].data.cpu().numpy()
        boxes2 = boxes2[valid2 == 1]
        img2 = np.clip(img2, 0, 255).astype(np.uint8).copy()

        # draw all circle
        for b in boxes1:
            img1 = cv2.circle(
                img1, tuple(
                    ((b + 1) / 2.0 * config['image_size']).astype(int),
                ), 20, [0, 0, 255],
                thickness=3,
            )

        for b in boxes2:
            img2 = cv2.circle(
                img2, tuple(
                    ((b + 1) / 2.0 * config['image_size']).astype(int),
                ), 20, [0, 0, 255],
                thickness=3,
            )

        gap_pixel = int(gap * config['image_size'])
        H, W, C = img1.shape
        img = np.ones((2*H+gap_pixel, W, C), dtype=np.uint8)*255
        img[:H, :W, :] = img1
        img[gap_pixel+H:, :] = img2
        # connect the boxes
        # draw the connected boxes
        for j, b1 in enumerate(boxes1):
            if index[j] >= config['max_object']:
                continue

            color = tuple((np.random.rand(3) * 255).astype(int).tolist())
            start_pt = tuple(
                ((b1 + 1) / 2.0 * config['image_size']).astype(int),
            )
            b2 = boxes_next[i, :, 0, 0, :].data.cpu().numpy()[index[j]]
            end_pt = tuple(((b2 + 1) / 2.0 * config['image_size']).astype(int))
            end_pt = (end_pt[0], end_pt[1]+h+gap_pixel)
            img = cv2.circle(img, start_pt, 20, color, thickness=3)
            img = cv2.circle(img, end_pt, 20, color, thickness=3)
            img = cv2.line(img, start_pt, end_pt, color, thickness=3)

#         if 'save_images_folder' in config and iteration!=-1:
        cv2.imwrite(
            os.path.join(
                img_path, f'{iteration:06}.png',
            ), img,
        )

        # img = torch.from_numpy(img.astype(np.float) - config['mean_pixel']).permute(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float)).permute(2, 0, 1)
        images.append(img)
    return torch.stack(images, dim=0)
