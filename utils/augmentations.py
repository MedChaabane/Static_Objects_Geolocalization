import types

import cv2
import numpy as np
import torch
from numpy import random

import utils.operation as op
from config.config import config

# large part of this code is from https://github.com/shijieS/SST
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose:
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        for t in self.transforms:
            img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens = \
                t(
                    img_pre, img_next, boxes_pre, boxes_next, labels, current_features,
                    next_features, img_org_pre, img_org_next, current_tokens, next_tokens,
                )
        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class Lambda:
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts:
    def __call__(
        self, img_pre, img_next,
        boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None,
    ):
        return img_pre.astype(np.float32), img_next.astype(np.float32), \
            boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        img_pre = img_pre.astype(np.float32)
        img_pre -= self.mean
        img_next = img_next.astype(np.float32)
        img_next -= self.mean
        return img_pre.astype(np.float32), img_next.astype(np.float32), boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre.astype(np.float32), img_org_next.astype(np.float32), current_tokens, next_tokens


class ToPercentCoords:
    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        height, width, channels = img_pre.shape
        boxes_pre[:, 0] /= width
        boxes_pre[:, 2] /= width
        boxes_pre[:, 1] /= height
        boxes_pre[:, 3] /= height

        boxes_next[:, 0] /= width
        boxes_next[:, 2] /= width
        boxes_next[:, 1] /= height
        boxes_next[:, 3] /= height

        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class Resize:
    def __init__(self, size=config['image_size']):
        self.size = size

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):

        img_pre = cv2.resize(img_pre, (self.size, self.size))
        img_next = cv2.resize(img_next, (self.size, self.size))

        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class RandomSaturation:
    def __init__(self, lower=config['lower_saturation'], upper=config['upper_saturation']):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img_pre[:, :, 1] = img_pre[:, :, 1] * alpha
            img_next[:, :, 1] = img_next[:, :, 1] * alpha

        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class RandomHue:
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img_pre[:, :, 0] = img_pre[:, :, 0] + delta
            img_pre[:, :, 0][img_pre[:, :, 0] > 360.0] -= 360.0
            img_pre[:, :, 0][img_pre[:, :, 0] < 0.0] += 360.0

            img_next[:, :, 0] = img_next[:, :, 0] + delta
            img_next[:, :, 0][img_next[:, :, 0] > 360.0] -= 360.0
            img_next[:, :, 0][img_next[:, :, 0] < 0.0] += 360.0
        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class RandomLightingNoise:
    def __init__(self):
        self.perms = (
            (0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0),
        )

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            img_pre = shuffle(img_pre)
            img_next = shuffle(img_next)
        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class ConvertColor:
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2HSV)
            img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            img_pre = cv2.cvtColor(img_pre, cv2.COLOR_HSV2BGR)
            img_next = cv2.cvtColor(img_next, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class RandomContrast:
    def __init__(self, lower=config['lower_contrast'], upper=config['upper_constrast']):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    # expects float image
    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img_pre = img_pre*alpha
            img_next = img_next*alpha
        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img_pre = img_pre + delta
            img_next = img_next + delta
        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class RandomSampleCrop:
    """Crop
    Arguments:
        mode (float tuple): the min and max jaccard overlaps
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.7, None),
            (0.8, None),
            (0.85, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def crop(self, image, boxes, labels, mode, min_iou, max_iou, w, h, left, top, isPre=True):

        current_image = image
        # convert to integer rect x1,y1,x2,y2
        rect = np.array([int(left), int(top), int(left + w), int(top + h)])

        #  IoU (jaccard calculateoverlap) b/t the cropped and gt boxes

        overlap = jaccard_numpy(
            boxes[:, :],
            rect,
        )

        # is min and max overlap constraint satisfied? if not try again
        if overlap.min() < min_iou and max_iou < overlap.max():
            return None

        # cut the crop from the image
        current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

        # keep overlap with gt box IF center in sampled patch
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

        # mask in all gt boxes that above and to the left of centers
        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

        # mask in all gt boxes that under and to the right of centers
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

        # mask in that both m1 and m2 are true
        mask = m1 * m2

        # have any valid boxes? try again if not
        if not mask.any():
            return None

        # take only matching gt boxes
        current_boxes = boxes[mask, :].copy()

        # pading the mask
        mask = np.pad(
            mask,
            (0, config['max_object'] - len(mask)),
            'constant',
        )
        mask = (mask == False)

        # take only matching gt labels
        # Important change: instead assign to zero, we should delete the row
        current_labels = labels
        h, w = labels.shape
        if isPre:
            current_labels = current_labels[np.logical_not(mask), :]
            current_labels = np.pad(
                current_labels, [
                    [0, h-current_labels.shape[0]], [0, 0],
                ], mode='constant', constant_values=0.0,
            )
        else:
            current_labels = current_labels[:, np.logical_not(mask)]
            current_labels = np.pad(
                current_labels, [
                    [0, 0], [
                        0, w - current_labels.shape[1],
                    ],
                ], mode='constant', constant_values=0.0,
            )

        # should we use the box left and top corner or the crop's
        current_boxes[:, :2] = np.maximum(
            current_boxes[:, :2],
            rect[:2],
        )
        # adjust to crop (by substracting crop's left,top)
        current_boxes[:, :2] -= rect[:2]

        current_boxes[:, 2:] = np.minimum(
            current_boxes[:, 2:],
            rect[2:],
        )
        # adjust to crop (by substracting crop's left,top)
        current_boxes[:, 2:] -= rect[:2]

        return current_image, current_boxes, current_labels

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        height, width, _ = img_pre.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):

                w = random.uniform(0.8 * width, width)
                h = random.uniform(0.8 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                res_pre = self.crop(
                    img_pre, boxes_pre, labels, mode, min_iou, max_iou, w, h, left, top, isPre=True,
                )
                if res_pre is None:
                    continue

                res_next = self.crop(
                    img_next, boxes_next, res_pre[2], mode, min_iou, max_iou, w, h, left, top, isPre=False,
                )
                if res_next is None:
                    continue
                else:
                    return res_pre[0], res_next[0], res_pre[1], res_next[1], res_next[2], current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class Expand:
    def __init__(self, mean=config['mean_pixel']):
        self.mean = mean

    def expand(self, image, height, width, depth, ratio, left, top):
        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype,
        )
        expand_image[:, :, :] = self.mean
        expand_image[
            int(top):int(top + height), int(left):int(left + width)
        ] = image
        return expand_image

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        if random.randint(2):
            return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens

        height, width, depth = img_pre.shape
        # new: adjust the max_expand to 2.0 (orgin is 4.0)
        ratio = random.uniform(1, config['max_expand'])
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        img_pre = self.expand(img_pre, height, width, depth, ratio, left, top)
        img_next = self.expand(
            img_next, height, width,
            depth, ratio, left, top,
        )

        boxes_pre = boxes_pre.copy()
        boxes_pre[:, :2] += (int(left), int(top))
        boxes_pre[:, 2:] += (int(left), int(top))

        boxes_next = boxes_next.copy()
        boxes_next[:, :2] += (int(left), int(top))
        boxes_next[:, 2:] += (int(left), int(top))

        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class RandomMirror:
    def mirror(self, image, boxes):
        _, width, _ = image.shape
        image = np.array(image[:, ::-1])
        boxes = boxes.copy()
        boxes[:, 0] = width - boxes[:, 0]
        boxes[:, 2] = width - boxes[:, 2]
        return image, boxes

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        if random.randint(2):
            res_pre = self.mirror(img_pre, boxes_pre)
            res_next = self.mirror(img_next, boxes_next)
            img_pre = res_pre[0]
            img_next = res_next[0]
            boxes_pre = res_pre[1]
            boxes_next = res_next[1]

        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class SwapChannels:
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """

        image = image[:, :, self.swaps]
        return image


class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        im_pre = img_pre.copy()
        im_next = img_next.copy()
        img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens = \
            self.rand_brightness(
                im_pre, im_next, boxes_pre, boxes_next, labels, current_features,
                next_features, img_org_pre, img_org_next, current_tokens, next_tokens,
            )
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])

        img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens = distort(
            im_pre, im_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens,
        )

        return self.rand_light_noise(im_pre, im_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens)


class ResizeShuffleBoxes:
    def show_matching_hanlded_rectangle(self, img_pre, img_next, boxes_pre, boxes_next, labels):
        img_pre = (img_pre + np.array(config['mean_pixel'])).astype(np.uint8)
        img_next = (img_next + np.array(config['mean_pixel'])).astype(np.uint8)
        h = img_pre.shape[0]

        return op.show_matching_rectangle(img_pre, img_next, boxes_pre[:, :]*h, boxes_next[:, :]*h, labels)

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        def resize_f(boxes): return \
            (
                boxes.shape[0],
                np.vstack((
                    boxes,
                    np.full(
                        (
                            config['max_object'] - len(boxes),
                            boxes.shape[1],
                        ),
                        np.inf,
                    ),
                )),
            )

        # show the shuffling result

        cv2.imwrite(
            'gt_matching.jpg', self.show_matching_hanlded_rectangle(
                img_pre, img_next, boxes_pre, boxes_next, labels,
            ),
        )
        size_pre, boxes_pre = resize_f(boxes_pre)
        size_next, boxes_next = resize_f(boxes_next)

        size_pre1, current_features = resize_f(current_features)
        size_next1, next_features = resize_f(next_features)

        indexes_pre = np.arange(config['max_object'])
        indexes_next = np.arange(config['max_object'])
        np.random.shuffle(indexes_pre)
        np.random.shuffle(indexes_next)

        boxes_pre = boxes_pre[indexes_pre, :]
        boxes_next = boxes_next[indexes_next, :]

        current_features = current_features[indexes_pre, :]
        next_features = next_features[indexes_next, :]

        labels = labels[indexes_pre, :]
        labels = labels[:, indexes_next]

        mask_pre = indexes_pre < size_pre
        mask_next = indexes_next < size_next

        # add false object label
        false_object_pre = (labels.sum(1) == 0).astype(
            float,
        )       # should consider unmatched object
        false_object_pre[np.logical_not(mask_pre)] = 0.0

        false_object_next = (labels.sum(0) == 0).astype(
            float,
        )  # should consider unmatched object
        false_object_next[np.logical_not(mask_next)] = 0.0

        false_object_pre = np.expand_dims(false_object_pre, axis=1)
        labels = np.concatenate((labels, false_object_pre), axis=1)  # 60x61

        false_object_next = np.append(false_object_next, [0])
        false_object_next = np.expand_dims(false_object_next, axis=0)
        labels = np.concatenate((labels, false_object_next), axis=0)  # 60x61

        mask_pre = np.append(mask_pre, [True])  # 61
        mask_next = np.append(mask_next, [True])  # 61
        return img_pre, img_next, \
            [boxes_pre, mask_pre], \
            [boxes_next, mask_next], \
            labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class FormatBoxes:
    '''
    note: format the label in order to input into the selector net.
    '''

    def __init__(self, keep_box=False):
        self.keep_box = keep_box

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):
        '''
        boxes_pre: [N, 4]
        '''
        if not self.keep_box:
            # convert the center to [-1, 1]
            def f(boxes): return np.expand_dims(
                np.expand_dims(
                    (boxes[:, :2] + boxes[:, 2:]) - 1,
                    axis=1,
                ),
                axis=1,
            )
        else:
            def f(boxes): return np.expand_dims(
                np.expand_dims(
                    np.concatenate(
                        [(boxes[:, :2] + boxes[:, 2:]) - 1, boxes[:, 2:6]], axis=1,
                    ),
                    axis=1,
                ),
                axis=1,
            )

        # remove inf
        box_pre = np.copy(boxes_pre[0])
        box_next = np.copy(boxes_next[0])
        box_pre[box_pre == np.inf] = -1.0
        box_pre[box_pre == np.inf] = -1.0
        boxes_pre[0] = f(boxes_pre[0])
        boxes_pre[0][boxes_pre[0] == np.inf] = 1.5

        boxes_next[0] = f(boxes_next[0])
        boxes_next[0][boxes_next[0] == np.inf] = 1.5

        current_features[current_features == np.inf] = 0.0
        next_features[next_features == np.inf] = 0.0
        boxes_pre = [boxes_pre[0], boxes_pre[1], box_pre]
        boxes_next = [boxes_next[0], boxes_next[1], box_next]
        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class ToTensor:
    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None, current_features=None, next_features=None, img_org_pre=None, img_org_next=None, current_tokens=None, next_tokens=None):

        img_pre = torch.from_numpy(img_pre.astype(np.float32)).permute(2, 0, 1)
        img_next = torch.from_numpy(
            img_next.astype(np.float32),
        ).permute(2, 0, 1)

        boxes_pre[0] = torch.from_numpy(boxes_pre[0].astype(float))
        boxes_pre[1] = torch.from_numpy(boxes_pre[1].astype(np.uint8))
        boxes_pre[2] = torch.from_numpy(boxes_pre[2].astype(float))

        boxes_next[0] = torch.from_numpy(boxes_next[0].astype(float))
        boxes_next[1] = torch.from_numpy(boxes_next[1].astype(np.uint8))
        boxes_next[2] = torch.from_numpy(boxes_next[2].astype(float))
        labels = torch.from_numpy(labels).unsqueeze(0)

        current_features = torch.from_numpy(current_features)
        next_features = torch.from_numpy(next_features)
        img_org_pre = torch.from_numpy(img_org_pre)
        img_org_next = torch.from_numpy(img_org_next)
        token = 'bfbts'
        return img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_org_pre, img_org_next, current_tokens, next_tokens


class SSJAugmentation:
    def __init__(self, size=config['image_size'], mean=config['mean_pixel'], type=config['type']):
        self.mean = mean
        self.size = size
        if type == 'train':
            self.augment = Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
                ResizeShuffleBoxes(),
                FormatBoxes(),
                ToTensor(),
            ])
        elif type == 'test':
            self.augment = Compose([
                ConvertFromInts(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
                ResizeShuffleBoxes(),
                FormatBoxes(keep_box=True),
                ToTensor(),
            ])
        else:
            raise NameError(
                'config type is wrong, should be choose from (train, test)',
            )

    def __call__(self, img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, current_tokens, next_tokens):
        return self.augment(img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_pre, img_next, current_tokens, next_tokens)


class SSJEvalAugment:
    def __init__(self, size=config['image_size'], mean=config['mean_pixel']):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            ResizeShuffleBoxes(),
            FormatBoxes(),
            ToTensor(),
        ])

    def __call__(self, img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, current_tokens, next_tokens):
        return self.augment(img_pre, img_next, boxes_pre, boxes_next, labels, current_features, next_features, img_pre, img_next, current_tokens, next_tokens)


def collate_fn(batch):
    img_pre = []
    img_next = []
    img_org_pre = []
    img_org_next = []
    boxes_pre = []
    boxes_next = []
    box_pre = []
    box_next = []
    labels = []
    indexes_pre = []
    indexes_next = []
    current_features = []
    next_features = []
    for sample in batch:
        img_pre.append(sample[0])
        img_next.append(sample[1])
        img_org_pre.append(sample[7])
        img_org_next.append(sample[8])
        boxes_pre.append(sample[2][0].float())
        boxes_next.append(sample[3][0].float())
        box_pre.append(sample[2][2].float())
        box_next.append(sample[3][2].float())
        labels.append(sample[4].float())
        indexes_pre.append(sample[2][1].byte())
        indexes_next.append(sample[3][1].byte())
        current_features.append(sample[5].float())
        next_features.append(sample[6].float())
    return torch.stack(img_pre, 0), torch.stack(img_next, 0), \
        torch.stack(boxes_pre, 0), torch.stack(boxes_next, 0), \
        torch.stack(labels, 0), \
        torch.stack(indexes_pre, 0).unsqueeze(1), \
        torch.stack(indexes_next, 0).unsqueeze(1), torch.stack(current_features, 0), torch.stack(next_features, 0), torch.stack(
            box_pre, 0,
        ), torch.stack(box_pre, 0), torch.stack(img_org_pre, 0), torch.stack(img_org_next, 0), sample[9], sample[10]
