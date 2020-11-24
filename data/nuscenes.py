import os.path
import random

import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data

from config.config import config


class Node:
    def __init__(self, box, f, t, frame_id, next_fram_id=-1):
        self.box = box
        self.frame_id = frame_id
        self.next_frame_id = next_fram_id
        self.poses = f
        self.tokens = t


class Track:
    def __init__(self, id):
        self.nodes = list()
        self.id = id

    def add_node(self, n):
        if len(self.nodes) > 0:
            self.nodes[-1].next_frame_id = n.frame_id
        self.nodes.append(n)

    def get_node_by_index(self, index):
        return self.nodes[index]


class Tracks:
    def __init__(self):
        self.tracks = list()

    def add_node(self, node, id):
        node_add = False
        track_index = 0
        node_index = 0
        for t in self.tracks:
            if t.id == id:
                t.add_node(node)
                node_add = True
                track_index = self.tracks.index(t)
                node_index = t.nodes.index(node)
                break
        if not node_add:
            t = Track(id)
            t.add_node(node)
            self.tracks.append(t)
            track_index = self.tracks.index(t)
            node_index = t.nodes.index(node)

        return track_index, node_index

    def get_track_by_index(self, index):
        return self.tracks[index]


class GTSingleParser:
    def __init__(
        self, folder,
        min_gap=config['min_gap_frame'],
        max_gap=config['max_gap_frame'],
    ):
        self.min_gap = min_gap
        self.max_gap = max_gap
        # 1. get the gt path and image folder
        gt_file_path = os.path.join(folder, 'gt/gt.txt')
        self.folder = folder

        # 2. read the gt data
        gt_file = pd.read_csv(gt_file_path, header=None, sep=' ')
        gt_group = gt_file.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        self.max_frame_index = max(gt_group_keys)
        # 3. update tracks
        self.tracks = Tracks()
        self.recorder = {}
        for key in gt_group_keys:
            det = gt_group.get_group(key).values
            ids = np.array(det[:, 1]).astype(int)
            poses = np.array(det[:, 6:12]).astype(float)
            tokens = det[:, 12:]
            det = np.array(det[:, 2:6])
            det[:, 2:4] += det[:, :2]

            self.recorder[key-1] = list()
            # 3.1 update tracks
            for id, d, f, tok in zip(ids, det, poses, tokens):
                node = Node(d, f, tok,  key-1)
                track_index, node_index = self.tracks.add_node(node, id)
                self.recorder[key-1].append((track_index, node_index))

    def _getimage(self, frame_index):
        image_path = os.path.join(
            self.folder, f'img1/{frame_index}.jpg',
        )
        return cv2.imread(image_path)

    def get_item(self, frame_index):
        '''
        get the current_image, current_boxes, next_image, next_boxes, labels from the frame_index
        :param frame_index:
        :return: current_image, current_boxes, next_image, next_boxes, labels
        '''
#         print('get item')
        if not frame_index in self.recorder:
            return None, None, None, None, None,  None, None, None, None
        # get current_image, current_box, next_image, next_box and labels
        current_image = self._getimage(frame_index)
        current_boxes = list()
        current_poses = list()
        current_tokens = list()
        current = self.recorder[frame_index]
        next_frame_indexes = list()
        current_track_indexes = list()
        # 1. get current box
        for track_index, node_index in current:
            t = self.tracks.get_track_by_index(track_index)
            n = t.get_node_by_index(node_index)
            current_boxes.append(n.box)
            current_poses.append(n.poses)
            current_track_indexes.append(track_index)
            current_tokens.append(n.tokens)
            if n.next_frame_id != -1:
                next_frame_indexes.append(n.next_frame_id)
        current_tokens = current_tokens[0][1:]
        if len(next_frame_indexes) == 0:
            return None, None, None, None, None,  None, None, None, None
        if len(next_frame_indexes) == 1:
            next_frame_index = next_frame_indexes[0]
        else:
            max_next_frame_index = max(next_frame_indexes)
            is_choose_farest = bool(random.getrandbits(1))
            if is_choose_farest:
                next_frame_index = max_next_frame_index
            else:
                next_frame_index = random.choice(next_frame_indexes)
                gap_frame = random.randint(self.min_gap, self.max_gap)
                temp_frame_index = next_frame_index + gap_frame
                choice_gap = list(range(self.min_gap, self.max_gap))
                if self.min_gap != 0:
                    choice_gap.append(0)
                while not temp_frame_index in self.recorder:
                    gap_frame = random.choice(choice_gap)
                    temp_frame_index = next_frame_index + gap_frame
                next_frame_index = temp_frame_index

        # 3. get next image
        next_image = self._getimage(next_frame_index)

        # 4. get next frame boxes
        next = self.recorder[next_frame_index]
        next_boxes = list()
        next_poses = list()
        next_tokens = list()
        next_track_indexes = list()
        for track_index, node_index in next:
            t = self.tracks.get_track_by_index(track_index)
            next_track_indexes.append(track_index)
            n = t.get_node_by_index(node_index)
            next_boxes.append(n.box)
            next_poses.append(n.poses)
            next_tokens.append(n.tokens)
        next_tokens = next_tokens[0][1:]

        # 5. get the labels
        current_track_indexes = np.array(current_track_indexes)
        next_track_indexes = np.array(next_track_indexes)
        labels = np.repeat(np.expand_dims(np.array(current_track_indexes), axis=1), len(next_track_indexes), axis=1) == np.repeat(
            np.expand_dims(np.array(next_track_indexes), axis=0), len(current_track_indexes), axis=0,
        )

        # 6. return all values
        # 6.1 change boxes format
        current_boxes = np.array(current_boxes)
        next_boxes = np.array(next_boxes)
        current_poses = np.array(current_poses)
        next_poses = np.array(next_poses)
        # 6.2 return the corresponding values
        # print(current_image)
        return current_image, current_boxes, next_image, next_boxes, labels, current_poses, next_poses, current_tokens, next_tokens

    def __len__(self):
        return self.max_frame_index


class GTParser:
    def __init__(
        self, nuScenes_root=config['nuscenes_data_root'],
        type=config['type'],
    ):
        # analsis all the folder in nuScenes_root
        # 1. get all the folders
        nuScenes_root = os.path.join(nuScenes_root, type)
        all_folders = sorted(
            [
                os.path.join(nuScenes_root, i) for i in os.listdir(nuScenes_root)
                if os.path.isdir(os.path.join(nuScenes_root, i))
            ],
        )

        # print(all_folders)
        # 2. create single parser
        self.parsers = [
            GTSingleParser(folder) for folder in all_folders if os.stat(
                os.path.join(folder, 'gt/gt.txt'),
            ).st_size > 0
        ]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers]
        self.len = sum(self.lens)

    def __len__(self):
        # get the length of all the matching frame
        #         print(self.len)
        return self.len

    def __getitem__(self, item):
        if item < 0:
            return None, None, None, None, None, None, None, None, None
        # 1. find the parser
        total_len = 0
        index = 0
        current_item = item
        for l in self.lens:
            total_len += l
            if item < total_len:
                break
            else:
                index += 1
                current_item -= l

        # 2. get items
        if index >= len(self.parsers):
            return None, None, None, None, None, None, None, None, None
        return self.parsers[index].get_item(current_item)


class TrainDataset(data.Dataset):
    '''
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    '''

    def __init__(
        self,
        nuScenes_root=config['nuscenes_data_root'],
        transform=None,
        type=config['type'],
        max_object=config['max_object'],

    ):
        # 1. init all the variables
        self.nuScenes_root = nuScenes_root
        self.transform = transform
        self.type = type
        self.max_object = max_object

        # 2. init GTParser
        self.parser = GTParser(self.nuScenes_root)

    def __getitem__(self, item):
        current_image, current_box, next_image, next_box, labels, current_poses, next_poses, current_tokens, next_tokens = self.parser[
            item
        ]
        while current_image is None:
            item = item + \
                random.randint(
                    -config['max_gap_frame'],
                    config['max_gap_frame'],
                )
            if item < 0:
                item = item + random.randint(100, 300)
            if item > 11000:
                item = item - random.randint(100, 300)
            current_image, current_box, next_image, next_box, labels, current_poses, next_poses, current_tokens, next_tokens = self.parser[
                item
            ]
#             print('None processing.')
#             item=item+15+random.randint(0, config['max_gap_frame'])

            #print('None processing.')
        if self.transform is None:
            return current_image, current_box, next_image, next_box, labels, current_poses, next_poses, current_tokens, next_tokens

        # change the label to max_object x max_object
        #print('labels.shape[0] ',labels.shape[0])
        labels = np.pad(
            labels,
            [
                (0, self.max_object - labels.shape[0]),
                (0, self.max_object - labels.shape[1]),
            ],
            mode='constant',
            constant_values=0,
        )
        return self.transform(current_image, next_image, current_box, next_box, labels, current_poses, next_poses, current_tokens, next_tokens)

    def __len__(self):
        return len(self.parser)
