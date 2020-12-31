import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.optimize import linear_sum_assignment

from config.config import config
from layer.matching_network import build_network

# A large part of this code is from https://github.com/shijieS/SST , many thanks to their wonderful work. 
class TrackUtil:
    @staticmethod
    def convert_detection(detection):
        '''
        transform the current detection center to [-1, 1]
        :param detection: detection
        :return: translated detection
        '''
        # get the center, and format it in (-1, 1)
        center = (2 * detection[:, 0:2] + detection[:, 2:4]) - 1.0
        center = torch.from_numpy(center.astype(float)).float()
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)

        if TrackerConfig.cuda:
            return (center.cuda())
        return (center)

    @staticmethod
    def convert_image(image):
        '''
        transform image to the FloatTensor (1, 3,size, size)
        :param image: same as update parameter
        :return: the transformed image FloatTensor (i.e. 1x3x900x900)
        '''
        image = cv2.resize(image, TrackerConfig.image_size).astype(np.float32)
        image -= TrackerConfig.mean_pixel
        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        image.unsqueeze_(dim=0)
        if TrackerConfig.cuda:
            return (image.cuda())
        return (image)

    @staticmethod
    def get_iou(pre_boxes, next_boxes):
        h = len(pre_boxes)
        w = len(next_boxes)
        if h == 0 or w == 0:
            return []

        iou = np.zeros((h, w), dtype=float)
        for i in range(h):
            b1 = np.copy(pre_boxes[i, :])
            b1[2:] = b1[:2] + b1[2:]
            for j in range(w):
                b2 = np.copy(next_boxes[j, :])
                b2[2:] = b2[:2] + b2[2:]
                delta_h = min(b1[2], b2[2]) - max(b1[0], b2[0])
                delta_w = min(b1[3], b2[3])-max(b1[1], b2[1])
                if delta_h < 0 or delta_w < 0:
                    expand_area = (
                        max(b1[2], b2[2]) - min(b1[0], b2[0])
                    ) * (max(b1[3], b2[3]) - min(b1[1], b2[1]))
                    area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + \
                        (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou[i, j] = -(expand_area - area) / area
                else:
                    overlap = delta_h * delta_w
                    area = (b1[2]-b1[0])*(b1[3]-b1[1]) + \
                        (b2[2]-b2[0])*(b2[3]-b2[1]) - max(overlap, 0)
                    iou[i, j] = overlap / area

        return iou

    @staticmethod
    def get_node_similarity(n1, n2, frame_index, recorder):
        if n1.frame_index > n2.frame_index:
            n_max = n1
            n_min = n2
        elif n1.frame_index < n2.frame_index:
            n_max = n2
            n_min = n1
        else:  # in the same frame_index
            return None

        f_max = n_max.frame_index
        f_min = n_min.frame_index

        # not recorded in recorder
        if frame_index - f_min >= TrackerConfig.max_track_node:
            return None

        return recorder.all_similarity[f_max][f_min][n_min.id, n_max.id]

    @staticmethod
    def get_merge_similarity(t1, t2, frame_index, recorder):
        '''
        Get the similarity between two tracks
        :param t1: track 1
        :param t2: track 2
        :param frame_index: current frame_index
        :param recorder: recorder
        :return: the similairty (float value). if valid, return None
        '''
        merge_value = []
        if t1 is t2:
            return None

        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f1 in enumerate(all_f1):
            for j, f2 in enumerate(all_f2):
                compare_f = [f1 + 1, f1 - 1]
                for f in compare_f:
                    if f not in all_f1 and f == f2:
                        n1 = t1.nodes[i]
                        n2 = t2.nodes[j]
                        s = TrackUtil.get_node_similarity(
                            n1, n2, frame_index, recorder,
                        )
                        if s is None:
                            continue
                        merge_value += [s]

        if len(merge_value) == 0:
            return None
        return np.mean(np.array(merge_value))

    @staticmethod
    def merge(t1, t2):
        '''
        merge t2 to t1, after that t2 is set invalid
        :param t1: track 1
        :param t2: track 2
        :return: None
        '''
        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f2 in enumerate(all_f2):
            if f2 not in all_f1:
                insert_pos = 0
                for j, f1 in enumerate(all_f1):
                    if f2 < f1:
                        break
                    insert_pos += 1
                t1.nodes.insert(insert_pos, t2.nodes[i])

        # remove some nodes in t1 in order to keep t1 satisfy the max nodes
        if len(t1.nodes) > TrackerConfig.max_track_node:
            t1.nodes = t1.nodes[-TrackerConfig.max_track_node:]
        t1.age = min(t1.age, t2.age)
        t2.valid = False


class TrackerConfig:
    max_record_frame = 30
    max_track_age = 30
    max_track_node = 30
    max_draw_track_node = 30

    max_object = config['max_object']
    network_model_path = config['resume']
    cuda = config['cuda']
    mean_pixel = config['mean_pixel']
    image_size = (config['image_size'], config['image_size'])

    min_iou_frame_gap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_iou = [0.3, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0]

    min_merge_threshold = 0.9

    max_bad_node = 0.9

    decay = 0.995

    roi_verify_max_iteration = 2
    roi_verify_punish_rate = 0.6


class FeatureRecorder:
    '''
    Record features and boxes every frame
    '''

    def __init__(self):
        self.max_record_frame = TrackerConfig.max_record_frame
        self.all_frame_index = np.array([], dtype=int)
        self.all_features = {}
        self.all_boxes = {}
        self.all_similarity = {}
        self.all_iou = {}
        self.other_time = []

    def update(self, network, frame_index, features, boxes):
        # if the coming frame in the new frame
        if frame_index not in self.all_frame_index:
            # if the recorder have reached the max_record_frame.
            if len(self.all_frame_index) == self.max_record_frame:
                del_frame = self.all_frame_index[0]
                del self.all_features[del_frame]
                del self.all_boxes[del_frame]
                del self.all_similarity[del_frame]
                del self.all_iou[del_frame]
                self.all_frame_index = self.all_frame_index[1:]

            # add new item for all_frame_index, all_features and all_boxes. Besides, also add new similarity
            self.all_frame_index = np.append(self.all_frame_index, frame_index)
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes

            self.all_similarity[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                delta = pow(TrackerConfig.decay, (frame_index - pre_index)/3.0)

                t2 = time.time()
                pre_similarity = network.forward_stacker_features(
                    (
                        self.all_features[pre_index],
                    ), (features), fill_up_column=False,
                )
                t3 = time.time()
                self.other_time.append(t3-t2)
                self.all_similarity[frame_index][pre_index] = pre_similarity*delta

            self.all_iou[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                self.all_iou[frame_index][pre_index] = iou
        else:
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes
            index = self.all_frame_index.__index__(frame_index)

            for pre_index in self.all_frame_index[:index+1]:

                if pre_index == self.all_frame_index[-1]:
                    continue
                t2 = time.time()
                pre_similarity = network.forward_stacker_features(
                    (self.all_features[pre_index]), (
                        self.all_features[-1],
                    ),
                )
                t3 = time.time()
                self.other_time.append(t3-t2)

                self.all_similarity[frame_index][pre_index] = pre_similarity

                iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                self.all_similarity[frame_index][pre_index] = iou

    def get_feature(self, frame_index, detection_index):
        '''
        get the feature by the specified frame index and detection index
        :param frame_index: start from 0
        :param detection_index: start from 0
        :return: the corresponding feature at frame index and detection index
        '''

        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
            if len(features) == 0:
                return None
            if detection_index < len(features):
                return features[detection_index]

        return None

    def get_box(self, frame_index, detection_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
            if len(boxes) == 0:
                return None

            if detection_index < len(boxes):
                return boxes[detection_index]
        return None

    def get_features(self, frame_index):
        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
        else:
            return None
        if len(features) == 0:
            return None
        return features

    def get_boxes(self, frame_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
        else:
            return None

        if len(boxes) == 0:
            return None
        return boxes


class Node:
    '''
    The Node is the basic element of a track. it contains the following information:
    1) extracted feature (it'll get removed when it isn't active
    2) box (a box (l, t, r, b)
    3) label (active label indicating keeping the features)
    4) detection, the formated box
    '''

    def __init__(self, frame_index, id, pose):
        self.frame_index = frame_index
        self.id = id
        self.pose = pose

    def get_box(self, frame_index, recoder):
        if frame_index - self.frame_index >= TrackerConfig.max_record_frame:
            return None
        return recoder.all_boxes[self.frame_index][self.id, :]

    def get_iou(self, frame_index, recoder, box_id):
        if frame_index - self.frame_index >= TrackerConfig.max_track_node:
            return None
        return recoder.all_iou[frame_index][self.frame_index][self.id, box_id]


class Track:
    '''
    Track is the class of track. it contains all the node and manages the node. it contains the following information:
    1) all the nodes
    2) track id. it is unique it identify each track
    3) track pool id. it is a number to give a new id to a new track
    4) age. age indicates how old is the track
    5) max_age. indicates the dead age of this track
    '''
    _id_pool = 0

    def __init__(self):
        self.nodes = list()
        self.id = Track._id_pool
        Track._id_pool += 1
        self.age = 0
        self.valid = True   # indicate this track is merged
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())

    def __del__(self):
        for n in self.nodes:
            del n

    def add_age(self):
        self.age += 1

    def reset_age(self):
        self.age = 0

    def add_node(self, frame_index, recorder, node):
        # iou judge
        if len(self.nodes) > 0:
            n = self.nodes[-1]
            iou = n.get_iou(frame_index, recorder, node.id)
            delta_frame = frame_index - n.frame_index
            if delta_frame in TrackerConfig.min_iou_frame_gap:
                iou_index = TrackerConfig.min_iou_frame_gap.index(delta_frame)
                # if iou < TrackerConfig.min_iou[iou_index]:
                if iou < TrackerConfig.min_iou[-1]:
                    return False
        self.nodes.append(node)
        self.reset_age()
        return True

    def get_similarity(self, frame_index, recorder):
        similarity = []
        for n in self.nodes:
            f = n.frame_index
            id = n.id
            if frame_index - f >= TrackerConfig.max_track_node:
                continue
            similarity += [recorder.all_similarity[frame_index][f][id, :]]

        if len(similarity) == 0:
            return None
        return np.sum(np.array(similarity), axis=0)

    def verify(self, frame_index, recorder, box_id):
        for n in self.nodes:
            delta_f = frame_index - n.frame_index
            if delta_f == 1:
                iou = n.get_iou(frame_index, recorder, box_id)
                if iou < -0.2:
                    return False
            if delta_f == 2:
                iou = n.get_iou(frame_index, recorder, box_id)
                if iou < -0.9:
                    return False
            if delta_f == 3:
                iou = n.get_iou(frame_index, recorder, box_id)
                if iou < -1.2:
                    return False

        return True


class Tracks:
    '''
    Track set. It contains all the tracks and manage the tracks. it has the following information
    1) tracks. the set of tracks
    2) keep the previous image and features
    '''

    def __init__(self):
        self.tracks = list()  # the set of tracks
        self.max_drawing_track = TrackerConfig.max_draw_track_node

    def __getitem__(self, item):
        return self.tracks[item]

    def append(self, track):
        self.tracks.append(track)
        self.volatile_tracks()

    def volatile_tracks(self):
        if len(self.tracks) > TrackerConfig.max_object:
            # start to delete the most oldest tracks
            all_ages = [t.age for t in self.tracks]
            oldest_track_index = np.argmax(all_ages)
            del self.tracks[oldest_track_index]

    def get_track_by_id(self, id):
        for t in self.tracks:
            if t.id == id:
                return t
        return None

    def get_similarity(self, frame_index, recorder):
        ids = []
        similarity = []
        for t in self.tracks:
            s = t.get_similarity(frame_index, recorder)
            if s is None:
                continue
            similarity += [s]
            ids += [t.id]

        similarity = np.array(similarity)

        track_num = similarity.shape[0]
        if track_num > 0:
            box_num = similarity.shape[1]
        else:
            box_num = 0

        if track_num == 0:
            return np.array(similarity), np.array(ids)

        similarity = np.repeat(similarity, [1]*(box_num-1)+[track_num], axis=1)
        return np.array(similarity), np.array(ids)

    def one_frame_pass(self):
        keep_track_set = list()
        for i, t in enumerate(self.tracks):
            t.add_age()
            if t.age > TrackerConfig.max_track_age:
                continue
            keep_track_set.append(i)

        self.tracks = [self.tracks[i] for i in keep_track_set]

    def merge(self, frame_index, recorder):
        t_l = len(self.tracks)
        res = np.zeros((t_l, t_l), dtype=float)
        # get track similarity matrix
        for i, t1 in enumerate(self.tracks):
            for j, t2 in enumerate(self.tracks):
                s = TrackUtil.get_merge_similarity(
                    t1, t2, frame_index, recorder,
                )
                if s is None:
                    res[i, j] = 0
                else:
                    res[i, j] = s

        # get the track pair which needs merged
        used_indexes = []
        merge_pair = []
        for i, t1 in enumerate(self.tracks):
            if i in used_indexes:
                continue
            max_track_index = np.argmax(res[i, :])
            if i != max_track_index and res[i, max_track_index] > TrackerConfig.min_merge_threshold:
                used_indexes += [max_track_index]
                merge_pair += [(i, max_track_index)]

        # start merge
        for i, j in merge_pair:
            TrackUtil.merge(self.tracks[i], self.tracks[j])

        # remove the invalid tracks
        self.tracks = [t for t in self.tracks if t.valid]

    def show(self, frame_index, recorder, image):
        h, w, _ = image.shape

        # draw rectangle
        for t in self.tracks:
            if len(t.nodes) > 0 and t.age < 2:
                b = t.nodes[-1].get_box(frame_index, recorder)
                if b is None:
                    continue
                txt = f'({t.id})'
                image = cv2.putText(
                    image, txt, (
                        int(
                            b[0]*w,
                        ), int((b[1])*h),
                    ), cv2.FONT_HERSHEY_SIMPLEX, 1, t.color, 3,
                )
                image = cv2.rectangle(
                    image, (
                        int(
                            b[0]*w,
                        ), int((b[1])*h),
                    ), (int((b[0]+b[2])*w), int((b[1]+b[3])*h)), t.color, 2,
                )

        # draw line
        for t in self.tracks:
            if t.age > 1:
                continue
            if len(t.nodes) > self.max_drawing_track:
                start = len(t.nodes) - self.max_drawing_track
            else:
                start = 0
            for n1, n2 in zip(t.nodes[start:], t.nodes[start+1:]):
                b1 = n1.get_box(frame_index, recorder)
                b2 = n2.get_box(frame_index, recorder)
                if b1 is None or b2 is None:
                    continue
                c1 = (int((b1[0] + b1[2]/2.0)*w), int((b1[1] + b1[3])*h))
                c2 = (int((b2[0] + b2[2] / 2.0) * w), int((b2[1] + b2[3]) * h))
                image = cv2.line(image, c1, c2, t.color, 2)

        return image


class Tracker:
    def __init__(self, pose_estimator):
        Track._id_pool = 0
        self.first_run = True
        self.image_size = TrackerConfig.image_size
        self.model_path = TrackerConfig.network_model_path
        self.cuda = TrackerConfig.cuda
        self.mean_pixel = TrackerConfig.mean_pixel
        self.max_object = TrackerConfig.max_object
        self.frame_index = 0
        self.load_model(pose_estimator)
        self.recorder = FeatureRecorder()
        self.tracks = Tracks()
        self.times = []
        self.other_time = []
        self.total = 0

    def load_model(self, pose_estimator):
        # load the model
        self.network = build_network(
            'test', pose_estimator=pose_estimator, max_objects=config['max_object'], batch_size=1,
        )
        if self.cuda:
            cudnn.benchmark = True
            self.network.load_state_dict(torch.load(config['resume']))
            self.network = self.network.cuda()
        else:
            self.network.load_state_dict(
                torch.load(
                    config['resume'], map_location='cpu',
                ),
            )
        self.network.eval()

    def update(self, use_nuscenes_map, image, detection, show_image, frame_index, poses=None, tokens=None, force_init=False, current_cs_record=None, current_poserecord=None,  first_cs_record=None, first_poserecord=None, camera_intrinsic=None):
        '''
        Update the state of tracker, the following jobs should be done:
        1) extract the features
        2) stack the features together
        3) get the similarity matrix
        4) do assignment work
        5) save the previous image
        :param image: the opencv readed image, format is hxwx3
        :param detections: detection array. numpy array (l, r, w, h) and they all formated in (0, 1)
        '''

        self.frame_index = frame_index
        # format the image and detection
        h, w, _ = image.shape
        image_org = np.copy(image)
        image = TrackUtil.convert_image(image)
        detection_org = np.copy(detection)
        detection = TrackUtil.convert_detection(detection)

        # features can be (1, 10, 450)
        t0 = time.time()
        if use_nuscenes_map:
            poses = torch.from_numpy(poses.astype(float)).float()
            poses.unsqueeze_(0)
            poses = (poses.cuda())
            features, poses_features = self.network.forward_feature_extracter(
                use_nuscenes_map, image, detection, poses, image_org, detection_org, current_cs_record, current_poserecord, first_cs_record, first_poserecord, camera_intrinsic,
            )
        else:
            features, poses_features = self.network.forward_feature_extracter(
                use_nuscenes_map, image, detection, poses=None, image_org=None, detection_org=detection_org,
            )

        t1 = time.time()
        self.times.append(t1-t0)
        self.total = self.total+1

        # update recorder
        self.recorder.update(
            self.network, self.frame_index,
            features.data, detection_org,
        )

        if self.frame_index == 0 or force_init or len(self.tracks.tracks) == 0:
            for i in range(detection.shape[1]):
                t = Track()
                if use_nuscenes_map:
                    pose = poses_features[0, i, :]
                else:
                    pose = None
                n = Node(self.frame_index, i, pose)
                t.add_node(self.frame_index, self.recorder, n)
                self.tracks.append(t)
            self.tracks.one_frame_pass()
            # self.frame_index += 1
            return self.tracks.show(self.frame_index, self.recorder, image_org)

        # get tracks similarity
        y, ids = self.tracks.get_similarity(self.frame_index, self.recorder)

        if len(y) > 0:
            # 3) find the corresponding by the similar matrix
            row_index, col_index = linear_sum_assignment(-y)
            col_index[col_index >= detection_org.shape[0]] = -1

            # verification by iou
            verify_iteration = 0
            while verify_iteration < TrackerConfig.roi_verify_max_iteration:
                is_change_y = False
                for i in row_index:
                    box_id = col_index[i]
                    track_id = ids[i]

                    if box_id < 0:
                        continue
                    t = self.tracks.get_track_by_id(track_id)
                    if not t.verify(self.frame_index, self.recorder, box_id):
                        y[i, box_id] *= TrackerConfig.roi_verify_punish_rate
                        is_change_y = True
                if is_change_y:
                    row_index, col_index = linear_sum_assignment(-y)
                    col_index[col_index >= detection_org.shape[0]] = -1
                else:
                    break
                verify_iteration += 1

            # 4) update the tracks
            for i in row_index:
                track_id = ids[i]
                t = self.tracks.get_track_by_id(track_id)
                col_id = col_index[i]
                if col_id < 0:
                    continue
                if use_nuscenes_map:
                    pose = poses_features[0, col_id, :]
                else:
                    pose = None

                node = Node(self.frame_index, col_id, pose)
                t.add_node(self.frame_index, self.recorder, node)

            # 5) add new track
            for col in range(len(detection_org)):
                if col not in col_index:
                    if use_nuscenes_map:
                        pose = poses_features[0, col, :]
                    else:
                        pose = None
                    node = Node(self.frame_index, col, pose)
                    t = Track()
                    t.add_node(self.frame_index, self.recorder, node)
                    self.tracks.append(t)

        # remove the old track
        self.tracks.one_frame_pass()

        image_org = self.tracks.show(
            self.frame_index, self.recorder, image_org,
        )
        return image_org
