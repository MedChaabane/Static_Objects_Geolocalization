import argparse
import os.path
import time
import warnings
from os import path

import cv2
import numpy as np
import torch.nn.parallel
from nuscenes.nuscenes import NuScenes

from config.config import config
from data.data_reader import DataReader
from layer.pose_network import PoseNet
from tracker import Tracker
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description=' Tracker Test')
parser.add_argument(
    '--nuscenes_data_root',
    default=config['nuscenes_data_root'], help='Location of nuscenes TL pose and tracking dataset',
)
parser.add_argument('--type', default=config['type'], help='train/test')
parser.add_argument(
    '--show_image', default=True,
    help='show image if true, or hidden',
)
parser.add_argument(
    '--log_folder', default=config['log_folder'], help='video saving or result saving folder',
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
# djnelqdfpe


def test(estimator=None):

    if args.Joint:
        nusc = NuScenes(
            version='v1.0-trainval', verbose=True,
            dataroot=args.nuscenes_root,
        )

    dataset_image_folder_format = os.path.join(
        args.nuscenes_data_root, args.type+'/'+'{}/img1',
    )
    detection_file_name_format = os.path.join(
        args.nuscenes_data_root, args.type+'/'+'{}/gt/gt.txt',
    )

    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    save_folder = args.log_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    saved_file_name_format = os.path.join(save_folder, 'Video'+'{}.txt')
    save_video_name_format = os.path.join(save_folder, 'Video'+'{}.avi')
    test_dataset_index = os.listdir(
        os.path.join(args.nuscenes_data_root, args.type),
    )

    def f(format_str): return [
        format_str.format(index)
        for index in test_dataset_index
    ]
    for image_folder, detection_file_name, saved_file_name, save_video_name in zip(f(dataset_image_folder_format), f(detection_file_name_format), f(saved_file_name_format), f(save_video_name_format)):
        if path.exists(image_folder) and os.path.getsize(detection_file_name) > 0:

            tracker = Tracker(estimator)
            reader = DataReader(
                image_folder=image_folder,
                detection_file_name=detection_file_name,
            )
            result = list()
            first_run = True
            for i, item in enumerate(reader):

                if i > len(reader):
                    break

                if item is None:
                    print('item is none')
                    continue

                img = item[0]
                det = item[1]
                if img is None or det is None or len(det) == 0:
                    continue
                if len(det) > config['max_object']:
                    det = det[:config['max_object'], :]

                h, w, _ = img.shape

                if first_run:
                    vw = cv2.VideoWriter(
                        save_video_name, cv2.VideoWriter_fourcc(
                            'M', 'J', 'P', 'G',
                        ), 10, (w, h),
                    )
                    first_run = False

                features = det[:, 6:12].astype(float)
                tokens = det[:, 12:]

                if args.Joint:
                    tokens = tokens[0][1:]
                    frame_token = tokens[0]
                    first_frame_token = tokens[1]

                    current_cam_record = nusc.get('sample_data', frame_token)
                    current_cam_path = nusc.get_sample_data_path(frame_token)
                    current_cam_path, boxes, current_camera_intrinsic = nusc.get_sample_data(
                        current_cam_record['token'],
                    )
                    current_cs_record = nusc.get(
                        'calibrated_sensor', current_cam_record['calibrated_sensor_token'],
                    )
                    current_poserecord = nusc.get(
                        'ego_pose', current_cam_record['ego_pose_token'],
                    )

                    first_cam_record = nusc.get(
                        'sample_data', first_frame_token,
                    )
                    first_cam_path = nusc.get_sample_data_path(
                        first_frame_token,
                    )
                    first_cam_path, boxes, first_camera_intrinsic = nusc.get_sample_data(
                        first_cam_record['token'],
                    )
                    first_cs_record = nusc.get(
                        'calibrated_sensor', first_cam_record['calibrated_sensor_token'],
                    )
                    first_poserecord = nusc.get(
                        'ego_pose', first_cam_record['ego_pose_token'],
                    )

                det[:, [2, 4]] /= float(w)
                det[:, [3, 5]] /= float(h)
                if args.Joint:
                    image_org = tracker.update(
                        args.Joint, img, det[
                            :,
                            2:6
                        ], args.show_image, i, features, tokens, False,
                        current_cs_record, current_poserecord, first_cs_record, first_poserecord, first_camera_intrinsic,
                    )
                else:
                    image_org = tracker.update(
                        args.Joint, img, det[:, 2:6], args.show_image, i,
                    )

                vw.write(image_org)

                # save result
                for t in tracker.tracks:
                    n = t.nodes[-1]
                    if t.age == 1:
                        b = n.get_box(tracker.frame_index-1, tracker.recorder)
#                         print('n.pose ',n.pose.cpu().data.numpy())
                        if args.Joint:
                            result.append(
                                [i] + [t.id] + [
                                    b[0]*w, b[1]*h, b[2]*w, b[3]
                                    * h,
                                ] + list(n.pose.cpu().data.numpy()),
                            )
                        else:
                            result.append(
                                [i] + [t.id] + [b[0]*w, b[1]*h, b[2]*w, b[3]*h],
                            )
            # save data
            np.savetxt(saved_file_name, np.array(result), fmt='%10f')


if __name__ == '__main__':
    estimator = PoseNet()
    test(estimator)
