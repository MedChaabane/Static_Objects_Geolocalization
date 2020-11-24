import argparse
import os

import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from pyquaternion import Quaternion
from torch.autograd import Variable
from YOLO.PyTorchYOLOv3.models import *
from YOLO.PyTorchYOLOv3.utils.datasets import *
from YOLO.PyTorchYOLOv3.utils.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_def', type=str,
        default='/s/red/a/nobackup/vision/nuScenes/YOLO/PyTorchYOLOv3/config/yolov3.cfg', help='path to model definition file',
    )
    parser.add_argument(
        '--weights_path', type=str,
        default='/s/red/a/nobackup/vision/nuScenes/YOLO/PyTorchYOLOv3/weights/yolov3.weights', help='path to weights file',
    )
    parser.add_argument(
        '--class_path', type=str,
        default='/s/red/a/nobackup/vision/nuScenes/YOLO/PyTorchYOLOv3/data/coco.names', help='path to class label file',
    )
    parser.add_argument(
        '--conf_thres', type=float,
        default=0.05, help='object confidence threshold',
    )
    parser.add_argument(
        '--nms_thres', type=float, default=0.4,
        help='iou thresshold for non-maximum suppression',
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=1, help='size of the batches',
    )
    parser.add_argument(
        '--n_cpu', type=int, default=0,
        help='number of cpu threads to use during batch generation',
    )
    parser.add_argument(
        '--img_size', type=int, default=416,
        help='size of each image dimension',
    )
    parser.add_argument(
        '--data_directory', type=str, default='/s/red/a/nobackup/vision/nuScenes/Tracking/data8',
        help='directory to save the extracted data',
    )
    parser.add_argument(
        '--nuscenes_root', type=str,
        default='/s/red/a/nobackup/vision/nuScenes/data/sets/nuscenes', help='path to nuscenes directory',
    )

    opt = parser.parse_args()
    data_directory = opt.data_directory
    os.mkdir(data_directory + '/train/')
    os.mkdir(data_directory + '/test/')
    device = torch.device('cpu')
    nusc = NuScenes(
        version='v1.0-trainval', verbose=True,
        dataroot=opt.nuscenes_root,
    )

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith('.weights'):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    file1 = open('videos_tokens.txt')
    tokens_list = file1.readlines()
    file1.close()

    valid_channels = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
    ]
    lll = []
    splits = create_splits_scenes()
    for camera_channel in valid_channels:
        for token in tokens_list:
            first_time = True
            Dict = {}
            current_recs = {}
            num_objects = 0
            scene_token = token[:-1]
            scene_record = nusc.get('scene', scene_token)
            scene_rec = scene_record
            scene = nusc.get('scene', scene_token)
            scene_name = scene_record['name']

            scene_id = int(scene_name.replace('scene-', ''))
            if scene_name in splits['val']:
                spl = '/test/'
            else:
                spl = '/train/'
            os.mkdir(data_directory + spl+str(scene_id)+camera_channel)
            os.mkdir(data_directory + spl+str(scene_id)+camera_channel+'/img1')
            os.mkdir(data_directory + spl+str(scene_id)+camera_channel+'/gt')
            file_w = open(
                data_directory+spl+str(scene_id) +
                camera_channel+'/gt/gt.txt', 'w',
            )

            count_frames = 0
            sample_token = scene['first_sample_token']
            found = False
            sample_record = nusc.get('sample', sample_token)
            freq = 10
            # Time-stamps are measured in micro-seconds.
            time_step = 1 / freq * 1e6
            first_sample_rec = nusc.get(
                'sample', scene_rec['first_sample_token'],
            )
            last_sample_rec = nusc.get(
                'sample', scene_rec['last_sample_token'],
            )
            channel = camera_channel
            current_recs[channel] = nusc.get(
                'sample_data', first_sample_rec['data'][channel],
            )
            current_time = first_sample_rec['timestamp']
            while current_time < last_sample_rec['timestamp']:

                current_time += time_step
                for channel, sd_rec in current_recs.items():
                    while sd_rec['timestamp'] < current_time and sd_rec['next'] != '':
                        sd_rec = nusc.get('sample_data', sd_rec['next'])
                        current_recs[channel] = sd_rec

                count_frames = count_frames+1
                cam_token = sd_rec['token']
                cam_record = nusc.get('sample_data', cam_token)
                cam_path = nusc.get_sample_data_path(cam_token)
                cam_path, boxes, camera_intrinsic = nusc.get_sample_data(
                    sd_rec['token'],
                )
                im = Image.open(cam_path)
                im.save(
                    data_directory + spl + str(scene_id) +
                    camera_channel + '/img1/' +
                    str(count_frames)+'.jpg', 'JPEG',
                )
                TL_found = False
                bb = []
                patch_radius = 700
                sample_record = nusc.get('sample', sample_token)
                log_record = nusc.get('log', scene_record['log_token'])
                log_location = log_record['location']
                nusc_map = NuScenesMap(
                    dataroot=opt.nuscenes_root, map_name=log_location,
                )
                img = transforms.ToTensor()(Image.open(cam_path))
                # Pad to square resolution
                img, _ = pad_to_square(img, 0)
                # Resize
                img = resize(img, 416)
                input_imgs = Variable(img.type(Tensor)).unsqueeze(0)
                cs_record = nusc.get(
                    'calibrated_sensor',
                    cam_record['calibrated_sensor_token'],
                )
                poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
                ego_pose = poserecord['translation']
                if first_time == True:
                    first_rec = cs_record.copy()
                    first_cam_token = first_sample_rec['data'][camera_channel]
                    first_cam_record = nusc.get(
                        'sample_data', first_cam_token,
                    ).copy()
                    first_poserecord = nusc.get(
                        'ego_pose', first_cam_record['ego_pose_token'],
                    ).copy()
                    first_cam_intrinsic = np.array(
                        first_rec['camera_intrinsic'],
                    ).copy()
                first_time = False
                # Get detections
                with torch.no_grad():
                    detections = model(input_imgs)
                    detections = non_max_suppression(
                        detections, opt.conf_thres, opt.nms_thres,
                    )[0]
                    if detections is not None:
                        orig_h, orig_w = im.size
                        ss = orig_w, orig_h
                        detections = rescale_boxes(detections, 416, ss)

                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                            box_w = x2 - x1
                            box_h = y2 - y1

                            if classes[int(cls_pred)] == 'traffic light':
                                TL_found = True
                                box = (
                                    int(x1-10), int(y1-10),
                                    int(x2+10), int(y2+10),
                                )
                                crop = im.crop(box).convert('RGB')

                                bb.append([x1, y1, x2, y2])

                if TL_found:
                    im_size = im.size
                    cs_record = nusc.get(
                        'calibrated_sensor', cam_record['calibrated_sensor_token'],
                    )
                    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

                    layer_names = ['traffic_light']
                    cam_record = nusc.get('sample_data', cam_token)
                    cam_path = nusc.get_sample_data_path(cam_token)
                    poserecord = nusc.get(
                        'ego_pose', cam_record['ego_pose_token'],
                    )
                    ego_pose = poserecord['translation']
                    if scene_id == 53:

                        ego_pose[2] = -1.2
                    elif scene_id == 57:
                        ego_pose[2] = -2.4
                    elif scene_id == 59:
                        ego_pose[2] = -1.8
                    elif scene_id == 60:
                        ego_pose[2] = -1.8
                    elif scene_id == 93:
                        ego_pose[2] = -1.8
                    elif scene_id >= 94 and scene_id <= 96:
                        ego_pose[2] = -0.6
                    elif scene_id >= 97 and scene_id <= 98:
                        ego_pose[2] = -1.0
                    elif scene_id >= 104 and scene_id <= 200:
                        ego_pose[2] = -0.3
                    elif scene_id >= 200 and scene_id <= 250:
                        ego_pose[2] = -0.9
                    elif scene_id >= 250 and scene_id <= 300:
                        ego_pose[2] = -0.5
                    else:
                        ego_pose[2] = -0.3

                    box_coords = (
                        ego_pose[0] - patch_radius,
                        ego_pose[1] - patch_radius,
                        ego_pose[0] + patch_radius,
                        ego_pose[1] + patch_radius,
                    )
                    records_in_patch = nusc_map.get_records_in_patch(
                        box_coords, layer_names, 'intersect',
                    )

                    near_plane = 1e-8
    #                 # Retrieve and render each record.
                    taken_already = []
                    index = -1
                    for layer_name in layer_names:
                        for token in records_in_patch[layer_name]:
                            record = nusc_map.get(layer_name, token)

                            line = nusc_map.extract_line(record['line_token'])
                            if line.is_empty:  # Skip lines without nodes
                                continue
                            xs, ys = line.xy
                            points = np.array(
                                [
                                    [record['pose']['tx']], [
                                        record['pose']
                                        ['ty'],
                                    ], [record['pose']['tz']],
                                ],
                            )
                            # Transform into the ego vehicle frame for the timestamp of the image.
                            points = points - \
                                np.array(poserecord['translation']).reshape(
                                    (-1, 1),
                                )
                            points = np.dot(
                                Quaternion(
                                    poserecord['rotation'],
                                ).rotation_matrix.T, points,
                            )
                            # Transform into the camera.
                            points = points - \
                                np.array(cs_record['translation']).reshape(
                                    (-1, 1),
                                )
                            points = np.dot(
                                Quaternion(
                                    cs_record['rotation'],
                                ).rotation_matrix.T, points,
                            )
                            # Remove points that are partially behind the camera.
                            depths = points[2, :]
                            Tx, Ty, Tz = points[0, :][0], points[
                                1,
                                :
                            ][0], points[2, :][0]
                            behind = depths < near_plane
                            if np.all(behind):
                                continue

                            # Grab the depths before performing the projection (z axis points away from the camera).
                            how_far = points[2, :][0]
                            # Take the actual picture (matrix multiplication with camera-matrix + renormalization).

                            points = view_points(
                                points[:3, :], cam_intrinsic, normalize=True,
                            )

                            depths = points[2, :]
                            # Skip polygons where all points are outside the image.
                            # Leave a margin of 1 pixel for aesthetic reasons.
                            inside = np.ones(depths.shape[0], dtype=bool)
                            inside = np.logical_and(inside, points[0, :] > -1)
                            inside = np.logical_and(
                                inside, points[0, :] < im.size[0] + 1,
                            )
                            inside = np.logical_and(inside, points[1, :] > -1)
                            inside = np.logical_and(
                                inside, points[1, :] < im.size[1] + 1,
                            )

                            if np.any(np.logical_not(inside)):
                                continue
                            Cx, Cy, d = points[
                                0,
                                :
                            ][0], points[1, :][0], how_far

                            if d < 70 and d > 0:
                                dis = 10000
                                best_bb = []
                                for ind, [x1, y1, x2, y2] in enumerate(bb):
                                    xx = (0.5*(x1+x2), 0.5*(y1+y2))
                                    yy = (Cx, Cy)
                                    distance = math.sqrt(
                                        sum([(a - b) ** 2 for a, b in zip(xx, yy)]),
                                    )

                                    if distance < dis and ind not in taken_already:
                                        best_bb = [[x1, y1, x2, y2]]
                                        dis = distance
                                        index = ind
                                taken_already.append(index)
                                for [x1, y1, x2, y2] in best_bb:
                                    if Cx < x2+70 and Cx > x1-70 and Cy < y2+70 and Cy > y1-70:
                                        points = np.array(
                                            [[xs[1]], [ys[1]], [0]],
                                        )
                                        # Transform into the ego vehicle frame for the timestamp of the image.
                                        points = points - \
                                            np.array(poserecord['translation']).reshape(
                                                (-1, 1),
                                            )
                                        points = np.dot(
                                            Quaternion(
                                                poserecord['rotation'],
                                            ).rotation_matrix.T, points,
                                        )

                                        # Transform into the camera.
                                        points = points - \
                                            np.array(cs_record['translation']).reshape(
                                                (-1, 1),
                                            )
                                        points = np.dot(
                                            Quaternion(
                                                cs_record['rotation'],
                                            ).rotation_matrix.T, points,
                                        )

                                        points = points - \
                                            np.array([[Tx], [Ty], [Tz]]).reshape(
                                                (-1, 1),
                                            )
                                        rotation = points.reshape(1, -1)
                                        rotation[0][1] = 0
                                        rotation = rotation / \
                                            np.linalg.norm(rotation)
                                        if token not in Dict:
                                            num_objects = num_objects+1
                                            Dict[token] = num_objects
                                        file_w.write(
                                            str(count_frames)+' '+str(Dict[token])+' '+str(x1.item())+' '+str(y1.item())+' '+str(x2.item()-x1.item())+' '+str(y2.item()-y1.item())+' '+str(
                                                Cx,
                                            )+' '+str(Cy)+' '+str(d)+' '+str(rotation[0][0])+' '+str(rotation[0][1])+' '+str(rotation[0][2])+' '+token+' '+cam_token+' '+first_cam_token+'\n',
                                        )
