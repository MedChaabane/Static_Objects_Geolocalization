config = {
    'nuscenes_data_root': '/s/red/a/nobackup/vision/nuScenes/Tracking/final_data',
    'save_folder': '/s/red/a/nobackup/vision/nuScenes/Tracking/static_objects_geolocalization/weights',
    'log_folder': '/s/red/a/nobackup/vision/nuScenes/Tracking/static_objects_geolocalization/logs',
    'base_net_folder': '/s/red/a/nobackup/vision/nuScenes/Tracking/SST-master/weights/vgg16_reducedfc.pth',
    'resume': '/s/red/a/nobackup/vision/nuScenes/Tracking/static_objects_geolocalization/weights/model_200000.pth',
    'start_iter': 1,
    'cuda': True,
    'batch_size': 1,
    'num_workers': 8,
    'iterations': 200000,
    'learning_rate': 1e-4,
    'false_constant': 10.0,
    'type': 'train',  # choose from ('Test', 'Train')
    'dataset_type': 'train',  # choose from ('test', 'train')
    'max_object': 60,  # N
    'max_gap_frame': 30,  # not the hard gap
    'min_gap_frame': 0,  # not the hard gap
    'image_size': 900,
    'mean_pixel': (104, 117, 123),
    'max_expand': 1.2,
    'lower_contrast': 0.7,
    'upper_constrast': 1.5,
    'lower_saturation': 0.7,
    'upper_saturation': 1.5,
    'alpha_valid': 0.8,

    'base_net': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256,
        'C', 512, 512, 512, 'M', 512, 512, 512,
    ],
    'extra_net': [
        256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256,
        128, 'S', 256, 128, 256,
    ],
    'selector_size': (255, 113, 56, 28, 14, 12, 10, 5, 3),
    'selector_channel': (60, 80, 100, 80, 60, 50, 40, 30, 20),
    'final_net':
    [1052, 512, 256, 128, 64, 1],
    'vgg_source': [15, 25, -1],
    'default_mbox': [4, 6, 6, 6, 4, 4],
}
