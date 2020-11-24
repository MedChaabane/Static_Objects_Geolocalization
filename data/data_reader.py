import os

import cv2
import pandas as pd


class DataReader:
    def __init__(self, image_folder, detection_file_name):
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, '{}.jpg')
        self.detection = pd.read_csv(
            self.detection_file_name, header=None, sep=' ',
        )

        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

    def __len__(self):
        return len(self.detection_group_keys)

    def get_detection_by_index(self, index):
        if index > len(self.detection_group_keys) or self.detection_group_keys.count(index) == 0:
            return None

        return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        if index > len(self.detection_group_keys):
            return None

        return cv2.imread(self.image_format.format(index))

    def __getitem__(self, item):
        return (
            self.get_image_by_index(item+1),
            self.get_detection_by_index(item+1),
        )
