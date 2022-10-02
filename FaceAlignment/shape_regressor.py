import cv2
import numpy as np
import pickle
from typing import List


class ShapeRegressor:

    def __init__(self):
        self.first_level_num = 0

    def fit(self, images: list[np.ndarray],
            ground_truth_shapes: list[np.ndarray],
            bounding_box,
            first_level_num,
            second_level_num,
            candidate_pixel_num,
            fern_pixel_num,
            initial_num):
        print("Start of training...")
        bounding_box_ = bounding_box
        training_shapes_= ground_truth_shapes
        first_level_num_ = first_level_num

