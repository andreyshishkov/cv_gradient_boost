import cv2
import numpy as np
from typing import List
from utils import calculate_covariance


class Fern:

    def __init__(self, fern_pixel_num_: int,
                 landmark_num_: int,
                 selected_nearest_landmark_index_: np.ndarray = None,
                 threshold_: np.ndarray = None,
                 selected_pixel_index_: np.ndarray = None,
                 selected_pixel_locations_: np.ndarray = None,
                 bin_output_: np.ndarray = None
                 ):
        self.__landmark_num_ = landmark_num_
        self.__selected_nearest_landmark_index_ = selected_nearest_landmark_index_
        self.__threshold_ = threshold_
        self.__selected_pixel_index_ = selected_pixel_index_
        self.__selected_pixel_locations_ = selected_pixel_locations_
        self.__bin_output_ = bin_output_
        self.fern_pixel_num = 0

    def fit(self, candidate_pixel_intensity: np.ndarray,
            covariance: np.ndarray,
            candidate_pixel_locations: np.ndarray,
            nearest_landmark_index: List[int],
            regression_targets: np.ndarray,
            fern_pixel_num: int
            ):
        self.fern_pixel_num = fern_pixel_num
        landmark_num_ = regression_targets[0].shape[0]
        self.__selected_pixel_index_ = np.zeros((fern_pixel_num, 2))
        self.__selected_pixel_locations_ = np.zeros((fern_pixel_num, 4))
        self.__selected_nearest_landmark_index_ = np.zeros((fern_pixel_num, 2))
        candidate_pixel_num = candidate_pixel_locations.shape[0]

        self.__threshold_ = np.zeros((fern_pixel_num, 1))

        for i in range(fern_pixel_num):
            random_direction = np.random.rand((landmark_num_, 2))

            projection_result = np.zeros(regression_targets.shape[0])
            for j, sample in enumerate(regression_targets):
                projection_result[j] = np.sum(sample.dot(random_direction))

            covariance_projection_density = np.zeros((candidate_pixel_num, 1))
            for j in range(candidate_pixel_num):
                covariance_projection_density[j][0] = \
                    calculate_covariance(projection_result, candidate_pixel_intensity[j])

            # find max correlations
            max_correlation = -1
            max_pixel_index_1, max_pixel_index_2 = 0, 0
            for j in range(candidate_pixel_num):
                for k in range(candidate_pixel_num):

                    temp1 = covariance[j, j] + covariance[k, k] - 2 * covariance[j, k]
                    if abs(temp1) < 1e-10:
                        continue

                    flag = False
                    for p in range(i):
                        if j == self.__selected_pixel_index_[p, 0] and k == self.__selected_pixel_index_[p, 1]:
                            flag = True
                            break
                        elif j == self.__selected_pixel_index_[p, 1] and k == self.__selected_pixel_index_[p, 0]:
                            flag = True
                            break

                    if flag:
                        continue

                    temp = (covariance_projection_density[j][0] - covariance_projection_density[k][0]) / np.sqrt(temp1)
                    if abs(temp) > max_correlation:
                        max_correlation = temp
                        max_pixel_index_1 = j
                        max_pixel_index_2 = k

            self.__selected_pixel_index_[i, 0] = max_pixel_index_1
            self.__selected_pixel_index_[i, 1] = max_pixel_index_2
            self.__selected_pixel_locations_[i, 0] = candidate_pixel_locations[max_pixel_index_1, 0]
            self.__selected_pixel_locations_[i, 1] = candidate_pixel_locations[max_pixel_index_1, 1]
            self.__selected_pixel_locations_[i, 2] = candidate_pixel_locations[max_pixel_index_2, 0]
            self.__selected_pixel_locations_[i, 3] = candidate_pixel_locations[max_pixel_index_2, 1]
            self.__selected_nearest_landmark_index_[i, 0] = nearest_landmark_index[max_pixel_index_1]
            self.__selected_nearest_landmark_index_[i, 1] = nearest_landmark_index[max_pixel_index_2]

            # get threshold for pair
            max_diff = -1
            for j in range(len(candidate_pixel_intensity[max_pixel_index_1])):
                temp = candidate_pixel_intensity[max_pixel_index_1][j] - candidate_pixel_intensity[max_pixel_index_2][j]
                if abs(temp) > max_diff:
                    max_diff = abs(temp)

            self.__threshold_[i] = np.random.uniform(-0.2 * max_diff, 0.2 * max_diff)

        # determine the bins of each shape
        bin_num = 2 ** fern_pixel_num
        shape_in_bin = [[] for _ in range(bin_num)]
        for i in range(len(regression_targets)):
            index = 0
            for j in range(fern_pixel_num):
                density_1 = candidate_pixel_intensity[self.__selected_pixel_index_[j, 0], i]
                density_2 = candidate_pixel_intensity[self.__selected_pixel_index_[j, 1], i]
                if density_1 - density_2 >= self.__threshold_[j]:
                    index = index + 2 ** j

            shape_in_bin[index].append(i)

        # get bin output
        self.__bin_output_ = [[] for _ in range(bin_num)]
        prediction = [[] for _ in range(len(regression_targets))]
        for i in range(bin_num):
            temp = np.zeros((landmark_num_, 2))
            bin_size = len(shape_in_bin[i])
            for j in range(bin_size):
                index = shape_in_bin[i][j]
                temp = temp + regression_targets[index]
            if bin_size == 0:
                self.__bin_output_[i] = temp
                continue
            temp = (1.0 / ((1.0 + 1000.0 / bin_size) * bin_size)) * temp
            self.__bin_output_[i] = temp
            for j in range(bin_size):
                index = shape_in_bin[i][j]
                prediction[index] = temp

        return prediction

    def predict(self, image,
                shape,
                rotation,
                bounding_box,
                scale):
        index = 0
        for i in range(self.fern_pixel_num):
            nearest_landmark_index_1 = self.__selected_nearest_landmark_index_[i, 0]
            nearest_landmark_index_2 = self.__selected_nearest_landmark_index_[i, 1]

            # get intensity 1
            x = self.__selected_pixel_locations_[i, 0]
            y = self.__selected_pixel_locations_[i, 1]
            project_x = scale * (rotation[0, 0] * x + rotation[0, 1] * y) * \
                        bounding_box.width / 2.0 + shape[nearest_landmark_index_1, 0]
            project_y = scale * (rotation[1, 0] * x + rotation[1, 1] * y) * bounding_box.height / 2.0 + \
                        shape[nearest_landmark_index_1, 1]

            project_x = int(
                max(0.0, min(project_x, image.shape[1] - 1.0))
            )
            project_y = int(
                max(0.0, min(project_y, image.shape[0] - 1.0))
            )
            intensity_1 = image[project_y, project_x]

            # get intensity 2
            x = self.__selected_pixel_locations_[i, 2]
            y = self.__selected_pixel_locations_[i, 3]
            project_x = scale * (rotation[0, 0] * x + rotation[0, 1] * y) * \
                        bounding_box.width / 2.0 + shape[nearest_landmark_index_2, 0]
            project_y = scale * (rotation[1, 0] * x + rotation[1, 1] * y) * bounding_box.height / 2.0 + \
                        shape[nearest_landmark_index_2, 1]
            project_x = int(
                max(0.0, min(project_x, image.shape[1] - 1.0))
            )
            project_y = int(
                max(0.0, min(project_y, image.shape[0] - 1.0))
            )
            intensity_2 = image[project_y, project_x]

            if intensity_1 - intensity_2 >= self.__threshold_[i]:
                index += 2 ** i

        return self.__bin_output_[index]


    def read(self):
        pass

    def save(self):
        pass
