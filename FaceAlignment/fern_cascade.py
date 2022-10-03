import numpy as np
from utils import project_shape, similarity_transform, calculate_covariance
from bounding_box import BoundingBox
from fern import Fern


class FernCascade:

    def fit(self, images: list[np.ndarray],
            current_shapes: list[np.ndarray],
            ground_truth_shapes: list[np.ndarray],
            bounding_box: list[BoundingBox],
            mean_shape: np.ndarray,
            second_level_num: int,
            candidate_pixel_num: int,
            fern_pixel_num: int,
            curr_level_num: int,
            first_level_num: int):
        candidate_pixel_locations = np.zeros((candidate_pixel_num, 2))
        nearest_landmark_index = np.zeros((candidate_pixel_num, 1))

        regression_targets = []
        for i in range(len(current_shapes)):
            regression_target = project_shape(ground_truth_shapes[i], bounding_box[i]) \
                                - project_shape(current_shapes[i], bounding_box[i])

            rotation, scale = similarity_transform(mean_shape, project_shape(current_shapes[i], bounding_box[i]))
            regression_targets.append(scale * regression_target.dot(rotation.T))

        # get candidate pixel locations, please refer to 'shape-indexed features'
        for i in range(candidate_pixel_num):
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            while (x ** 2 + y ** 2) > 1:
                x = np.random.uniform(-1.0, 1.0)
                y = np.random.uniform(-1.0, 1.0)

            # find nearest landmark index
            min_dist = 1e10
            min_index = 0
            for j in range(mean_shape.shape[0]):
                temp = (mean_shape[j, 0] - x) ** 2 + (mean_shape[j, 1] - y) ** 2
                if temp < min_dist:
                    min_dist = temp
                    min_index = j

            candidate_pixel_locations[i, 0] = x - mean_shape[min_index, 0]
            candidate_pixel_locations[i, 1] = y - mean_shape[min_index, 1]
            nearest_landmark_index[i, 0] = min_index

        # get densities of candidate pixels for each image
        # for densities: each row is the pixel densities at each candidate pixels for an image
        densities = [[] for _ in range(candidate_pixel_num)]
        for i in range(len(images)):
            temp = project_shape(current_shapes[i], bounding_box[i])
            rotation, scale = similarity_transform(temp, mean_shape)
            for j in range(candidate_pixel_num):
                project_x = rotation[0, 0] * candidate_pixel_locations[j, 0] + \
                            rotation[0, 1] * candidate_pixel_locations[j, 1]

                project_y = rotation[1, 0] * candidate_pixel_locations[j, 0] + \
                            rotation[1, 1] * candidate_pixel_locations[j, 1]

                project_x *= scale * bounding_box[i].width / 2.0
                project_y *= scale * bounding_box[i].height / 2.0

                index = nearest_landmark_index[j]
                real_x = project_x + current_shapes[i][index, 0]
                real_y = project_y + current_shapes[i][index, 1]
                densities[j].append(int(images[i][real_y, real_x]))

        # calculate the covariance between densities at each candidate pixels
        covariance = np.zeros((candidate_pixel_num, candidate_pixel_num))
        for i in range(candidate_pixel_num):
            for j in range(candidate_pixel_num):
                correlation_result = calculate_covariance(densities[i], densities[j])
                covariance[i, j] = correlation_result
                covariance[j, i] = correlation_result

        # train ferns
        prediction = [np.zeros((mean_shape.shape[0], 2))
                      for _ in range(len(regression_targets))
                      ]
        ferns = [Fern() for _ in range(second_level_num)]
        for i in range(second_level_num):
            temp = ferns[i].fit(densities, covariance, candidate_pixel_locations,
                                nearest_landmark_index, regression_targets, fern_pixel_num)
            # update regression targets
            for j in range(len(temp)):
                prediction[j] += temp[j]
                regression_targets[j] -= temp[j]
            if (i + 1) % 50 == 0:
                print(f"Fern cascades: {curr_level_num} out of {first_level_num}")
                print(f"Ferns: {i + 1} out of {first_level_num}")


