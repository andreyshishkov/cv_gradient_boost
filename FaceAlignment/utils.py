import cv2
import numpy as np


def calculate_covariance(v1: np.ndarray,
                         v2: np.ndarray):
    covariance = np.cov(v1, v2)[0][1]
    return covariance


def project_shape(shape, bounding_box):
    temp = np.zeros(
        (len(shape), 2)
    )
    for j in range(len(shape)):
        temp[j, 0] = (shape[j, 0]-bounding_box.centroid_x) / (bounding_box.width / 2.0)
        temp[j, 1] = (shape[j, 1] - bounding_box.centroid_y) / (bounding_box.height / 2.0)
    return temp


def re_project_shape(shape: np.ndarray,
                     bounding_box):
    temp = np.zeros((shape.shape[0], 2))
    for i in range(shape.shape[0]):
        temp[i, 0] = shape[i, 0] * bounding_box.width / 2 + bounding_box.centroid_x
        temp[i, 1] = shape[i, 1] * bounding_box.height / 2 + bounding_box.centroid_y
    return temp


def similarity_transform(shape_1,
                         shape_2):
    rotation = np.zeros((2, 2))
    center_x_1 = center_y_1 = center_x_2 = center_y_2 = 0
    for i in range(len(shape_1)):
        center_x_1 += shape_1[i, 0]
        center_y_1 += shape_1[i, 1]
        center_x_2 += shape_2[i, 0]
        center_y_2 += shape_2[i, 1]
    center_x_1 /= len(shape_1)
    center_y_1 /= len(shape_1)
    center_x_2 /= len(shape_2)
    center_y_2 /= len(shape_2)

    temp_1 = np.copy(shape_1)
    temp_2 = np.copy(shape_2)
    for i in range(len(shape_1.shape[0])):
        temp_1[i, 0] -= center_x_1
        temp_1[i, 1] -= center_y_1
        temp_2[i, 0] -= center_x_2
        temp_2[i, 1] -= center_y_2
    covariance_1 = np.cov(temp_1, rowvar=False)
    covariance_2 = np.cov(temp_2, rowvar=False)

    mean_1 = np.mean(temp_1, axis=0)
    mean_2 = np.mean(temp_2, axis=0)

    s1 = np.linalg.norm(covariance_1)
    s2 = np.linalg.norm(covariance_2)
    scale = s1 / s2
    temp_1 *= 1.0 / s1
    temp_2 *= 1.0 / s2

    num = den = 0
    for i in range(shape_1.shape[0]):
        num += temp_1[i, 1] * temp_2[i, 0] - temp_1[i, 0] * temp_2[i, 1]
        den = den + temp_1[i, 0] * temp_2[i, 0] + temp_1[i, 1] * temp_2[i, 1]

    norm = np.sqrt(num ** 2 + den ** 2)
    sin_theta = num / norm
    cos_theta = den / norm
    rotation[0, 0] = cos_theta
    rotation[0, 1] = - sin_theta
    rotation[1, 0] = sin_theta
    rotation[1, 1] = cos_theta

    return rotation, scale


def get_mean_shape(shapes: list[np.ndarray],
                   bounding_box):
    result = np.zeros(
        (shapes[0].shape[0], 2)
    )
    for i in range(len(shapes)):
        result += project_shape(shapes[i], bounding_box[i])
    result *= 1.0 / len(shapes)
    return result
