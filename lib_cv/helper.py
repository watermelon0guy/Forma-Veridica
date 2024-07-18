import pathlib

import cv2
import glob

import numpy as np
import prettytable as pt


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def flatten(xii):
    return [x for xi in xii for x in xi]


def read_all_images(image_folder: pathlib.Path, extension='jpg'):
    images_path = image_folder.glob('*.' + extension)
    image_objects = []
    for f_name in images_path:
        image_objects.append(cv2.imread(str(f_name)))
    return image_objects


def print_pretty_table(matr, message, header=False, float_format="0.5"):
    print(message)
    table = pt.PrettyTable()
    table.add_rows(matr)
    table.header = header
    table.float_format = float_format
    print(table)
    return table.get_latex_string()


def crop_with_roi(img, roi):
    x, y, w, h = [int(i) for i in roi]
    img = img[y:y + h, x:x + w]
    return img


def set_camera_resolution(cam, x, y):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, x)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, y)


def remove_inf_vectors(arr):
    return arr[~np.any(np.isinf(arr), axis=1)]


def image_size(image):
    """
    Return image size in OpenCV's format.\n
    Возвращает размеры изображения в формате OpenCV
    :param image:
    :return:
    """
    h, w, _ = image.shape
    return (w, h)


def undistort_2_images(image_1, image_2,
                       camera_matrix_1, camera_matrix_1_optimal, distortion_coefficients_1,
                       camera_matrix_2, camera_matrix_2_optimal, distortion_coefficients_2):
    undistort_1 = cv2.undistort(image_1, camera_matrix_1, distortion_coefficients_1, None, camera_matrix_1_optimal)
    undistort_2 = cv2.undistort(image_2, camera_matrix_2, distortion_coefficients_2, None, camera_matrix_2_optimal)
    return undistort_1, undistort_2
