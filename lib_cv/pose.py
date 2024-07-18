import cv2 as cv
import numpy as np


def recover_pose_stereo_calib(object_points_1, image_points_1,
                              object_points_2, image_points_2,
                              camera_matrix_1, distortion_1,
                              camera_matrix_2, distortion_2,
                              image_size,
                              criteria_stereo=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                              flags=cv.CALIB_FIX_INTRINSIC):
    """
    Recover relative pose of second camera.\n
    Вычисляет положение второй камеры относительно второй.
    :param object_points_1:
    :param image_points_1:
    :param object_points_2:
    :param image_points_2:
    :param camera_matrix_1:
    :param distortion_1:
    :param camera_matrix_2:
    :param distortion_2:
    :param image_size:
    :param criteria_stereo:
    :param flags:
    :return:
    """
    ret_stereo, camera_matrix_1, distortion_1, camera_matrix_2, distortion_2, R, T, E, F = cv.stereoCalibrate(
        [object_points_1],
        [image_points_1],
        [image_points_2],
        camera_matrix_1,
        distortion_1,
        camera_matrix_2,
        distortion_2,
        image_size,
        criteria_stereo,
        flags)

    return R, T, E, F


def save_extrinsic_params(file_name, rotation, transformation, essential, fundamental):
    """
    Save camera pose.\n
    Сохраняет внешние параметры камер в numpy-архив (.npz).
    :param file_name:
    :param rotation:
    :param transformation:
    :param essential:
    :param fundamental:
    :return:
    """
    np.savez(f'{file_name}.npz',
             rotation=rotation, transformation=transformation,
             essential=essential, fundamental=fundamental)


def load_extrinsic_params(file_name) -> tuple[
    cv.Mat | np.ndarray, cv.Mat | np.ndarray, cv.Mat | np.ndarray, cv.Mat | np.ndarray]:
    """
    Загружает из numpy-архива (.npz) отн параметры камеры.
    :param file_name:
    :return:
    """

    loaded = np.load(file_name)
    return loaded['rotation'], loaded['transformation'], loaded['essential'], loaded['fundamental']
