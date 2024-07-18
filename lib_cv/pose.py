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
        object_points_1,
        image_points_1,
        image_points_2,
        camera_matrix_1,
        distortion_1,
        camera_matrix_2,
        distortion_2,
        image_size,
        criteria_stereo,
        flags)

    return R, T, E, F


def compute_fundamental_matrix(points_1, points_2, ransac_reproj_threshold=3.0, ransac_confidence=0.99):
    # Вычисление фундаментальной матрицы с помощью метода RANSAC
    fundamental_matrix, mask = cv.findFundamentalMat(points_1, points_2, cv.FM_RANSAC, ransac_reproj_threshold,
                                                     ransac_confidence)

    return fundamental_matrix, mask


# def recover_pose_solvePnP(object_points_1, image_points_1,
#                           object_points_2, image_points_2,
#                           camera_matrix_1, distortion_1,
#                           camera_matrix_2, distortion_2,
#                           image_size):
#     # Использование solvePnP для первой камеры
#     success1, r_vec_1, t_vec_1 = cv.solvePnP(object_points_1, image_points_1, camera_matrix_1, distortion_1)
#     rotation_matrix_1, _ = cv.Rodrigues(r_vec_1)
#
#     # Использование solvePnP для второй камеры
#     success2, r_vec_2, t_vec_2 = cv.solvePnP(object_points_1, image_points_2, camera_matrix_2, distortion_2)
#     rotation_matrix_2, _ = cv.Rodrigues(r_vec_2)
#
#     R = rotation_matrix_2 @ rotation_matrix_1.T
#     T = t_vec_2 - R @ t_vec_1
#     F, mask_F = compute_fundamental_matrix(charuco_corners_1, charuco_corners_2)


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
    Загружает из numpy-архива (.npz) внешние параметры камеры.
    :param file_name:
    :return:
    """

    loaded = np.load(file_name)
    return loaded['rotation'], loaded['transformation'], loaded['essential'], loaded['fundamental']
