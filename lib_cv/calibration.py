import types

import numpy as np
import cv2 as cv


def calibrate_camera_chessboard(images, pattern_size=(9, 6), square_size=1,
                                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    """
    Не рекомендовано к использованию. Используйте calibrate_with_charuco
    :param images:
    :param pattern_size:
    :param square_size:
    :param criteria:
    :return:
    """
    # Список для хранения 3D точек шахматной доски
    obj_points = []

    # Список для хранения 2D точек углов шахматной доски на изображениях
    img_points = []

    # Генерация координат 3D точек шахматной доски
    obj_p = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    if len(images) == 0:
        raise RuntimeError("Calibration procedure need image array to work")

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Найти углы шахматной доски на изображении
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

        # Если углы найдены, добавить их в списки obj_points и img_points
        if ret:
            obj_points.append(obj_p)
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
        # print("Обработано", len(img_points), "из", len(images))

    # Калибровка камеры
    ret, K, D, r_vecs, t_vecs = cv.calibrateCamera(obj_points, img_points, gray.shape[1::-1], None, None)

    K, roi = cv.getOptimalNewCameraMatrix(K, D, gray.shape[1::-1], 1, gray.shape[1::-1])
    # roi = 0
    mean_error = 0
    for i in range(len(obj_points)):
        img_points_, _ = cv.projectPoints(obj_points[i], r_vecs[i], t_vecs[i], K, D)
        error = cv.norm(img_points[i], img_points_, cv.NORM_L2) / len(img_points_)
        mean_error += error

    print("total error: {}".format(mean_error / len(obj_points)))

    return ret, K, D, r_vecs, t_vecs, roi, obj_points, img_points


def calibrate_with_charuco(images, charuco_board: cv.aruco.CharucoBoard, alpha=0):
    """
    Вычисление внутренних параметров камеры. Входные данные - фотографии ChArUco доски.
    Благодаря особенностям ChArUco, ситуации, где паттерн виден частично, также подходят и будут корректно обработаны
    :param images:
    :param charuco_board:
    :param alpha: Параметр в диапазоне от 0 (избавления от черных участков без информации)
            до 1 (все пиксели исходного изображения сохранены, но есть черные участки).
    :return:
    """

    all_charuco_corners = []
    all_charuco_ids = []
    all_object_points = []
    all_image_points = []

    charuco_detector = cv.aruco.CharucoDetector(charuco_board)

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        curr_charuco_corners, curr_charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        if not isinstance(curr_charuco_corners, types.NoneType):
            curr_obj_points, curr_img_points = charuco_board.matchImagePoints(curr_charuco_corners, curr_charuco_ids)
            if curr_img_points.size == 0 or curr_obj_points.size == 0:
                continue
            all_charuco_corners.append(curr_charuco_corners)
            all_charuco_ids.append(curr_charuco_ids)
            all_image_points.append(curr_img_points)
            all_object_points.append(curr_obj_points)

    h, w = gray.shape

    ret, camera_matrix, distortion_coefficients, r_vecs, t_vecs = cv.calibrateCamera(all_object_points,
                                                                                     all_image_points,
                                                                                     (w, h),
                                                                                     None,
                                                                                     None)

    camera_matrix_new, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), alpha, (w, h))

    return ret, camera_matrix, camera_matrix_new, distortion_coefficients, r_vecs, t_vecs, all_object_points, all_image_points


def save_single_cam_params(file_name, camera_matrix, camera_matrix_optimal, distortion_coefficients):
    """
    Сохранить результат калибровки (внутренние параметры камеры) в файл в формате numpy-архив (.npz)
    :param file_name:
    :param camera_matrix:
    :param camera_matrix_optimal:
    :param distortion_coefficients:
    :return:
    """

    np.savez(f'{file_name}.npz', camera_matrix=camera_matrix, camera_matrix_optimal=camera_matrix_optimal, distortion_coefficients=distortion_coefficients)


def load_single_cam_params(file_name) -> tuple[cv.Mat | np.ndarray, cv.Mat | np.ndarray, cv.Mat | np.ndarray]:
    """
    Загружает из numpy-архива (.npz) внутренние параметры камеры.
    :param file_name:
    :return:
    """

    loaded = np.load(file_name)
    return loaded['camera_matrix'], loaded['camera_matrix_optimal'], loaded['distortion_coefficients']
