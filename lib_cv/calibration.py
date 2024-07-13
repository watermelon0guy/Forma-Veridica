import types

import numpy as np
import cv2 as cv


def calibrate_camera_chessboard(images, pattern_size=(9, 6), square_size=1,
                                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
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


def calibrate_with_charuco(images, charuco_board: cv.aruco.CharucoBoard):
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

    ret, K, D, r_vecs, t_vecs = cv.calibrateCamera(all_object_points, all_image_points, (w, h), None, None)

    return ret, K, D, r_vecs, t_vecs, all_object_points, all_image_points
