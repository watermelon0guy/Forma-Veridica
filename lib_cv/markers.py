import cv2 as cv
import numpy as np

from lib_cv import correspondence, helper


def get_charuco(charuco_board: cv.aruco.CharucoBoard, image: cv.Mat | np.ndarray) -> tuple[cv.Mat | np.ndarray, cv.Mat | np.ndarray]:
    """
    Находит на изображении ChArUco доску и возвращает её координаты на изображении
    :param charuco_board:
    :param image:
    :return: кортеж из двух элементов:
            - 3D координаты в системе ChArUco доски
            - соответствующие точки на изображении
    """

    charuco_detector = cv.aruco.CharucoDetector(charuco_board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(image)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(image,
                                                                                            charuco_corners,
                                                                                            charuco_ids,
                                                                                            marker_corners,
                                                                                            marker_ids)
    object_points, image_points = charuco_board.matchImagePoints(charuco_corners, charuco_ids)

    return object_points, image_points


def match_points_sift_flann(image_1, image_2, F, ratio=0.65):
    kp_1, kp_2, des_1, des_2 = correspondence.sift(image_1, image_2)
    good_matches = correspondence.flann_knn_match(des_1, des_2, ratio=ratio)

    good_matches = helper.flatten(good_matches)

    points_1 = np.float32([kp_1[match.queryIdx].pt for match in good_matches])
    points_2 = np.float32([kp_2[match.trainIdx].pt for match in good_matches])

    error_sum = 0
    inlier_matches = []
    for i in range(len(points_1)):
        pt_1 = np.append(points_1[i], 1)
        pt_2 = np.append(points_2[i], 1)
        error = np.abs(np.dot(pt_2.T, np.dot(F, pt_1)))
        error_sum += error
        if error < 0.5:  # Порог ошибки можно настроить
            inlier_matches.append(good_matches[i])

    good_matches = inlier_matches

    points_1 = np.float32([kp_1[match.queryIdx].pt for match in good_matches])
    points_2 = np.float32([kp_2[match.trainIdx].pt for match in good_matches])
    return points_1, points_2


def triangulate_points(camera_matrix_1, camera_matrix_2, points_1, points_2, R, T):
    projection_matrix1 = camera_matrix_1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_matrix2 = camera_matrix_2 @ np.hstack((R, T))
    points_4d_homogeneous = cv.triangulatePoints(projection_matrix1, projection_matrix2, points_1.T, points_2.T).T
    points_3d = cv.convertPointsFromHomogeneous(points_4d_homogeneous)
    points_3d_formatted = points_3d.reshape(-1, 3)

    return points_3d_formatted


def write_ply(filename, points, colors):
    with open(filename, 'w') as f:
        num_points = len(points)
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(num_points))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write("{} {} {} {} {} {}\n".format(point[0], point[1], point[2], color[2], color[1], color[0]))
