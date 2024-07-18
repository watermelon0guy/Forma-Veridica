import cv2 as cv
import numpy as np


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


