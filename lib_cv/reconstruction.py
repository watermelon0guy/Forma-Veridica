import cv2 as cv
import numpy as np


def get_point_cloud(image_1, image_2):
    return None


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
