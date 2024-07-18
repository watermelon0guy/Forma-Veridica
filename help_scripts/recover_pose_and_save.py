import cv2 as cv
import numpy as np

from lib_cv import pose, calibration, helper, markers
from pathlib import Path

camera_mat_1, camera_mat_1_optimal, dist_coeff_1 = calibration.load_single_cam_params(
    Path('/home/watermelon0guy/Видео/new_exp/phone_1/intrinsic.npz'))
camera_mat_2, camera_mat_2_optimal, dist_coeff_2 = calibration.load_single_cam_params(
    Path('/home/watermelon0guy/Видео/new_exp/phone_2/intrinsic.npz'))

image_1 = cv.imread("/home/watermelon0guy/Видео/new_exp/phone_1/synced_pics/frame_265.png")
image_2 = cv.imread("/home/watermelon0guy/Видео/new_exp/phone_2/synced_pics/frame_265.png")

image_size = helper.image_size(image_1)

image_1, image_2 = helper.undistort_2_images(image_1, image_2,
                                             camera_mat_1, camera_mat_1_optimal, dist_coeff_1,
                                             camera_mat_2, camera_mat_2_optimal, dist_coeff_2)

camera_mat_1 = camera_mat_1_optimal
camera_mat_2 = camera_mat_2_optimal

charuco_board = cv.aruco.CharucoBoard((5, 5), 10.0, 7.0, cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50))

object_points_1, image_points_1 = markers.get_charuco(charuco_board, image_1)
object_points_2, image_points_2 = markers.get_charuco(charuco_board, image_2)

R, T, E, F = pose.recover_pose_stereo_calib([object_points_1], [image_points_1], [object_points_2], [image_points_2],
                                            camera_mat_1, dist_coeff_1, camera_mat_2, dist_coeff_2, image_size)

pose.save_extrinsic_params(Path("/home/watermelon0guy/Видео/new_exp/extrinsic"), R, T, E, F)
