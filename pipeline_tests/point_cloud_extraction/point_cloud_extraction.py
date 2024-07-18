import cv2 as cv
from lib_cv import pose, calibration, helper, correspondence, reconstruction
from pathlib import Path

camera_mat_1, dist_coeff_1 = calibration.load_single_cam_params(
    Path('/home/watermelon0guy/Видео/new_exp/phone_1/intrinsic.npz'))
camera_mat_2, dist_coeff_2 = calibration.load_single_cam_params(
    Path('/home/watermelon0guy/Видео/new_exp/phone_2/intrinsic.npz'))

image_1 = cv.imread("/home/watermelon0guy/Видео/new_exp/phone_1/synced_pics/frame_6736.png")
image_2 = cv.imread("/home/watermelon0guy/Видео/new_exp/phone_2/synced_pics/frame_6737.png")

image_size = helper.image_size(image_1)

R, T, E, F = pose.load_extrinsic_params(Path("/home/watermelon0guy/Видео/new_exp/extrinsic.npz"))

image_points_1, image_points_2 = correspondence.match_points_sift_flann(image_1, image_2, F)

points_3d = reconstruction.triangulate_points(camera_mat_1, camera_mat_2, image_points_1, image_points_2, R, T)

colors = correspondence.get_colors_from_points(image_1, image_points_1)

reconstruction.write_ply(Path('/home/watermelon0guy/Code/Python/cv_reconstruct/res/membrane_exp/bulge.ply'), points_3d, colors)
