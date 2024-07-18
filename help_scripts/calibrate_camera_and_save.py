import cv2 as cv
from lib_cv import calibration, helper
import pathlib

calibration_folder = pathlib.Path("/home/watermelon0guy/Видео/new_exp/phone_2/calib")
params_file = pathlib.Path("/home/watermelon0guy/Видео/new_exp/phone_2/intrinsic")

calibration_images = helper.read_all_images(calibration_folder, extension="png")

h, w, ch = calibration_images[0].shape

charuco_board = cv.aruco.CharucoBoard((5, 5), 10.0, 7.0, cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50))

ret, camera_matrix, distortion_coefficients, _, _, _, _ = calibration.calibrate_with_charuco(calibration_images, charuco_board)

calibration.save_single_cam_params(params_file, camera_matrix, distortion_coefficients)
