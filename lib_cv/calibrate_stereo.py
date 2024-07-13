import cv2 as cv

import calibration
import helper
import numpy as np


def calibrate_different_stereo_cameras(cam_1_images, cam_2_images, board: cv.aruco.CharucoBoard):
    default_image_size_1 = cam_1_images[0].shape[1::-1]
    default_image_size_2 = cam_2_images[0].shape[1::-1]
    image_size = default_image_size_1

    ret_1, K_1, D_1, _, _, obj_points_1, img_points_1 = calibration.calibrate_with_charuco(cam_1_images, board)
    ret_2, K_2, D_2, _, _, obj_points_2, img_points_2 = calibration.calibrate_with_charuco(cam_2_images, board)

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret_stereo, K_1, D_1, K_2, D_2, R, T, E, F = cv.stereoCalibrate(obj_points_1, img_points_1, img_points_2, K_1, D_1,
                                                                    K_2, D_2, image_size, criteria_stereo, flags)
    print("Расстояние между камерами", np.linalg.norm(T))
    print("Качество стереокалибровки: ", ret_stereo)
    print("Матрица вращения: \n", R)
    print("Вектор трансляции: \n", T)

    rectify_scale = 1
    rect_L, rect_R, proj_matrix_L, proj_matrix_R, Q, roi_1, roi_2 = cv.stereoRectify(K_1, D_1,
                                                                                     K_2, D_2,
                                                                                     image_size, R, T,
                                                                                     rectify_scale, (0, 0))

    stereo_map_L = cv.initUndistortRectifyMap(K_1, D_1, rect_L, proj_matrix_L, image_size,
                                              cv.CV_16SC2)
    stereo_map_R = cv.initUndistortRectifyMap(K_2, D_2, rect_R, proj_matrix_R, image_size,
                                              cv.CV_16SC2)

    print("Saving parameters!")
    cv2_file = cv.FileStorage('stereo_map.xml', cv.FILE_STORAGE_WRITE)

    cv2_file.write('stereo_map_L_x', stereo_map_L[0])
    cv2_file.write('stereo_map_L_y', stereo_map_L[1])
    cv2_file.write('stereo_map_R_x', stereo_map_R[0])
    cv2_file.write('stereo_map_R_y', stereo_map_R[1])
    cv2_file.write('roi_1', roi_1)
    cv2_file.write('roi_2', roi_2)
    cv2_file.write('q', Q)

    cv2_file.release()