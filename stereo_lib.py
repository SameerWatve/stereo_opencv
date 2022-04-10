from typing import Tuple

import cv2
import numpy as np

# KITTI Parameters:

R_L = np.transpose(np.matrix([[9.999758e-01, -5.267463e-03, -4.552439e-03],
                              [5.251945e-03, 9.999804e-01, -3.413835e-03],
                              [4.570332e-03, 3.389843e-03, 9.999838e-01]]))
R_R = np.matrix([[9.995599e-01, 1.699522e-02, -2.431313e-02],
                 [-1.704422e-02, 9.998531e-01, -1.809756e-03],
                 [2.427880e-02, 2.223358e-03, 9.997028e-01]])
R_KITTI = R_L * R_R
T_L = np.transpose(np.matrix([5.956621e-02, 2.900141e-04, 2.577209e-03]))
T_R = np.transpose(np.matrix([-4.731050e-01, 5.551470e-03, -5.250882e-03]))
T_KITTI = T_L - T_R
distCoeffs1_KITTI = np.array(
    [-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02]
)

distCoeffs2_KITTI = np.array(
    [-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]
)

K1_KITTI = np.array(
    [[9.597910e+02, 0.000000e+00, 6.960217e+02], [0.000000e+00, 9.569251e+02,
                                                  2.241806e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]]
)

K2_KITTI = np.array(
    [[9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02,
                                                  2.242509e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]]
)


def rectify(
    left: np.ndarray,
    right: np.ndarray,
    distCoeffs1: np.ndarray = distCoeffs1_KITTI,
    K1: np.ndarray = K1_KITTI,
    distCoeffs2: np.ndarray = distCoeffs2_KITTI,
    K2: np.ndarray = K2_KITTI,
    R: np.ndarray = R_KITTI,
    T: np.ndarray = T_KITTI,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stereo pair rectification based on stereo camera parameters.

    :params
    left - unrectified image from the left camera
    right - unrectified image from the right camera
    distCoeffs1 - left camera distortion coefficients
    K1 - left camera matrix
    distCoeffs2 - right camera distortion coefficients
    K2 - right camera matrix
    R - rotation matrix between the cameras
    T - translation vector between the cameras

    :return
    left_rectified - left rectified image
    right_rectified - right rectified image
    """

    size = (left.shape[1], left.shape[0])

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K1, distCoeffs1, K2, distCoeffs2, size, R, T,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, distCoeffs1, R1, P1, size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, distCoeffs2, R2, P2, size, cv2.CV_32FC1
    )

    left_rectified = cv2.remap(
        left, map1x, map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0
    )
    right_rectified = cv2.remap(
        right, map2x, map2y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0
    )

    return left_rectified, right_rectified


def compute_disparity(
    left_rectified: np.ndarray, right_rectified: np.ndarray
) -> np.ndarray:
    """
    Function computing disparity map based on the rectified images

    params:
    left_rectified - left rectified image
    right_rectified - right rectified image

    return:
    disparity_int16 - disparity map in 12.4 format, i.e., a value of 16
        actually corresponds to a disparity of 1.
    """

    if np.ndim(left_rectified) == 3:
        left_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    if np.ndim(right_rectified) == 3:
        right_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    window_size = 9
    minDisparity = 1
    alg = cv2.StereoSGBM_create(
        blockSize=10,
        numDisparities=64,
        preFilterCap=10,
        minDisparity=minDisparity,
        P1=4 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_HH
    )

    disparity_int16 = alg.compute(left_rectified, right_rectified)

    return disparity_int16


def colorize_disparity_map(disparity_int16: np.ndarray) -> np.ndarray:
    """
    Convert disparity map (int16) into a BGR image for visualization.
    Input disparity map, `disparity_int16`, is in 12.4 format.

    params:
    disparity_int16 - disparity map in 12.4 format

    return:
    disparity_color - colorized disparity map, which can be visualized
    """

    disparity_scaled = (disparity_int16 / 16.0).astype(np.uint8)

    # Apply non-linear scaling to disparity map to emphasize targets at long range
    C = np.sqrt(255)
    disparity_scaled = C * np.sqrt(disparity_scaled)
    disparity_scaled = disparity_scaled.astype(np.uint8)
    disparity_color = cv2.applyColorMap(disparity_scaled, cv2.COLORMAP_JET)

    # Return color image
    return disparity_color


def disparity_to_pointcloud(
    disparity_int16: np.ndarray,
    distCoeffs1: np.ndarray = distCoeffs1_KITTI,
    K1: np.ndarray = K1_KITTI,
    distCoeffs2: np.ndarray = distCoeffs2_KITTI,
    K2: np.ndarray = K2_KITTI,
    R: np.ndarray = R_KITTI,
    T: np.ndarray = T_KITTI,
) -> np.ndarray:
    """
    Convert disparity map to a point cloud
    The output pointcloud is the 3D location of each pixel
    pointcloud.shape = (height, width, 3), where 3 is for (x,y,z)

    params:
    disparity_int16 - disparity map in 12.4 format
    distCoeffs1 - left camera distortion coefficients
    K1 - left camera matrix
    distCoeffs2 - right camera distortion coefficients
    K2 - right camera matrix
    R - rotation matrix between the cameras
    T - translation vector between the cameras

    return:
    pointcloud - point cloud in the same shape as the input disparity map
    """

    # Get 4x4 perspective transformation from stereoRectify()
    size = (disparity_int16.shape[1],
            disparity_int16.shape[0])  # e.g., (1280,720)
    IMG_SIZE = (1392, 512)
    _, _, _, _, Q, _, _ = cv2.stereoRectify(
        K1, distCoeffs1, K2, distCoeffs2, IMG_SIZE, R_L, T_L, newImageSize=size)
    # Get 3D location of each pixel
    # pointcloud.shape = (height, width, 3), where 3 is for (x,y,z)
    # "16*" because disparity_s16 is in 11.4 format
    pointcloud = -16 * cv2.reprojectImageTo3D(disparity_int16, Q)

    return pointcloud
