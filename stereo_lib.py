from typing import Tuple

import cv2
import numpy as np

# KITTI Parameters:
R_KITTI = np.array(
    [
        [0.99955687, 0.02232752, -0.0196868],
        [-0.02230226, 0.99975017, 0.00150172],
        [0.01971541, -0.001062, 0.9998051],
    ]
)
T_KITTI = np.array([0.53267121, -0.00526146, 0.00782809])
distCoeffs1_KITTI = np.array(
    [-0.3691481, 0.1968681, 0.00135347, 0.00056776, -0.06770705]
)
distCoeffs2_KITTI = np.array(
    [-0.3639558, 0.1788651, 0.00060297, -0.00039224, -0.0538246]
)
K1_KITTI = np.array(
    [[959.791, 0.0, 696.0217], [0.0, 956.9251, 224.1806], [0.0, 0.0, 1.0]]
)
K2_KITTI = np.array(
    [[903.7596, 0.0, 695.7519], [0.0, 901.9653, 224.2509], [0.0, 0.0, 1.0]]
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

    alg = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=192,
        blockSize=3,
        P1=9,
        P2=108,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_HH,
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

    _, _, _, _, Q, _, _ = cv2.stereoRectify(
        K1, distCoeffs1, K2, distCoeffs2, size, R, T
    )

    # Get 3D location of each pixel
    # pointcloud.shape = (height, width, 3), where 3 is for (x,y,z)
    # "16*" because disparity_s16 is in 11.4 format
    pointcloud = -16 * cv2.reprojectImageTo3D(disparity_int16, Q)

    return pointcloud
