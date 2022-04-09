import os
import cv2
import numpy as np
import stereo_lib as lib


def get_disparity(
        left_images_path: str,
        right_images_path: str,
        result_path: str,
        rectify: bool = False,
        colorize: bool = True) -> None:
    """
    function to generate left right stereo disparity 

    Args:
        left_images_path (str): path to left images
        right_images_path (str): path to right images
        result_path (str): path to save output
        rectify (bool, optional): flag to do rectification of input images
        colorize (bool, optional): flag to do colorization of output
    """
    for file in os.listdir(left_images_path):
        left_image = cv2.imread(left_images_path + "/" + file)
        right_image = cv2.imread(right_images_path + "/" + file)
        if(left_image is None or right_image is None):
            print("corresponding left/right image not found, skipping")
            continue
        if(rectify):
            left_image, right_image = lib.rectify(left_image, right_image)
        stereo_disparity = lib.compute_disparity(left_image, right_image)
        if(colorize):
            stereo_disparity = lib.colorize_disparity_map(stereo_disparity)
        cv2.imwrite(result_path + "/" + file, stereo_disparity)
    


def get_pointcloud(
        disparity_images: str,
        result_path: str):
    """function to generate 3D pointcloud from stereo disparity

    Args:
        disparity_images (str): path to stereo disparity images
        result_path (str): path to save output
    """
    for file in os.listdir(disparity_images):
        disparity_image = cv2.imread(disparity_images + "/" + file)
        if np.ndim(disparity_image) == 3:
            disparity_image_gray = cv2.cvtColor(
                disparity_image, cv2.COLOR_BGR2GRAY)
        else:
            disparity_image_gray = disparity_image
        pointcloud = lib.disparity_to_pointcloud(disparity_image_gray)
        # Get rid of points with value 0
        mask = disparity_image_gray > disparity_image_gray.min()
        pointcloud_corrected = pointcloud[mask]
        disparity_image_corrected = disparity_image[mask]
        file_name = file.split('.')[0]+".ply"
        save_pointcloud(pointcloud_corrected, disparity_image_corrected,
                        result_path + "/" + file_name)


def save_pointcloud(
        pointcloud: np.ndarray,
        colors: np.ndarray,
        filename: str):
    """
    function to write numpy pointcloud as .ply format 

    Args:
        pointcloud (np.ndarray): pointcloud as numpy array
        colors (np.ndarray): disparity color image
        filename (str): name of ouput file
    """
    colors = colors.reshape(-1, 3)
    pointcloud = np.hstack([pointcloud.reshape(-1, 3), colors])

    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(pointcloud)))
        np.savetxt(f, pointcloud, '%f %f %f %d %d %d')


def main():
    # path to folder containing left camera images
    left_images_path = "./data/left"
    # path to folder containing right camera images
    right_images_path = "./data/right"
    # path to save disparity images
    disparity_path = "./results/stereo_disparity"
    get_disparity(left_images_path, right_images_path, disparity_path)
    # path to save pointcloud
    pointcloud_path = "./results/stereo_pointcloud"
    get_pointcloud(disparity_path, pointcloud_path)


if __name__ == "__main__":
    main()
