import os
import cv2
import nodar_challenge_lib


def get_disparity(left_images_path, right_images_path, result_path):
    for file in os.listdir(left_images_path):
        left_image = cv2.imread(left_images_path + "/" + file)
        right_image = cv2.imread(right_images_path + "/" + file)
        if(right_image is None):
            print("corresponding left/right image not found, skipping")
            continue
        stereo_disparity = nodar_challenge_lib.compute_disparity(
            left_image, right_image)
        # stereo_disparity = nodar_challenge_lib.colorize_disparity_map(
        #     stereo_disparity)
        cv2.imwrite(result_path + "/" + file, stereo_disparity)


def get_pointcloud(disparity_images):
    pass


def main():
    # path to folder containing left camera images
    left_images_path = "./data/left"
    # path to folder containing right camera images
    right_images_path = "./data/right"
    # path to save disparity images
    disparity_path = "./results/stereo_disparity"
    get_disparity(left_images_path, right_images_path, disparity_path)
    get_pointcloud(disparity_path)


if __name__ == "__main__":
    main()
