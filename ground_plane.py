
import numpy as np
import os
import struct
import open3d as o3d


def load_pointcloud(filename: str) -> np.ndarray:
    """
    Function to load pointcloud data as np array
    supported formats are .bin or .ply

    Args:
        filename (str): path to pointcloud file
    Return:
        pointcloud data as numpy array
    """
    if(filename.split('.')[-1] == "bin"):
        print(1)
        # binary lidar pointcloud files (as KITTI velodyne data)
        list_pcd = []
        with open(filename, "rb") as f:
            byte = f.read(4*4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(4*4)
        np_pcd = np.asarray(list_pcd)
    elif(filename.split('.')[-1] == "ply"):

        ply_pcd = o3d.io.read_point_cloud(filename)
        np_pcd = np.asarray(ply_pcd.points)
    else:
        raise ValueError(
            "can't read this pointcloud, only .bin and .ply formats supported")

    return np_pcd


def segment_ground_plane(
        pointcloud_path: str,
        visualize: bool = False):
    """
    Function to segment ground plane points in given pointcloud
    It uses Ransac for fitting a model ax + by + cz + d = 0
    Function prints equation of plane on console, option to visualize results using point3d

    Args:
        pointcloud_path (str): path to folder container pointcloud data
        visualize (bool, optional): Visualize segmented pointcloud in 3D view
    """
    for file in os.listdir(pointcloud_path):
        if(file.endswith('.bin') or file.endswith('.ply')):
            np_pcd = load_pointcloud(pointcloud_path + "/" + file)
        else:
            print("pointcloud format not supported, skipping")
            continue
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pcd.points = v3d(np_pcd)
        best_eq, inliers = o3d.geometry.PointCloud.segment_plane(pcd, distance_threshold=0.2,
                                                                 ransac_n=10,
                                                                 num_iterations=10000)
        print("equation of ground plane: \n", best_eq)
        if(visualize):
            o3d.visualization.draw_geometries(
                [pcd], window_name="Original PCL", point_show_normal=True)
            not_ground = o3d.geometry.PointCloud.select_by_index(
                pcd, inliers, invert=True)
            ground = o3d.geometry.PointCloud.select_by_index(pcd, inliers)
            not_ground = o3d.geometry.PointCloud.paint_uniform_color(not_ground, [
                0.8, 0.8, 0.8])
            ground = o3d.geometry.PointCloud.paint_uniform_color(ground, [
                1.0, 0, 0])
            o3d.visualization.draw_geometries(
                [not_ground, ground], window_name="Segmented PCL", point_show_normal=True)


def main():
    # path to folder containing stereo pointcloud
    stereo_pcl_path = "./results/stereo_pointcloud"
    # path to folder containing KITTI velodyne pointcloud
    velodyne_pcl_path = "./data/velodyne"
    segment_ground_plane(stereo_pcl_path, visualize=True)
    segment_ground_plane(velodyne_pcl_path, visualize=True)


if __name__ == "__main__":
    main()
