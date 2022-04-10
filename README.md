# Stereo Image Processing and Ground segmentation of Lidar Pointcloud

This repository contains opencv stereo processing functions to compute disparity and generate pointcloud from stereo images using calibration params

It also contains ground plane fitting and segmentation of pointcloud using open3d library

### Usage
```
git clone git@github.com:SameerWatve/stereo_opencv.git
pip install -r requirements.txt
```

### Stereo processing

Stereo mathcing uses **Semi Global Block Matching** algorithm in OpenCV.
Update results and input paths from `disparity_pointcloud.py` and run `python3 disparity_pointcloud.py`

### Lidar Ground Plane Segmentation

RANSAC based plane fitting method in open3d to segment pointcloud ground points
Update results and input paths from `ground_plane.py` and run `python3 ground_plane.py`