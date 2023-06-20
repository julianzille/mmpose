# 2D and 3D Monocular Pose-Estimation on AcinoSet

This is the repository of my research into the use of monocular pose-estimation technologies in estimating and tracking the pose of a cheetah.
The repository is forked from [Open MMLab-MMPose](https://github.com/open-mmlab/mmpose/). 

The Jupyter Notebook used in developing object detection models is similarly forked from Open MMLab-MMDetection, and can be found [here](https://github.com/julianzille/mmdetection/blob/master/Animal%20Detect.ipynb).

The Jupyter Notebook file, [AnimalPose.ipynb](https://github.com/julianzille/mmpose/blob/master/AnimalPose.ipynb), contains all project-specific code. Function definitions (including MMPose function adaptations) can be found in [setup_env.py](https://github.com/julianzille/mmpose/blob/master/setup_env.py). 

Config files for 2D pose-estimation are located [here](https://github.com/julianzille/mmpose/tree/master/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/acino).

Config files for temporal pose-lifting are located [here](https://github.com/julianzille/mmpose/tree/master/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m).
