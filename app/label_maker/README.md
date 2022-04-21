# Label Maker for 6D Pose Estimation
This is a simple offline label maker for **instance-level** 6D pose estimation ground truth data generation, forming by several python scripts. The whole pipeline is as follows:
- Collect RGBD data from sensor.
- Run RGBD dense reconstruction.
- Initialize the pose for each instance in integrated scene.
- Rendering mask and generate labels.

## Requirements
- `opencv-python`
- `open3d`
- `misc3d` (with `reconstruction` module enabled)

## Pipeline
### Step 1: Collect RGBD data
We provide a simple script (`label_maker/record_data.py`) to collect RGBD data from realsense RGBD camera. There are some arguments you can set in the script to control record mode and frame rate.

You can also collect the data by yourself. But it is recommended to use the following structure:
```
dataset/
    - color/
        - 000000.png
        - 000001.png
        - ...
    - depth/
        - 000000.png
        - 000001.png
        - ...
```

**Note:**
You should collect no more than 300 pairs of RGBD data for each dataset, otherwise the resonstruction steo in next step will be very slow. You can record your data by multiple times with different scene (different model)

### Step 2: Run RGBD dense reconstruction
Run `python3 reconstruction.py config.json` to get the reconstructed scene.