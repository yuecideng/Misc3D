# RGBD Dense Reconstruction
This tutorial provides the explanation of the workflow of Misc3D RGBD dense reconstruction. We use the code in `example` and dataset in `data` for demonstration.

## Build Reconstruction Module
1. Install OpenCV C++ library in your system.
2. Configure the `cmake` to find the path to OpenCV.
3. Set `BUILD_RECONSTRUCTION` ON.

## Workflow
### Prepare Dataset
As shown in `data` folder, you should create two folders, `color` and `depth`, storing the color and depth images seperately (you can change the name of `data` to other name). The naming of the images is not required to be the same as demonstrated in this tutorial but need to stored with identity order. **Please save your color image in `png` format**.

### Create Pipeline Configuration
Create a `json` file with the following elements:
```json
"data_path": "../data",
"camera": {
    "width": 640,
    "height": 480,
    "fx": 598.7568,
    "fy": 598.7568,
    "cx": 326.3443,
    "cy":  250.2448,
    "depth_scale": 1000.0
},
"make_fragments": {
    "descriptor_type": "orb",
    "feature_num": 100,
    "n_frame_per_fragment": 40,
    "keyframe_ratio": 0.5
},
"local_refine": "color",
"global_registration": "teaser",
"optimization_param": {
    "preference_loop_closure_odometry": 0.1,
    "preference_loop_closure_registration": 5.0
},
"max_depth": 3.0,
"max_depth_diff": 0.05,
"voxel_size": 0.01,
"integration_voxel_size": 0.008,
"enable_slac": true
```
These are the whole parameters of the reconstruction pipeline that can be tuned. If you do not specify part of these parameters, the default value will be used.

#### Parameters Description
- `make_fragments`:
    1. `descriptor_type`: The type of feature descriptor. It can be `orb` or `sift`.
    2. `feature_num`: The number of features extracted from each color image.
    3. `n_frame_per_fragment`: The number of frames used to make a fragment.
    4. `keyframe_ratio`: The ratio of keyframes used for loop closure computation.

- `local_refine`: Fragements are refined by ICP, which has the variant of `point2point`, `point2plane`, `color` and `generalized`.

- `global_registration`: The type of global registration method. It can be `teaser` or `ransac`.

- `optimization_param`:
    1. `preference_loop_closure_odometry`: The preference of loop closure odometry.
    2. `preference_loop_closure_registration`: The preference of loop closure registration.

- `max_depth`: The maximum depth value to limit the depth range.

- `max_depth_diff`: The maximum depth difference between two consecutive frames.

- `voxel_size`: The voxel size used to downsampling the fragments point cloud.

- `integration_voxel_size`: The voxel size used to create TSDF volume.

- `enable_slac`: Whether to enable SLAC for fragments pose graph optimization.

### Run Reconstruction
Run the pipeline by serveral lines of code:

```python
config_path = 'config.json'
pipeline = m3d.reconstruction.ReconstructionPipeline(config_path)
pipeline.run_system()
```

### Reconstructed Results
- `data/fragments`: The integrated fragments point cloud and its pose graph file.
- `data/scene`: The reconstructed scene triangle mesh and the trajectory json file which stores the whole odometry of the RGBD data.