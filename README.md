# Misc3D
An unified library for 3D data processing and analysis with both C++&amp;Python API based on [Open3D](https://github.com/isl-org/Open3D).

This library aims at providing some useful 3d processing algorithms which Open3D is not yet provided or not easy to use, and sharing the same data structures used in Open3D.

Core modules:
- `common`: 
    1. Normals estimaiton from PointMap 
    2. Ransac for primitives fitting, including plane, sphere and cylinder, and support parallel computing.
    3. K nearest neighbors search based on [annoy](https://github.com/spotify/annoy). It has the similar API as `open3d.geometry.KDTreeFlann` class (the radius search is not supported).
- `preprocessing`: 
    1. Farthest point sampling
    2. Crop ROI of point clouds.
    3. Project point clouds into a plane. 
- `features`:
    1. Edge points detection from point clouds.
- `registration`:
    1. Corresponding matching with descriptors.
    2. 3D rigid transformation solver including SVD, RANSAC and [TEASERPP](https://github.com/MIT-SPARK/TEASER-plusplus).
- `pose_estimation`: 
    1. Point Pair Features (PPF) based 6D pose estimator.
- `segmentation`: 
    1. Proximity extraction in scalable implementation with different vriants, including distance, and normal angle.
- `vis`: Helper tools for drawing 6D pose, painted point cloud, triangle mesh and etc.

## How to build 
### Requirements
- `cmake` >= 3.10
- `python` >= 3.6
- `eigen` >= 3.3
- `open3d` == 0.14.1 
- `pybind11` >= 2.6.2

### Build
##### Linux (currently only supported)
1. Build `open3d` as external library. You can follow the instruction from here [guide](https://github.com/intel-isl/open3d-cmake-find-package). Build `pybind11` in your system as well.

2. Git clone the repo and run:
    ```
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=</path/to/installation>
    make install -j
    ```
    If you don't want to build python binding, just add `-DBUILD_PYTHON=OFF`.

3. After installation, add these two lines to `~/.bashrc` file:
    ```
    export PYTHONPATH="$PYTHONPATH:</path/to/installation>/misc3d/lib/python"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:</path/to/installation>/misc3d/lib"
    ```
    Run `sources ~/.bashrc` to save changes.
### How to use
The example python scripts can be found in `examples/python`. You can run it after you install the library successfully.

You can import `misc3d` same as `open3d`:
```
import open3d as o3d
import misc3d as m3d
```

#### Python API examples
```
# estimate normals
m3d.common.estimate_normals(pcd, (848, 480), 3)

# ransac for primitives fitting
w, index = m3d.common.fit_plane(pcd, 0.01, 100, enable_parallel=False)
w, index = m3d.common.fit_sphere(pcd, 0.01, 100, enable_parallel=False)
w, index = m3d.common.fit_cylinder(pcd, 0.01, 100, enable_parallel=False)

# farthest point sampling
indices = m3d.preprocessing.farthest_point_sampling(pcd, 1000)

# crop ROI point clouds
pcd_roi = m3d.preprocessing.crop_roi_pointcloud(pcd, (500, 300, 600, 400), (848, 480))

# project point clouds into a plane
pcd_plane = m3d.preprocessing.project_into_plane(pcd)

# edge points detection
index = m3d.features.detect_edge_points(
    pcd, o3d.geometry.KDTreeSearchParamHybrid(0.02, 30))
edges = pcd.select_by_index(index)

# feature matching using FLANN
# `fpfh_src` is open3d.pipeline.registration.Feature instance which is computed using FPFH 3d descriptor.
index1, index2 = m3d.registration.match_correspondence(fpfh_src, fpfh_dst, True)

# solve 3d rigid transformation
# ransac solver
pose = m3d.registration.compute_transformation_ransac(pc_src, pc_dst, (index1, index2), 0.03, 100000)
# svd solver
pose = m3d.registration.compute_transformation_svd(pc_src, pc_dst)
# teaser solver
pose = m3d.registration.compute_transformation_teaser(pc_src, pc_dst, 0.01)

# ppf pose estimator
# init config for ppf pose estimator
config = m3d.pose_estimation.PPFEstimatorConfig()
config.training_param.rel_sample_dist = 0.04
config.score_thresh = 0.1
config.refine_param.method = m3d.pose_estimation.PPFEstimatorConfig.PointToPlane
ppf = m3d.pose_estimation.PPFEstimator(config)
ret = ppf.train(model)
ret, results = ppf.match(scene)

# proximity extraction
pe = m3d.segmentation.ProximityExtractor(100)
ev = m3d.segmentation.DistanceProximityEvaluator(0.02)
index_list = pe.segment(pc, 0.02, ev)

# vis
# draw a pose represented as a axis
m3d.vis.draw_pose(vis, size=0.1)
# draw point clouds painted with red
m3d.vis.draw_point_cloud(vis, pcd, (1, 0, 0), size=3.0)
```

