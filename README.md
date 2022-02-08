# Misc3D
An unified library for 3D data processing and analysis with both C++&amp;Python API based on [Open3D](https://github.com/isl-org/Open3D).

This library aims at providing some useful 3d processing algorithms which Open3D is not yet provided or not easy to use, and sharing the same data structures used in Open3D.

Core modules:
- `common`: 
    1. Normals estimaiton from PointMap 
    2. Ransac for primitives fitting, including plane, sphere and cylinder. 
    3. ROI point clouds cropping from given bounding box of RGBD data (TODO)
- `preprocessing`: 
    1. Crop ROI of point clouds.
    2. Project point clouds into a plane. 
- `features`:
    1. Edge points detection from point clouds.
- `registration`:
    1. Corresponding matching with descriptors.
    2. 3D rigid transformation solver including SVD, RANSAC and [TEASERPP](https://github.com/MIT-SPARK/TEASER-plusplus).
- `pose_estimation`: 
    1. Point Pair Features (PPF) based 6D pose estimator.
- `segmentation`: 
    1. Proximity extraction in scalable implementation with different vriants, including distance, and normal angle.
- `vis`: Helper tools for drawing 6D pose, painted point cloud and etc.

## How to build 
### Requirements
- `cmake` >= 3.10
- `python` >= 3.6
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

###
