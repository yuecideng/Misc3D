#include <open3d/geometry/PointCloud.h>
#include <iostream>

class foo : public open3d::geometry::PointCloud {
public:
    foo() {}
    ~foo() {}
    void receive(const open3d::geometry::PointCloud &points) {
        std::cout << "success" << std::endl;
    }

    open3d::geometry::PointCloud test_;
};
