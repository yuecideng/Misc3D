#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <misc3d/common/normal_estimation.h>
#include <misc3d/logger.h>
#include <misc3d/utils.h>

namespace misc3d {
namespace common {

void VectorToPointer(const std::vector<Eigen::Vector3d> &data, double *ptr) {
    const size_t num = data.size();
#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        ptr[3 * i] = data[i](0);
        ptr[3 * i + 1] = data[i](1);
        ptr[3 * i + 2] = data[i](2);
    }
}

void PointerToVector(const double *ptr, int num,
                     std::vector<Eigen::Vector3d> &data) {
    data.resize(num);
#pragma omp parallel for
    for (size_t i = 0; i < num; i++) {
        Eigen::Vector3d d(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]);
        data[i] = d;
    }
}

template <typename T0, typename T1>
void SumDense(const T0 *data, const bool *mask, const unsigned int w,
              const unsigned int h, const unsigned int k, T1 *dst) {
    const size_t range_rows[2] = {k, h - k}, range_cols[2] = {k + 1, w - k},
                 double_k = 2 * k;
#pragma omp parallel for
    for (int r = (int)range_rows[0]; r < (int)range_rows[1]; r++) {
        size_t idx_r = (size_t)r * w;
        T1 *ptr = dst + idx_r + k;
        *ptr = 0;
        for (size_t r0 = (size_t)r - k, idx_r0; r0 <= (size_t)r + k; r0++) {
            idx_r0 = r0 * w;
            for (size_t c0 = 0; c0 <= double_k; c0++) {
                *ptr += data[idx_r0 + c0];
            }
        }
        ptr++;
        for (size_t c = range_cols[0]; c < range_cols[1]; c++, ptr++) {
            *ptr = *(ptr - 1);
            for (size_t r0 = (size_t)r - k, c0 = c - k - 1, c1 = c + k;
                 r0 <= (size_t)r + k; r0++) {
                *ptr += *(data + r0 * w + c1) - *(data + r0 * w + c0);
            }
        }
    }
}

void CalcNormalsFromPointMap(const double *xyzs, const unsigned int w,
                             const unsigned int h, const unsigned int k,
                             double *normals, const double view_point[3]) {
    size_t expand_w = w + 2 * k, expand_h = h + 2 * k,
           expand_wh = expand_w * expand_h;
    const size_t range_rows[2] = {k, expand_h - k},
                 range_cols[2] = {k, expand_w - k};
    bool *mask = (bool *)calloc(expand_wh, sizeof(bool));
    double *buffer0 = (double *)calloc(expand_wh * 9, sizeof(double));
    double *x = buffer0;
    double *y = x + expand_wh;
    double *z = y + expand_wh;
    double *xx = z + expand_wh;
    double *xy = xx + expand_wh;
    double *xz = xy + expand_wh;
    double *yy = xz + expand_wh;
    double *yz = yy + expand_wh;
    double *zz = yz + expand_wh;

#pragma omp parallel for
    for (int r = (int)range_rows[0]; r < (int)range_rows[1]; r++) {
        size_t idx_r = (size_t)r * expand_w, idx;
        const double *xyz = xyzs + ((size_t)r - k) * w * 3;
        for (size_t c = range_cols[0]; c < range_cols[1]; c++, xyz += 3) {
            idx = idx_r + c;
            if (xyz[2] == xyz[2]) {
                mask[idx] = true;
                x[idx] = xyz[0];
                y[idx] = xyz[1];
                z[idx] = xyz[2];
                xx[idx] = xyz[0] * xyz[0];
                xy[idx] = xyz[0] * xyz[1];
                xz[idx] = xyz[0] * xyz[2];
                yy[idx] = xyz[1] * xyz[1];
                yz[idx] = xyz[1] * xyz[2];
                zz[idx] = xyz[2] * xyz[2];
            }
        }
    }

    size_t *neighbor_nums = new size_t[expand_wh];
    double *buffer1 = (double *)malloc(expand_wh * 9 * sizeof(double));
    double *sum_x = buffer1;
    double *sum_y = sum_x + expand_wh;
    double *sum_z = sum_y + expand_wh;
    double *sum_xx = sum_z + expand_wh;
    double *sum_xy = sum_xx + expand_wh;
    double *sum_xz = sum_xy + expand_wh;
    double *sum_yy = sum_xz + expand_wh;
    double *sum_yz = sum_yy + expand_wh;
    double *sum_zz = sum_yz + expand_wh;

    SumDense(x, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_x);
    SumDense(y, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_y);
    SumDense(z, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_z);
    SumDense(xx, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_xx);
    SumDense(xy, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_xy);
    SumDense(xz, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_xz);
    SumDense(yy, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_yy);
    SumDense(yz, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_yz);
    SumDense(zz, mask, (unsigned int)expand_w, (unsigned int)expand_h, k, sum_zz);
    SumDense(mask, mask, (unsigned int)expand_w, (unsigned int)expand_h, k,
             neighbor_nums);

#pragma omp parallel for
    for (int r = (int)range_rows[0]; r < (int)range_rows[1]; r++) {
        size_t idx_r = (size_t)r * expand_w, idx;
        for (size_t c = range_cols[0]; c < range_cols[1]; c++) {
            idx = idx_r + c;
            if (!mask[idx])
                continue;
            double scale = 1. / neighbor_nums[idx];
            double hat_x = sum_x[idx] * scale, hat_y = sum_y[idx] * scale,
                   hat_z = sum_z[idx] * scale;
            double covariance[3][3] = {sum_xx[idx] * scale - hat_x * hat_x,
                                       sum_xy[idx] * scale - hat_x * hat_y,
                                       sum_xz[idx] * scale - hat_x * hat_z,
                                       0,
                                       sum_yy[idx] * scale - hat_y * hat_y,
                                       sum_yz[idx] * scale - hat_y * hat_z,
                                       0,
                                       0,
                                       sum_zz[idx] * scale - hat_z * hat_z};
            covariance[1][0] = covariance[0][1];
            covariance[2][0] = covariance[0][2];
            covariance[2][1] = covariance[1][2];

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor | Eigen::DontAlign>
                covariance_matrix(covariance[0]);
            Eigen::SelfAdjointEigenSolver<
                Eigen::Matrix<double, 3, 3, Eigen::RowMajor | Eigen::DontAlign>>
                solver;
            solver.compute(covariance_matrix, Eigen::ComputeEigenvectors);
            Eigen::Vector3d v0 = solver.eigenvectors().col(0);
            double *temp_n = v0.data();
            if ((view_point[0] - x[idx]) * temp_n[0] +
                    (view_point[1] - y[idx]) * temp_n[1] +
                    (view_point[2] - z[idx]) * temp_n[2] <
                0) {
                temp_n[0] *= -1;
                temp_n[1] *= -1;
                temp_n[2] *= -1;
            }
            memcpy(normals + (((size_t)r - k) * w + c - k) * 3, v0.data(),
                   sizeof(double) * 3);
        }
    }

    delete[] neighbor_nums;
    free(buffer0);
    free(buffer1);
    free(mask);
}

void EstimateNormalsFromMap(const PointCloudPtr &pc,
                            const std::tuple<int, int> shape, int k,
                            const std::array<double, 3> &view_point) {
    const size_t num = pc->points_.size();
    const int w = std::get<0>(shape);
    const int h = std::get<1>(shape);

    if (num != w * h) {
        MISC3D_ERROR("The point cloud size is not equal to given point map size.");
        return;
    }

    double *normals_ptr = new double[num * 3];
    double *points_ptr = new double[num * 3];
    VectorToPointer(pc->points_, points_ptr);
    CalcNormalsFromPointMap(points_ptr, w, h, k, normals_ptr, view_point.data());

    // assign normals
    std::vector<Eigen::Vector3d> normals;
    PointerToVector(normals_ptr, num, normals);
    pc->normals_ = normals;

    delete[] normals_ptr;
    delete[] points_ptr;
}

}  // namespace common
}  // namespace misc3d