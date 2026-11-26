// SLAM to Go - shared C++ core (iOS, Android, Linux tools).
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdint>
#include <vector>

namespace slamcore {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using VecX = Eigen::VectorXd;
using MatX = Eigen::MatrixXd;
template <int N> using Vec = Eigen::Matrix<double, N, 1>;
template <int R, int C> using Mat = Eigen::Matrix<double, R, C>;

/// One IMU sample in the body (device) frame.
/// acc: specific force [m/s^2] (reads +9.81 along "up" at rest), gyr: angular rate [rad/s].
struct ImuSample {
    double t = 0;
    Vec3 acc = Vec3::Zero();
    Vec3 gyr = Vec3::Zero();
};

/// Pinhole intrinsics of the image given to the tracker.
struct PinholeCamera {
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;
    Vec2 project(const Vec3& p) const { return {fx * p.x() / p.z() + cx, fy * p.y() / p.z() + cy}; }
    Vec2 normalize(const Vec2& px) const { return {(px.x() - cx) / fx, (px.y() - cy) / fy}; }
};

struct GrayImage {
    int width = 0, height = 0;
    std::vector<uint8_t> data;  // row-major, tightly packed
    uint8_t at(int x, int y) const { return data[static_cast<size_t>(y) * width + x]; }
};

/// Metric depth with the same field of view as the camera image (any resolution), e.g. iPhone LiDAR 256x192.
struct DepthMap {
    int width = 0, height = 0;
    std::vector<float> depth;           // metres, 0 = invalid
    std::vector<uint8_t> confidence;    // 0 low, 1 medium, 2 high (empty = all high)
    bool valid() const { return width > 0 && height > 0; }
};

/// Rigid transform camera -> world (OpenCV camera: x right, y down, z forward).
struct Pose {
    Mat3 R = Mat3::Identity();
    Vec3 p = Vec3::Zero();
};

/// Body (IMU) <- camera extrinsics: X_body = R_bc * X_cam + p_bc.
struct Extrinsics {
    Mat3 R_bc = Mat3::Identity();
    Vec3 p_bc = Vec3::Zero();
};

/// Continuous-time IMU noise densities.
struct ImuNoise {
    double gyr_noise = 2e-3;   // rad/s/sqrt(Hz)
    double acc_noise = 2e-2;   // m/s^2/sqrt(Hz)
    double gyr_walk = 2e-5;    // rad/s^2/sqrt(Hz)
    double acc_walk = 2e-3;    // m/s^3/sqrt(Hz)
};

}  // namespace slamcore
