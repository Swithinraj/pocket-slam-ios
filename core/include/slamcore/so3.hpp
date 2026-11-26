#pragma once

#include "slamcore/types.hpp"

#include <cmath>

namespace slamcore::so3 {

inline Mat3 hat(const Vec3& w) {
    Mat3 m;
    m << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
    return m;
}

inline Mat3 exp(const Vec3& w) {
    const double th = w.norm();
    if (th < 1e-10) return Mat3::Identity() + hat(w);
    return Eigen::AngleAxisd(th, w / th).toRotationMatrix();
}

inline Vec3 log(const Mat3& R) {
    Eigen::AngleAxisd aa(R);
    double th = aa.angle();
    if (th > M_PI) th -= 2 * M_PI;
    return aa.axis() * th;
}

/// Right Jacobian of SO(3): exp(w + d) ~= exp(w) exp(Jr(w) d).
inline Mat3 rightJacobian(const Vec3& w) {
    const double th = w.norm();
    const Mat3 W = hat(w);
    if (th < 1e-6) return Mat3::Identity() - 0.5 * W;
    return Mat3::Identity() - (1 - std::cos(th)) / (th * th) * W + (th - std::sin(th)) / (th * th * th) * W * W;
}

inline Mat3 rightJacobianInv(const Vec3& w) {
    const double th = w.norm();
    const Mat3 W = hat(w);
    if (th < 1e-6) return Mat3::Identity() + 0.5 * W;
    return Mat3::Identity() + 0.5 * W +
           (1.0 / (th * th) - (1 + std::cos(th)) / (2 * th * std::sin(th))) * W * W;
}

/// Re-orthonormalise (guards against drift from repeated multiplication).
inline Mat3 normalize(const Mat3& R) { return Eigen::Quaterniond(R).normalized().toRotationMatrix(); }

}  // namespace slamcore::so3
