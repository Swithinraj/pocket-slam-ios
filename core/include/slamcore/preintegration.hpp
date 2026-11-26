#pragma once

#include "slamcore/types.hpp"

#include <vector>

namespace slamcore {

/// On-manifold IMU preintegration (Forster et al., TRO 2017) between two keyframes,
/// with first-order bias correction and noise covariance for [dtheta, dv, dp].
class Preintegration {
public:
    Preintegration(const Vec3& ba, const Vec3& bg, const ImuNoise& noise);

    /// Integrate a constant measurement over dt (callers pass the midpoint of two samples).
    void integrate(double dt, const Vec3& acc, const Vec3& gyr);
    /// Re-run integration from the stored measurements with new linearisation biases.
    void reintegrate(const Vec3& ba, const Vec3& bg);

    double dt() const { return dt_; }
    const Vec3& ba() const { return ba_; }
    const Vec3& bg() const { return bg_; }

    /// Bias-corrected deltas.
    Mat3 deltaR(const Vec3& bg) const;
    Vec3 deltaV(const Vec3& ba, const Vec3& bg) const;
    Vec3 deltaP(const Vec3& ba, const Vec3& bg) const;

    /// Covariance of [dtheta, dv, dp] (9x9) and bias random walk over dt.
    const Mat<9, 9>& covariance() const { return cov_; }
    const ImuNoise& noise() const { return noise_; }

private:
    void reset();
    void step(double dt, const Vec3& acc, const Vec3& gyr);

    ImuNoise noise_;
    Vec3 ba_, bg_;
    double dt_ = 0;
    Mat3 dR_;
    Vec3 dv_, dp_;
    Mat3 JR_bg_, Jv_ba_, Jv_bg_, Jp_ba_, Jp_bg_;
    Mat<9, 9> cov_;

    struct Meas {
        double dt;
        Vec3 acc, gyr;
    };
    std::vector<Meas> meas_;
};

}  // namespace slamcore
