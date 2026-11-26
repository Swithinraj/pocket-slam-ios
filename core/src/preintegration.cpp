#include "slamcore/preintegration.hpp"

#include "slamcore/so3.hpp"

namespace slamcore {

Preintegration::Preintegration(const Vec3& ba, const Vec3& bg, const ImuNoise& noise)
    : noise_(noise), ba_(ba), bg_(bg) {
    reset();
}

void Preintegration::reset() {
    dt_ = 0;
    dR_.setIdentity();
    dv_.setZero();
    dp_.setZero();
    JR_bg_.setZero();
    Jv_ba_.setZero();
    Jv_bg_.setZero();
    Jp_ba_.setZero();
    Jp_bg_.setZero();
    cov_.setZero();
}

void Preintegration::integrate(double dt, const Vec3& acc, const Vec3& gyr) {
    if (dt <= 0) return;
    meas_.push_back({dt, acc, gyr});
    step(dt, acc, gyr);
}

void Preintegration::reintegrate(const Vec3& ba, const Vec3& bg) {
    ba_ = ba;
    bg_ = bg;
    reset();
    for (const auto& m : meas_) step(m.dt, m.acc, m.gyr);
}

void Preintegration::step(double dt, const Vec3& acc, const Vec3& gyr) {
    const Vec3 a = acc - ba_;
    const Vec3 w = gyr - bg_;
    const Mat3 dRk = so3::exp(w * dt);
    const Mat3 Jr = so3::rightJacobian(w * dt);
    const Mat3 aHat = so3::hat(a);

    // Noise propagation, state order [dtheta, dv, dp].
    Mat<9, 9> A = Mat<9, 9>::Identity();
    A.block<3, 3>(0, 0) = dRk.transpose();
    A.block<3, 3>(3, 0) = -dR_ * aHat * dt;
    A.block<3, 3>(6, 0) = -0.5 * dR_ * aHat * dt * dt;
    A.block<3, 3>(6, 3) = Mat3::Identity() * dt;
    Mat<9, 3> Bg = Mat<9, 3>::Zero();
    Bg.block<3, 3>(0, 0) = Jr * dt;
    Mat<9, 3> Ba = Mat<9, 3>::Zero();
    Ba.block<3, 3>(3, 0) = dR_ * dt;
    Ba.block<3, 3>(6, 0) = 0.5 * dR_ * dt * dt;
    const double sg = noise_.gyr_noise * noise_.gyr_noise / dt;   // discrete variance
    const double sa = noise_.acc_noise * noise_.acc_noise / dt;
    cov_ = A * cov_ * A.transpose() + sg * Bg * Bg.transpose() + sa * Ba * Ba.transpose();

    // Bias Jacobians (p before v before R: each uses the previous values).
    Jp_ba_ += Jv_ba_ * dt - 0.5 * dR_ * dt * dt;
    Jp_bg_ += Jv_bg_ * dt - 0.5 * dR_ * aHat * JR_bg_ * dt * dt;
    Jv_ba_ += -dR_ * dt;
    Jv_bg_ += -dR_ * aHat * JR_bg_ * dt;
    JR_bg_ = dRk.transpose() * JR_bg_ - Jr * dt;

    // Deltas.
    dp_ += dv_ * dt + 0.5 * dR_ * a * dt * dt;
    dv_ += dR_ * a * dt;
    dR_ = so3::normalize(dR_ * dRk);
    dt_ += dt;
}

Mat3 Preintegration::deltaR(const Vec3& bg) const { return dR_ * so3::exp(JR_bg_ * (bg - bg_)); }

Vec3 Preintegration::deltaV(const Vec3& ba, const Vec3& bg) const {
    return dv_ + Jv_ba_ * (ba - ba_) + Jv_bg_ * (bg - bg_);
}

Vec3 Preintegration::deltaP(const Vec3& ba, const Vec3& bg) const {
    return dp_ + Jp_ba_ * (ba - ba_) + Jp_bg_ * (bg - bg_);
}

}  // namespace slamcore
