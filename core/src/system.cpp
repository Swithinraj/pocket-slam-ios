#include "slamcore/system.hpp"

#include "slamcore/so3.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace slamcore {

System::System(const SystemConfig& cfg)
    : cfg_(cfg), tracker_(cfg.tracker), estimator_(cfg.estimator, cfg.imu, cfg.extrinsics) {}

void System::addImu(const ImuSample& s) {
    if (!imu_.empty() && s.t <= imu_.back().t) return;
    imu_.push_back(s);
    // Keep ~2 s before the last frame (enough for initialisation and interpolation).
    while (imu_.size() > 2 && last_frame_t_ > 0 && imu_[1].t < last_frame_t_ - 2.0) imu_.pop_front();
}

ImuSample System::interpolate(double t) const {
    if (imu_.empty()) return {};
    if (t <= imu_.front().t) return imu_.front();
    if (t >= imu_.back().t) return imu_.back();
    auto it = std::lower_bound(imu_.begin(), imu_.end(), t, [](const ImuSample& s, double v) { return s.t < v; });
    const ImuSample& b = *it;
    const ImuSample& a = *(it - 1);
    const double w = (t - a.t) / (b.t - a.t);
    return {t, (1 - w) * a.acc + w * b.acc, (1 - w) * a.gyr + w * b.gyr};
}

/// Calls f(dt, acc, gyr) with midpoint measurements for every IMU interval inside [t0, t1].
template <class F>
void System::forEachImuSegment(double t0, double t1, F&& f) const {
    if (t1 <= t0) return;
    ImuSample prev = interpolate(t0);
    for (const ImuSample& s : imu_) {
        if (s.t <= t0) continue;
        if (s.t >= t1) break;
        f(s.t - prev.t, 0.5 * (prev.acc + s.acc), 0.5 * (prev.gyr + s.gyr));
        prev = s;
    }
    const ImuSample end = interpolate(t1);
    f(t1 - prev.t, 0.5 * (prev.acc + end.acc), 0.5 * (prev.gyr + end.gyr));
}

Pose System::toCamera(const NavState& x) const {
    const Extrinsics& e = cfg_.extrinsics;
    return {x.R * e.R_bc, x.p + x.R * e.p_bc};
}

std::vector<FeatureObs> System::observations(const std::vector<Feature>& feats, const PinholeCamera& cam,
                                             const DepthMap* depth) const {
    std::vector<FeatureObs> out;
    out.reserve(feats.size());
    for (const Feature& f : feats) {
        FeatureObs o{f.id, cam.normalize(f.px)};
        if (depth && depth->valid()) {
            const int W = depth->width, H = depth->height;
            const int u = static_cast<int>(std::lround((f.px.x() + 0.5) * W / cam.width - 0.5));
            const int v = static_cast<int>(std::lround((f.px.y() + 0.5) * H / cam.height - 0.5));
            if (u >= 2 && v >= 2 && u < W - 2 && v < H - 2) {
                const float d = depth->depth[static_cast<size_t>(v) * W + u];
                const int c = depth->confidence.empty() ? 2 : depth->confidence[static_cast<size_t>(v) * W + u];
                float lo = d, hi = d;
                for (int dv = -2; dv <= 2; ++dv)
                    for (int du = -2; du <= 2; ++du) {
                        const float x = depth->depth[static_cast<size_t>(v + dv) * W + u + du];
                        lo = std::min(lo, x);
                        hi = std::max(hi, x);
                    }
                const double dd = d;
                if (c >= 1 && dd > cfg_.estimator.depth_min && dd < 5.0 && lo > 0 &&
                    (hi - lo) < cfg_.depth_flatness * (1.0 + dd)) {
                    o.depth = dd;
                    o.depth_sigma = cfg_.depth_sigma_scale * ((c == 2) ? 0.01 + 0.01 * dd : 0.03 + 0.03 * dd);
                }
            }
        }
        out.push_back(o);
    }
    return out;
}

FrameResult System::addFrame(double t, const GrayImage& image, const PinholeCamera& cam, const DepthMap* depth) {
    FrameResult res;
    res.t = t;
    const Mat3& R_bc = cfg_.extrinsics.R_bc;

    // Gyro rotation since the previous frame (bias-corrected) -> tracker prediction.
    Mat3 dR = Mat3::Identity();
    const Vec3 bg = estimator_.initialized() ? estimator_.latest().x.bg : Vec3::Zero();
    if (last_frame_t_ > 0)
        forEachImuSegment(last_frame_t_, t, [&](double dt, const Vec3&, const Vec3& w) { dR = dR * so3::exp((w - bg) * dt); });
    const Mat3 R_cur_prev = R_bc.transpose() * dR.transpose() * R_bc;
    const auto& feats = tracker_.track(image, cam, R_cur_prev);
    res.tracked = static_cast<int>(feats.size());

    // Camera interrupted (no frames) or visual tracking lost: IMU dead-reckoning diverges within
    // seconds, so start a fresh map instead (the caller re-anchors it).
    const bool wasInitialized = estimator_.initialized();
    if (wasInitialized && (t - last_frame_t_ > cfg_.max_frame_gap || res.tracked < cfg_.min_tracked)) {
        estimator_ = Estimator(cfg_.estimator, cfg_.imu, cfg_.extrinsics);
        pre_.reset();
    }

    if (!estimator_.initialized()) {
        // Gravity from the accelerometer around t (specific force points "up").
        Vec3 f = Vec3::Zero();
        int n = 0;
        for (const ImuSample& s : imu_)
            if (std::abs(s.t - t) < 0.25) { f += s.acc; ++n; }
        if (n == 0 || feats.size() < 20) {
            last_frame_t_ = t;
            return res;
        }
        f /= n;
        // R_wb such that R_wb * f_hat = +z (world z up), yaw = 0.
        const Mat3 R_wb = Eigen::Quaterniond::FromTwoVectors(f.normalized(), Vec3::UnitZ()).toRotationMatrix();
        estimator_.initialize(t, R_wb, Vec3::Zero(), observations(feats, cam, depth));
        pre_ = std::make_unique<Preintegration>(Vec3::Zero(), Vec3::Zero(), cfg_.imu);
        last_frame_t_ = last_kf_t_ = t;
        kf_features_.clear();
        for (const Feature& ft : feats) kf_features_[ft.id] = ft.px;
        R_bkf_bcur_.setIdentity();
        res.initialized = true;
        res.keyframe = true;
        res.reinitialized = wasInitialized;
        res.camera = toCamera(estimator_.latest().x);
        return res;
    }

    // Preintegrate last frame -> this frame.
    forEachImuSegment(last_frame_t_, t, [&](double dt, const Vec3& a, const Vec3& w) { pre_->integrate(dt, a, w); });
    R_bkf_bcur_ = pre_->deltaR(pre_->bg());
    last_frame_t_ = t;

    // Keyframe decision: rotation-compensated parallax, track survival, time.
    const Mat3 R_cur_kf = R_bc.transpose() * R_bkf_bcur_.transpose() * R_bc;
    double parallax = 0;
    int common = 0;
    for (const Feature& ft : feats) {
        auto it = kf_features_.find(ft.id);
        if (it == kf_features_.end()) continue;
        const Vec2 xn = cam.normalize(it->second);
        const Vec3 r = R_cur_kf * Vec3(xn.x(), xn.y(), 1.0);
        if (r.z() < 0.1) continue;
        parallax += (cam.project(r) - ft.px).norm();
        ++common;
    }
    parallax = common > 0 ? parallax / common : 1e9;
    const double ratio = kf_features_.empty() ? 0 : static_cast<double>(common) / kf_features_.size();
    const double dt_kf = t - last_kf_t_;
    const bool isKf = dt_kf >= cfg_.kf_min_interval &&
                      (parallax > cfg_.kf_parallax_px || ratio < cfg_.kf_track_ratio || dt_kf > cfg_.kf_max_interval);

    if (isKf) {
        estimator_.addKeyframe(t, std::move(pre_), observations(feats, cam, depth));
        estimator_.process();
        tracker_.remove(estimator_.takeRejected());
        const NavState& x = estimator_.latest().x;
        pre_ = std::make_unique<Preintegration>(x.ba, x.bg, cfg_.imu);
        last_kf_t_ = t;
        kf_features_.clear();
        for (const Feature& ft : tracker_.features()) kf_features_[ft.id] = ft.px;
        R_bkf_bcur_.setIdentity();
        res.keyframe = true;
        res.solve_ms = estimator_.lastSolveMs();
        res.camera = toCamera(x);
        res.velocity = x.v;
    } else {
        // IMU propagation from the latest optimised keyframe.
        const NavState& a = estimator_.latest().x;
        const double T = pre_->dt();
        const Vec3& g = cfg_.estimator.gravity;
        NavState x = a;
        x.R = so3::normalize(a.R * pre_->deltaR(a.bg));
        x.v = a.v + g * T + a.R * pre_->deltaV(a.ba, a.bg);
        x.p = a.p + a.v * T + 0.5 * g * T * T + a.R * pre_->deltaP(a.ba, a.bg);
        res.camera = toCamera(x);
        res.velocity = x.v;
    }
    res.initialized = true;
    res.landmarks = estimator_.activeLandmarks();
    return res;
}

std::vector<std::pair<double, Pose>> System::takeFinalizedKeyframes() {
    std::vector<std::pair<double, Pose>> out;
    for (const auto& [t, x] : estimator_.takeFinalized()) out.push_back({t, toCamera(x)});
    return out;
}

std::vector<std::pair<double, Pose>> System::windowKeyframes() const {
    std::vector<std::pair<double, Pose>> out;
    for (const auto& kf : estimator_.window()) out.push_back({kf.t, toCamera(kf.x)});
    return out;
}

}  // namespace slamcore
