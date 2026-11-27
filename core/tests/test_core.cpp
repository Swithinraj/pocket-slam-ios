// Synthetic ground-truth tests for the VLIO core:
//  1. IMU preintegration reproduces the true relative motion.
//  2. The sliding-window estimator tracks a hand-held-like trajectory from noisy IMU (with biases),
//     noisy feature observations and noisy LiDAR depth, through many marginalisations.
//  3. Loop verification recovers a known relative pose between two views of a textured wall.
#include "slamcore/estimator.hpp"
#include "slamcore/loop_closure.hpp"
#include "slamcore/so3.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

using namespace slamcore;

namespace {

const Vec3 kGravity(0, 0, -9.81);
Mat3 R_bc() {
    Mat3 R;
    R << 0, -1, 0, -1, 0, 0, 0, 0, -1;
    return R;
}

// Camera looks horizontally along yaw psi, with a small wobble.
Mat3 R_wc(double t) {
    const double psi = 0.5 * t + 0.3 * std::sin(0.9 * t);
    const Vec3 f(std::cos(psi), std::sin(psi), 0), right(std::sin(psi), -std::cos(psi), 0), down(0, 0, -1);
    Mat3 R;
    R.col(0) = right;
    R.col(1) = down;
    R.col(2) = f;
    return R * so3::exp(Vec3(0.05 * std::sin(2.0 * t), 0.08 * std::sin(1.7 * t), 0.03 * std::sin(2.3 * t)));
}
Mat3 R_wb(double t) { return R_wc(t) * R_bc().transpose(); }
Vec3 p_wb(double t) { return Vec3(1.0 * std::cos(0.4 * t), 1.0 * std::sin(0.4 * t), 0.1 * std::sin(1.3 * t)); }

ImuSample trueImu(double t) {
    const double h = 1e-4;
    const Vec3 a = (p_wb(t + h) - 2 * p_wb(t) + p_wb(t - h)) / (h * h);
    const Vec3 w = so3::log(R_wb(t - h).transpose() * R_wb(t + h)) / (2 * h);
    return {t, R_wb(t).transpose() * (a - kGravity), w};
}
Vec3 v_wb(double t) { return (p_wb(t + 1e-5) - p_wb(t - 1e-5)) / 2e-5; }

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    failures += ok ? 0 : 1;
}

void testPreintegration() {
    std::printf("Preintegration vs ground truth (1 s, 200 Hz, no noise)\n");
    Preintegration pre(Vec3::Zero(), Vec3::Zero(), ImuNoise{});
    const double t0 = 2.0, t1 = 3.0, dt = 1.0 / 200;
    for (double t = t0; t < t1 - 1e-9; t += dt) {
        const ImuSample a = trueImu(t), b = trueImu(t + dt);
        pre.integrate(dt, 0.5 * (a.acc + b.acc), 0.5 * (a.gyr + b.gyr));
    }
    const Mat3 Ri = R_wb(t0);
    const double T = t1 - t0;
    const Mat3 Rj = Ri * pre.deltaR(Vec3::Zero());
    const Vec3 pj = p_wb(t0) + v_wb(t0) * T + 0.5 * kGravity * T * T + Ri * pre.deltaP(Vec3::Zero(), Vec3::Zero());
    const double rotErr = so3::log(Rj.transpose() * R_wb(t1)).norm() * 180 / M_PI;
    const double posErr = (pj - p_wb(t1)).norm();
    std::printf("  rotation error %.4f deg, position error %.2f mm\n", rotErr, posErr * 1000);
    check(rotErr < 0.05 && posErr < 0.005, "preintegrated motion matches the true motion");
}

void testEstimator() {
    std::printf("Sliding-window VLIO on a synthetic 12 s hand-held trajectory\n");
    std::mt19937 rng(7);
    std::normal_distribution<double> N(0, 1);
    std::uniform_real_distribution<double> U(0, 1);

    std::vector<Vec3> world;
    for (int i = 0; i < 4000; ++i) {
        const double a = 2 * M_PI * U(rng);
        world.emplace_back(4.0 * std::cos(a), 4.0 * std::sin(a), -1.0 + 2.5 * U(rng));
    }

    const double f = 500.0, pxSigma = 0.7;
    EstimatorConfig cfg;
    cfg.focal = f;
    ImuNoise noise;
    Extrinsics ext;
    ext.R_bc = R_bc();
    Estimator est(cfg, noise, ext);

    const Vec3 bgTrue(0.004, -0.003, 0.002), baTrue(0.06, -0.04, 0.03);
    const double imuRate = 200, kfDt = 0.15, T = 12.0, t0 = 0.5;
    const double sg = noise.gyr_noise * std::sqrt(imuRate) * 0.2, sa = noise.acc_noise * std::sqrt(imuRate) * 0.2;

    auto observe = [&](double t) {
        std::vector<FeatureObs> obs;
        const Mat3 Rwc = R_wc(t);
        const Vec3 pwc = p_wb(t) + R_wb(t) * ext.p_bc;
        for (size_t i = 0; i < world.size() && obs.size() < 150; ++i) {
            const Vec3 Pc = Rwc.transpose() * (world[i] - pwc);
            if (Pc.z() < 0.3 || Pc.z() > 8) continue;
            const Vec2 xn(Pc.x() / Pc.z(), Pc.y() / Pc.z());
            if (std::abs(xn.x()) > 0.6 || std::abs(xn.y()) > 0.45) continue;
            FeatureObs o{static_cast<int>(i), xn + Vec2(N(rng), N(rng)) * pxSigma / f};
            if (i % 5 < 3) {   // 60% get LiDAR depth
                o.depth_sigma = 0.01 + 0.01 * Pc.z();
                o.depth = Pc.z() + N(rng) * o.depth_sigma;
            }
            obs.push_back(o);
        }
        return obs;
    };
    auto imuAt = [&](double t) {
        ImuSample s = trueImu(t);
        s.gyr += bgTrue + Vec3(N(rng), N(rng), N(rng)) * sg;
        s.acc += baTrue + Vec3(N(rng), N(rng), N(rng)) * sa;
        return s;
    };

    est.initialize(t0, R_wb(t0), Vec3::Zero(), observe(t0));
    std::vector<std::pair<double, NavState>> all;
    double tPrev = t0, solveMax = 0, solveSum = 0;
    int kfs = 0;
    ImuSample prev = imuAt(t0);
    for (double t = t0 + kfDt; t < t0 + T; t += kfDt) {
        auto pre = std::make_unique<Preintegration>(est.latest().x.ba, est.latest().x.bg, noise);
        const double dt = 1.0 / imuRate;
        double tt = tPrev;
        while (tt + dt <= t + 1e-9) {
            const ImuSample s = imuAt(tt + dt);
            pre->integrate(dt, 0.5 * (prev.acc + s.acc), 0.5 * (prev.gyr + s.gyr));
            prev = s;
            tt += dt;
        }
        est.addKeyframe(t, std::move(pre), observe(t));
        est.process();
        solveMax = std::max(solveMax, est.lastSolveMs());
        solveSum += est.lastSolveMs();
        ++kfs;
        for (auto& x : est.takeFinalized()) all.push_back(x);
        tPrev = tt;
    }
    for (const auto& kf : est.window()) all.push_back({kf.t, kf.x});

    // Errors (world frames coincide: the first state was initialised at the true pose with p = 0).
    const Vec3 origin = p_wb(t0);
    double se = 0, maxe = 0, rmax = 0;
    for (const auto& [t, x] : all) {
        const double e = (x.p - (p_wb(t) - origin)).norm();
        se += e * e;
        maxe = std::max(maxe, e);
        rmax = std::max(rmax, so3::log(x.R.transpose() * R_wb(t)).norm() * 180 / M_PI);
    }
    const double rmse = std::sqrt(se / all.size());
    const NavState& last = est.latest().x;
    std::printf("  %zu keyframes, path %.1f m: position RMSE %.1f mm, max %.1f mm, max rotation error %.2f deg\n",
                all.size(), 0.4 * T * 1.0, rmse * 1000, maxe * 1000, rmax);
    std::printf("  gyro bias est (%.4f %.4f %.4f) true (%.4f %.4f %.4f)\n", last.bg.x(), last.bg.y(), last.bg.z(),
                bgTrue.x(), bgTrue.y(), bgTrue.z());
    std::printf("  acc  bias est (%.3f %.3f %.3f) true (%.3f %.3f %.3f)\n", last.ba.x(), last.ba.y(), last.ba.z(),
                baTrue.x(), baTrue.y(), baTrue.z());
    std::printf("  solve time mean %.1f ms, max %.1f ms per keyframe\n", solveSum / kfs, solveMax);
    check(rmse < 0.03, "position RMSE < 3 cm over 4.8 m");
    check(rmax < 1.0, "rotation error < 1 deg");
    check((last.bg - bgTrue).norm() < 0.002, "gyro bias recovered (< 0.002 rad/s)");
}

void testLoopVerification() {
    std::printf("loop verification\n");
    // Fronto-parallel textured wall at Z = 2 m; view b is translated by t (camera b -> camera a: X_a = X_b + t).
    const int W = 480, H = 360;
    PinholeCamera cam;
    cam.fx = cam.fy = 350;
    cam.cx = W / 2.0;
    cam.cy = H / 2.0;
    cam.width = W;
    cam.height = H;
    const double Z = 2.0;
    const Vec3 t(0.12, -0.05, 0.0);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0, 1);
    struct Blob { double x, y, r, a; };
    std::vector<Blob> blobs(900);
    for (auto& b : blobs) b = {U(rng) * 4 - 2, U(rng) * 3 - 1.5, 0.01 + 0.04 * U(rng), U(rng) * 200 - 100};
    auto render = [&](const Vec3& offset) {     // camera centre at offset (wall coordinates)
        GrayImage img;
        img.width = W;
        img.height = H;
        img.data.resize(static_cast<size_t>(W) * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const double X = (x - cam.cx) / cam.fx * Z + offset.x(), Y = (y - cam.cy) / cam.fy * Z + offset.y();
                double v = 128;
                for (const Blob& b : blobs) {
                    const double d2 = (X - b.x) * (X - b.x) + (Y - b.y) * (Y - b.y);
                    if (d2 < 9 * b.r * b.r) v += b.a * std::exp(-d2 / (2 * b.r * b.r));
                }
                img.data[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
            }
        return img;
    };
    DepthMap depth;
    depth.width = 256;
    depth.height = 192;
    depth.depth.assign(256 * 192, static_cast<float>(Z));
    depth.confidence.assign(256 * 192, 2);
    LoopCloser lc;
    lc.addKeyframe(0.0, 0, render(Vec3::Zero()), cam, &depth);
    lc.addKeyframe(30.0, 0, render(t), cam, &depth);
    LoopEdge e;
    LoopStats st;
    Pose prior;
    prior.p = t + Vec3(0.05, 0.03, 0.02);      // trajectory estimate with drift
    prior.R = so3::exp(Vec3(0, 0, 0.02));
    const bool ok = lc.verify(0, 1, prior, e, &st);
    std::printf("  keypoints %zu / %zu, inliers %d, t = (%.3f %.3f %.3f), rot %.3f deg\n", lc.keyframes()[0].kps.size(),
                lc.keyframes()[1].kps.size(), e.inliers, e.T_ab.p.x(), e.T_ab.p.y(), e.T_ab.p.z(),
                Eigen::AngleAxisd(e.T_ab.R).angle() * 180 / M_PI);
    check(ok, "loop verified");
    check(ok && (e.T_ab.p - t).norm() < 0.01, "relative position error < 1 cm");
    check(ok && Eigen::AngleAxisd(e.T_ab.R).angle() < 0.3 * M_PI / 180, "relative rotation error < 0.3 deg");
}

}  // namespace

int main() {
    testPreintegration();
    testEstimator();
    testLoopVerification();
    std::printf(failures ? "\n%d FAILED\n" : "\nALL PASSED\n", failures);
    return failures ? 1 : 0;
}
