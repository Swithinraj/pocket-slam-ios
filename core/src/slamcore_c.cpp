#include "public/slamcore_c.h"

#include "slamcore/system.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

using namespace slamcore;

struct SlamHandle {
    std::unique_ptr<System> sys;
    std::vector<std::pair<double, Pose>> finalized;
    GrayImage img;     // reused every frame (assign() keeps capacity: no per-frame allocation)
    DepthMap depth;
};

extern "C" {

SlamHandle* slam_create(const double R_bc[9], const double p_bc[3], double focal) {
    SystemConfig cfg;
    cfg.extrinsics.R_bc << R_bc[0], R_bc[1], R_bc[2], R_bc[3], R_bc[4], R_bc[5], R_bc[6], R_bc[7], R_bc[8];
    cfg.extrinsics.p_bc = Vec3(p_bc[0], p_bc[1], p_bc[2]);
    cfg.estimator.focal = focal;
    auto* h = new SlamHandle;
    h->sys = std::make_unique<System>(cfg);
    return h;
}

void slam_destroy(SlamHandle* h) { delete h; }

void slam_add_imu(SlamHandle* h, double t, double ax, double ay, double az, double gx, double gy, double gz) {
    ImuSample s;
    s.t = t;
    s.acc = Vec3(ax, ay, az);
    s.gyr = Vec3(gx, gy, gz);
    h->sys->addImu(s);
}

void slam_downsample_luma(const uint8_t* src, int32_t width, int32_t height, int32_t stride, int32_t factor, uint8_t* dst) {
    const int W = width / factor, H = height / factor, n = factor * factor;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int s = 0;
            for (int v = 0; v < factor; ++v) {
                const uint8_t* row = src + static_cast<size_t>(y * factor + v) * stride + x * factor;
                for (int u = 0; u < factor; ++u) s += row[u];
            }
            dst[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(s / n);
        }
}

int32_t slam_add_frame(SlamHandle* h, double t, const uint8_t* gray, int32_t width, int32_t height, double fx, double fy,
                       double cx, double cy, const float* depth, const uint8_t* conf, int32_t dw, int32_t dh,
                       SlamFrameResult* out) {
    const auto t0 = std::chrono::steady_clock::now();
    GrayImage& img = h->img;
    img.width = width;
    img.height = height;
    img.data.assign(gray, gray + static_cast<size_t>(width) * height);
    PinholeCamera cam;
    cam.fx = fx;
    cam.fy = fy;
    cam.cx = cx;
    cam.cy = cy;
    cam.width = width;
    cam.height = height;
    DepthMap& dm = h->depth;
    dm.width = dm.height = 0;
    if (depth && dw > 0 && dh > 0) {
        dm.width = dw;
        dm.height = dh;
        dm.depth.assign(depth, depth + static_cast<size_t>(dw) * dh);
        if (conf) dm.confidence.assign(conf, conf + static_cast<size_t>(dw) * dh);
        else dm.confidence.clear();
    }
    const FrameResult r = h->sys->addFrame(t, img, cam, dm.valid() ? &dm : nullptr);
    for (auto& kf : h->sys->takeFinalizedKeyframes()) h->finalized.push_back(kf);
    if (out) {
        out->t = r.t;
        out->initialized = r.initialized;
        out->keyframe = r.keyframe;
        for (int i = 0; i < 9; ++i) out->R[i] = r.camera.R(i / 3, i % 3);
        for (int i = 0; i < 3; ++i) {
            out->p[i] = r.camera.p(i);
            out->v[i] = r.velocity(i);
        }
        out->tracked = r.tracked;
        out->landmarks = r.landmarks;
        out->solve_ms = r.solve_ms;
        out->reinitialized = r.reinitialized;
        out->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    return r.initialized ? 1 : 0;
}

int32_t slam_map_points(SlamHandle* h, float* xyz, int32_t max_points) {
    const auto pts = h->sys->mapPoints();
    const int n = std::min<int>(max_points, static_cast<int>(pts.size()));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) xyz[3 * i + k] = static_cast<float>(pts[i](k));
    return n;
}

int32_t slam_features(SlamHandle* h, float* uv, int32_t max_features) {
    const auto& f = h->sys->features();
    const int n = std::min<int>(max_features, static_cast<int>(f.size()));
    for (int i = 0; i < n; ++i) {
        uv[2 * i] = static_cast<float>(f[i].px.x());
        uv[2 * i + 1] = static_cast<float>(f[i].px.y());
    }
    return n;
}

int32_t slam_take_finalized(SlamHandle* h, double* out, int32_t max_keyframes) {
    const int n = std::min<int>(max_keyframes, static_cast<int>(h->finalized.size()));
    for (int i = 0; i < n; ++i) {
        const auto& [t, P] = h->finalized[i];
        double* o = out + 13 * i;
        o[0] = t;
        for (int k = 0; k < 9; ++k) o[1 + k] = P.R(k / 3, k % 3);
        for (int k = 0; k < 3; ++k) o[10 + k] = P.p(k);
    }
    h->finalized.erase(h->finalized.begin(), h->finalized.begin() + n);
    return n;
}

int32_t slam_window_keyframes(SlamHandle* h, double* out, int32_t max_keyframes) {
    const auto kfs = h->sys->windowKeyframes();
    const int n = std::min<int>(max_keyframes, static_cast<int>(kfs.size()));
    for (int i = 0; i < n; ++i) {
        const auto& [t, P] = kfs[i];
        double* o = out + 13 * i;
        o[0] = t;
        for (int k = 0; k < 9; ++k) o[1 + k] = P.R(k / 3, k % 3);
        for (int k = 0; k < 3; ++k) o[10 + k] = P.p(k);
    }
    return n;
}

}  // extern "C"
