// Runs the VLIO core on a slamtogo recording exactly as the app would (causal, frame by frame).
//
//   run_recording <recording_dir> <out_dir> [--track-width 480]
//
// Outputs (camera poses, OpenCV camera convention, world z-up):
//   vlio_frames_tum.txt     real-time pose of every RGB-D frame
//   vlio_keyframes_tum.txt  final (smoothed) keyframe poses
//   vlio_map.ply            landmarks still in the map at the end
//   vlio_stats.txt          timing
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

#include "slamcore/system.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using namespace slamcore;

namespace {

std::vector<std::vector<double>> readCsv(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::vector<double>> rows;
    std::string line;
    std::getline(in, line);   // header
    while (std::getline(in, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        if (!row.empty()) rows.push_back(std::move(row));
    }
    return rows;
}

bool loadGray(const std::string& path, int targetWidth, GrayImage& out) {
    int w, h, c;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &c, 1);
    if (!px) return false;
    const int f = std::max(1, w / targetWidth);   // integer box downscale
    out.width = w / f;
    out.height = h / f;
    out.data.assign(static_cast<size_t>(out.width) * out.height, 0);
    for (int y = 0; y < out.height; ++y)
        for (int x = 0; x < out.width; ++x) {
            int s = 0;
            for (int v = 0; v < f; ++v)
                for (int u = 0; u < f; ++u) s += px[(y * f + v) * w + x * f + u];
            out.data[static_cast<size_t>(y) * out.width + x] = static_cast<uint8_t>(s / (f * f));
        }
    stbi_image_free(px);
    return true;
}

bool loadDepth(const std::string& depthPath, const std::string& confPath, DepthMap& out) {
    int w, h, c;
    unsigned short* d = stbi_load_16(depthPath.c_str(), &w, &h, &c, 1);
    if (!d) return false;
    out.width = w;
    out.height = h;
    out.depth.resize(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < out.depth.size(); ++i) out.depth[i] = d[i] / 1000.0f;
    stbi_image_free(d);
    unsigned char* cf = stbi_load(confPath.c_str(), &w, &h, &c, 1);
    if (cf) {
        out.confidence.assign(cf, cf + static_cast<size_t>(w) * h);
        stbi_image_free(cf);
    }
    return true;
}

void writeTum(std::ofstream& f, double t, const Pose& P) {
    const Eigen::Quaterniond q(P.R);
    char buf[256];
    std::snprintf(buf, sizeof buf, "%.6f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n", t, P.p.x(), P.p.y(), P.p.z(), q.x(), q.y(),
                  q.z(), q.w());
    f << buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: run_recording <recording_dir> <out_dir> [--track-width 480]\n";
        return 1;
    }
    const std::string root = argv[1], outDir = argv[2];
    int trackWidth = 480;
    bool useDepth = true, hostDepth = true, targetDepth = true;
    double flatness = -1, sigmaScale = 1, pixelSigma = -1;
    Vec3 pbc = Vec3::Zero();
    int maxAge = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        const bool hasValue = i + 1 < argc;
        if (a == "--track-width" && hasValue) trackWidth = std::stoi(argv[++i]);
        else if (a == "--no-depth") useDepth = false;
        else if (a == "--flatness" && hasValue) flatness = std::stod(argv[++i]);
        else if (a == "--depth-sigma-scale" && hasValue) sigmaScale = std::stod(argv[++i]);
        else if (a == "--pixel-sigma" && hasValue) pixelSigma = std::stod(argv[++i]);
        else if (a == "--max-age" && hasValue) maxAge = std::stoi(argv[++i]);
        else if (a == "--no-host-depth") hostDepth = false;
        else if (a == "--no-target-depth") targetDepth = false;
        else if (a == "--pbc" && i + 3 < argc) {          // camera position in the IMU/body frame [m]
            pbc = Vec3(std::stod(argv[i + 1]), std::stod(argv[i + 2]), std::stod(argv[i + 3]));
            i += 3;
        }
    }

    const auto frames = readCsv(root + "/frames.csv");   // frame,t,tracking,tx..qw,fx,fy,cx,cy,w,h,exp,image
    const auto acc = readCsv(root + "/accel.csv");
    const auto gyr = readCsv(root + "/gyro.csv");
    std::string lidarDir = "/lidar/";

    // Merge accel onto gyro timestamps.
    std::vector<ImuSample> imu;
    size_t ai = 0;
    for (const auto& g : gyr) {
        const double t = g[0];
        while (ai + 1 < acc.size() && acc[ai + 1][0] < t) ++ai;
        if (ai + 1 >= acc.size()) break;
        const auto& a0 = acc[ai];
        const auto& a1 = acc[ai + 1];
        const double w = std::clamp((t - a0[0]) / (a1[0] - a0[0]), 0.0, 1.0);
        ImuSample s;
        s.t = t;
        s.acc = (1 - w) * Vec3(a0[1], a0[2], a0[3]) + w * Vec3(a1[1], a1[2], a1[3]);
        s.gyr = Vec3(g[1], g[2], g[3]);
        imu.push_back(s);
    }

    // Probe the first RGB-D frame: LiDAR folder (v2: lidar/, v1: depth/) and tracking-image focal length.
    double focal = 0;
    for (const auto& fr : frames)
        if (fr[17] > 0.5) {
            char name[16];
            std::snprintf(name, sizeof name, "%06d", static_cast<int>(fr[0]));
            if (!std::ifstream(root + "/lidar/" + name + ".png").good()) lidarDir = "/depth/";
            GrayImage probe;
            if (loadGray(root + "/rgb/" + name + ".jpg", trackWidth, probe)) focal = fr[10] * probe.width / fr[14];
            break;
        }
    if (focal <= 0) {
        std::cerr << "no RGB-D frames found\n";
        return 1;
    }

    SystemConfig cfg;
    // Body (device: x right, y up, z out of screen) <- camera (OpenCV, sensor landscape).
    cfg.extrinsics.R_bc << 0, -1, 0, -1, 0, 0, 0, 0, -1;
    cfg.estimator.focal = focal;
    if (flatness > 0) cfg.depth_flatness = flatness;
    cfg.depth_sigma_scale = sigmaScale;
    cfg.extrinsics.p_bc = pbc;
    cfg.tracker.max_age = maxAge;
    cfg.estimator.depth_on_host = hostDepth;
    cfg.estimator.depth_on_targets = targetDepth;
    if (pixelSigma > 0) cfg.estimator.pixel_sigma = pixelSigma;

    System sys(cfg);
    std::ofstream fo(outDir + "/vlio_frames_tum.txt"), fk(outDir + "/vlio_keyframes_tum.txt");
    fo << "# timestamp tx ty tz qx qy qz qw  (camera->world, OpenCV camera, world z-up)\n";
    fk << "# timestamp tx ty tz qx qy qz qw  (final keyframe poses)\n";

    size_t ii = 0;
    int n = 0, kfs = 0;
    double solveTotal = 0, solveMax = 0, frontTotal = 0;
    for (const auto& fr : frames) {
        if (fr[17] < 0.5) continue;   // no RGB-D for this frame
        const double t = fr[1];
        while (ii < imu.size() && imu[ii].t <= t + 0.02) sys.addImu(imu[ii++]);

        char name[16];
        std::snprintf(name, sizeof name, "%06d", static_cast<int>(fr[0]));
        GrayImage img;
        if (!loadGray(root + "/rgb/" + name + ".jpg", trackWidth, img)) continue;
        DepthMap depth;
        const bool hasDepth = loadDepth(root + lidarDir + name + ".png", root + "/confidence/" + name + ".png", depth);

        PinholeCamera cam;
        const double s = static_cast<double>(img.width) / fr[14];   // frames.csv intrinsics are for width x height
        cam.fx = fr[10] * s;
        cam.fy = fr[11] * s;
        cam.cx = fr[12] * s;
        cam.cy = fr[13] * s;
        cam.width = img.width;
        cam.height = img.height;

        const auto t0 = std::chrono::steady_clock::now();
        const FrameResult r = sys.addFrame(t, img, cam, hasDepth && useDepth ? &depth : nullptr);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        frontTotal += ms - r.solve_ms;
        if (r.reinitialized) std::printf("re-initialised at t=%.2f (tracked %d)\n", t - frames[0][1], r.tracked);
        if (r.initialized) writeTum(fo, t, r.camera);
        if (r.keyframe) {
            ++kfs;
            solveTotal += r.solve_ms;
            solveMax = std::max(solveMax, r.solve_ms);
        }
        for (const auto& [tk, P] : sys.takeFinalizedKeyframes()) writeTum(fk, tk, P);
        if (++n % 100 == 0)
            std::printf("frame %4d  t=%7.2f  tracked %3d  landmarks %3d  kf %d  solve %.1f ms\n", n, t - frames[0][1],
                        r.tracked, r.landmarks, kfs, r.solve_ms);
    }
    for (const auto& [tk, P] : sys.windowKeyframes()) writeTum(fk, tk, P);

    std::ofstream fp(outDir + "/vlio_map.ply");
    const auto pts = sys.mapPoints();
    fp << "ply\nformat ascii 1.0\nelement vertex " << pts.size()
       << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto& p : pts) fp << p.x() << " " << p.y() << " " << p.z() << "\n";

    std::ofstream fs(outDir + "/vlio_stats.txt");
    fs << "frames " << n << "\nkeyframes " << kfs << "\nsolve_mean_ms " << solveTotal / std::max(kfs, 1)
       << "\nsolve_max_ms " << solveMax << "\nfrontend_mean_ms " << frontTotal / std::max(n, 1) << "\n";
    std::printf("done: %d frames, %d keyframes, solve mean %.1f ms max %.1f ms, front-end mean %.1f ms\n", n, kfs,
                solveTotal / std::max(kfs, 1), solveMax, frontTotal / std::max(n, 1));
    return 0;
}
