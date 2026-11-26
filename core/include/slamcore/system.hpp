#pragma once

#include "slamcore/estimator.hpp"
#include "slamcore/tracker.hpp"

#include <deque>
#include <memory>
#include <unordered_map>

namespace slamcore {

struct SystemConfig {
    EstimatorConfig estimator;
    TrackerConfig tracker;
    ImuNoise imu;
    Extrinsics extrinsics;
    double kf_parallax_px = 12.0;    // rotation-compensated mean feature motion since the last keyframe
    double kf_min_interval = 0.08;   // s
    double kf_max_interval = 0.5;    // s
    double kf_track_ratio = 0.6;     // new keyframe if fewer than this fraction of features survive
    double max_frame_gap = 2.0;      // s without frames (camera interrupted) -> re-initialise instead of dead-reckoning
    int min_tracked = 8;             // fewer tracked features than this -> re-initialise
    double depth_flatness = 0.01;    // LiDAR used only where the 5x5 depth spread < flatness * (1 + d)
    double depth_sigma_scale = 1.0;  // multiplies the LiDAR depth noise model
};

struct FrameResult {
    double t = 0;
    bool initialized = false;
    bool keyframe = false;
    bool reinitialized = false;      // a new map/world frame was started at this frame
    Pose camera;                     // camera -> world (OpenCV camera), world z-up gravity-aligned
    Vec3 velocity = Vec3::Zero();
    int tracked = 0;
    int landmarks = 0;
    double solve_ms = 0;
};

/// Real-time VLIO pipeline: feed IMU samples (in time order) and frames.
/// A frame at time t must only be added once IMU samples beyond t have been added.
class System {
public:
    explicit System(const SystemConfig& cfg);

    void addImu(const ImuSample& s);
    FrameResult addFrame(double t, const GrayImage& image, const PinholeCamera& cam, const DepthMap* depth);

    /// Keyframes that left the sliding window: final smoothed camera poses.
    std::vector<std::pair<double, Pose>> takeFinalizedKeyframes();
    /// Camera poses of the keyframes still in the window (latest estimates).
    std::vector<std::pair<double, Pose>> windowKeyframes() const;
    std::vector<Vec3> mapPoints() const { return estimator_.landmarkPositions(); }
    const std::vector<Feature>& features() const { return tracker_.features(); }

private:
    ImuSample interpolate(double t) const;
    template <class F> void forEachImuSegment(double t0, double t1, F&& f) const;
    std::vector<FeatureObs> observations(const std::vector<Feature>& feats, const PinholeCamera& cam,
                                         const DepthMap* depth) const;
    Pose toCamera(const NavState& x) const;

    SystemConfig cfg_;
    FeatureTracker tracker_;
    Estimator estimator_;
    std::deque<ImuSample> imu_;
    std::unique_ptr<Preintegration> pre_;     // last keyframe -> current frame
    double last_frame_t_ = -1, last_kf_t_ = -1;
    std::unordered_map<int, Vec2> kf_features_;   // feature positions at the last keyframe
    Mat3 R_bkf_bcur_ = Mat3::Identity();          // body rotation since the last keyframe (gyro)
};

}  // namespace slamcore
