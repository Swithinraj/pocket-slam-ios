#pragma once

#include "slamcore/preintegration.hpp"
#include "slamcore/types.hpp"

#include <deque>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

namespace slamcore {

struct EstimatorConfig {
    int window = 10;                    // keyframes in the sliding window
    double pixel_sigma = 1.0;           // feature measurement noise [px of the tracking image]
    double huber = 1.345;               // on whitened residuals
    int max_iterations = 8;             // Levenberg-Marquardt
    double outlier_px = 3.0;            // reprojection outlier threshold [px]
    double depth_outlier_sigma = 4.0;
    double depth_min = 0.15, depth_max = 8.0;
    Vec3 gravity = Vec3(0, 0, -9.81);   // world frame: z up
    double focal = 500;                 // tracking-image focal length (pixel residual scaling)
    bool depth_on_host = true;          // LiDAR depth factor on the landmark's host observation
    bool depth_on_targets = true;       // ... and on every later observation
};

/// IMU (body) navigation state.
struct NavState {
    Mat3 R = Mat3::Identity();          // body -> world
    Vec3 p = Vec3::Zero(), v = Vec3::Zero(), ba = Vec3::Zero(), bg = Vec3::Zero();
    void plus(const Vec<15>& d);        // tangent order: [dtheta, dp, dv, dba, dbg]
    Vec<15> minus(const NavState& x0) const;
};

struct FeatureObs {
    int id;
    Vec2 xn;                 // normalised image coordinates
    double depth = 0;        // metric depth from LiDAR (0 = none)
    double depth_sigma = 0;
};

struct KeyFrame {
    int id;
    double t;
    NavState x;
    std::unique_ptr<Preintegration> pre;   // previous keyframe -> this one (null for the first)
};

struct Observation {
    int kf_id;
    Vec2 xn;
    double depth, depth_sigma;
};

struct Landmark {
    int id;
    int host_kf;              // keyframe of the first observation
    double inv_depth = 0;     // in the host camera
    bool initialized = false;
    std::vector<Observation> obs;
};

/// Tightly-coupled visual-LiDAR-inertial sliding-window estimator (VINS-style), with:
///  - IMU preintegration factors between consecutive keyframes (+ bias random walk),
///  - reprojection factors for inverse-depth landmarks,
///  - LiDAR depth factors on every observation that has a reliable LiDAR depth,
///  - marginalisation of the oldest keyframe into a linear prior (Schur complement).
class Estimator {
public:
    Estimator(const EstimatorConfig& cfg, const ImuNoise& noise, const Extrinsics& ext);

    bool initialized() const { return !window_.empty(); }
    void initialize(double t, const Mat3& R_wb, const Vec3& bg, const std::vector<FeatureObs>& feats);
    void addKeyframe(double t, std::unique_ptr<Preintegration> pre, const std::vector<FeatureObs>& feats);
    /// Optimise the window, reject outliers, marginalise if the window is full.
    void process();

    const KeyFrame& latest() const { return window_.back(); }
    const std::deque<KeyFrame>& window() const { return window_; }
    const Extrinsics& extrinsics() const { return ext_; }
    const EstimatorConfig& config() const { return cfg_; }

    std::unordered_set<int> takeRejected();
    /// Keyframes that left the window (final smoothed estimate): (t, body state).
    std::vector<std::pair<double, NavState>> takeFinalized();
    std::vector<Vec3> landmarkPositions() const;
    int activeLandmarks() const;
    double lastSolveMs() const { return last_solve_ms_; }
    double lastCost() const { return last_cost_; }

private:
    struct Block {
        int var;       // >= 0: keyframe window index (15 dof); < 0: landmark -(index+1) (1 dof)
        MatX J;
    };
    struct Lin {
        VecX r;
        std::vector<Block> blocks;
    };
    struct Prior {
        std::vector<int> kf_ids;
        std::vector<NavState> x0;
        MatX J;
        VecX r0;
        bool valid() const { return !kf_ids.empty(); }
    };

    int indexOf(int kf_id) const;
    bool usable(const Landmark& lm) const;
    void triangulate(Landmark& lm);
    Vec3 hostPoint(const Landmark& lm) const;     // landmark in world frame

    // Residuals (whitened).
    VecX imuResidual(const NavState& a, const NavState& b, const Preintegration& pre) const;
    Mat<15, 15> imuSqrtInfo(const Preintegration& pre) const;
    bool visualResidual(const NavState& host, const NavState& tgt, double inv_depth, const Vec2& xn_host,
                        const Observation& o, VecX& r) const;

    void linearizeImu(int j, std::vector<Lin>& out, double& cost, bool jac) const;
    void linearizeLandmark(const Landmark& lm, int lm_index, std::vector<Lin>& out, double& cost, bool jac) const;
    void linearizePrior(std::vector<Lin>& out, double& cost, bool jac) const;
    double robustify(VecX& r, std::vector<Block>* blocks) const;

    double evaluate();
    bool optimize();
    void rejectOutliers();
    void marginalizeOldest();
    void repropagate();

    EstimatorConfig cfg_;
    ImuNoise noise_;
    Extrinsics ext_;
    std::deque<KeyFrame> window_;
    std::map<int, Landmark> landmarks_;
    Prior prior_;
    int next_kf_id_ = 0;
    std::unordered_set<int> rejected_;
    std::vector<std::pair<double, NavState>> finalized_;
    std::vector<Landmark*> opt_landmarks_;
    double last_solve_ms_ = 0, last_cost_ = 0;
};

}  // namespace slamcore
