#pragma once

#include "slamcore/types.hpp"

#include <array>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace slamcore {

struct LoopConfig {
    // Keyframe features (ORB-style: FAST corners on a scale pyramid + oriented binary descriptors).
    int max_keypoints = 500;
    int pyramid_levels = 4;
    double pyramid_scale = 1.25;
    int fast_threshold = 7;           // low: corners are ranked by score, so weak scenes still get enough
    // Candidate selection (uses the current trajectory estimate, so drift must stay below these).
    double min_time_gap = 8.0;        // s between the two visits
    double max_distance = 2.0;        // m between camera centres
    double max_angle_deg = 45.0;      // between viewing directions
    double max_lookat_distance = 1.5; // m between the points the two cameras look at
    int candidates_per_keyframe = 4;
    double loop_spacing_s = 0.5;      // skip a pair if a verified loop links both moments within this
    // Geometric verification.
    int max_hamming = 64;             // of 256 bits
    double ratio = 0.8;
    double guided_radius_px = 40;     // guided matching around the predicted position (drift <~ 5 deg)
    int guided_max_hamming = 80;
    double guided_ratio = 0.9;
    double refine_radius_px = 12;     // second guided round with the RANSAC pose
    // Accept only loops consistent with the trajectory estimate within the drift expected after dt.
    double drift_m = 0.3, drift_m_per_s = 0.008;
    double drift_deg = 5.0, drift_deg_per_s = 0.05;
    int min_inliers = 20;
    double reproj_px = 3.0;
    // Pose graph (4 DoF: position + yaw; roll/pitch are observable from gravity).
    double odom_sigma_p = 0.003;      // m per keyframe step ...
    double odom_sigma_p_rel = 0.01;   // ... plus this fraction of the step length
    double odom_sigma_yaw = 0.002;    // rad per keyframe step
    double loop_sigma_p = 0.02;       // m
    double loop_sigma_yaw = 0.01;     // rad
    double loop_outlier_chi2 = 16.0;  // loop edges above this (whitened, 4 DoF) after optimisation are dropped
    int passes = 3;                   // detect -> optimise -> detect with the corrected trajectory ...
};

/// A keyframe as stored for loop closure: compact features with LiDAR depth, nothing else.
struct LoopKeyframe {
    double t = 0;
    int segment = 0;                  // map segment (a new one starts when the VLIO re-initialises)
    struct Keypoint {
        float x, y;                   // pixel in the tracking image
        float depth;                  // metric LiDAR depth, 0 = none
        uint8_t level;
        std::array<uint64_t, 4> desc;
    };
    std::vector<Keypoint> kps;
    double median_depth = 2.0;
};

struct LoopEdge {
    int a, b;                         // keyframe indices, a older than b
    Pose T_ab;                        // camera b -> camera a
    int inliers = 0;
};

struct LoopStats {
    int keyframes = 0;
    int candidates = 0;               // keyframe pairs tested
    int verified = 0;                 // geometrically verified
    int accepted = 0;                 // kept after the pose-graph outlier check
    double max_correction_m = 0;      // largest keyframe position change
    double max_correction_deg = 0;    // largest keyframe yaw change
    double ms = 0;
    // Why candidates failed verification: too few matches, no 3D-3D consensus, too few reprojection
    // inliers, inconsistent with the trajectory estimate.
    int rejected_matches = 0, rejected_ransac = 0, rejected_inliers = 0, rejected_prior = 0;
};

/// Place recognition + geometric verification + pose-graph optimisation over a whole recording.
class LoopCloser {
public:
    explicit LoopCloser(const LoopConfig& cfg = {});

    /// Extract and keep the features of a keyframe (called once per keyframe, in time order).
    void addKeyframe(double t, int segment, const GrayImage& image, const PinholeCamera& cam, const DepthMap* depth);

    /// poses[i]: final VLIO camera->world pose of keyframes()[i] (world z-up).
    /// Returns loop-closed poses of the same keyframes.
    std::vector<Pose> optimize(const std::vector<Pose>& poses, LoopStats* stats);

    const std::vector<LoopKeyframe>& keyframes() const { return kfs_; }
    const std::vector<LoopEdge>& loops() const { return loops_; }

    /// Verify one keyframe pair (exposed for tests). Returns false if not a valid loop.
    bool verify(int a, int b, const Pose& prior_ab, LoopEdge& out, LoopStats* stats = nullptr) const;

private:
    void extract(const GrayImage& img, const DepthMap* depth, LoopKeyframe& kf);
    std::vector<std::pair<int, int>> candidates(const std::vector<Pose>& poses,
                                                const std::unordered_set<uint64_t>& tried) const;
    std::vector<Pose> solve(const std::vector<Pose>& poses, std::vector<LoopEdge>& loops, int& accepted) const;

    LoopConfig cfg_;
    PinholeCamera cam_;
    std::vector<LoopKeyframe> kfs_;
    std::vector<LoopEdge> loops_;
    // Scratch (reused between keyframes).
    std::vector<GrayImage> pyr_, blur_;
    std::vector<float> score_;
};

}  // namespace slamcore
