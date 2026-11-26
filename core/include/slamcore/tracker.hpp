#pragma once

#include "slamcore/types.hpp"

#include <unordered_set>
#include <vector>

namespace slamcore {

struct TrackerConfig {
    int max_features = 150;
    int min_distance = 18;          // px between features
    int grid_cols = 8, grid_rows = 6;
    double min_quality = 0.005;     // Shi-Tomasi response relative to the image maximum
    int pyramid_levels = 3;
    int window_half = 7;            // 15x15 LK window
    int max_iterations = 30;
    double fb_threshold = 1.0;      // forward-backward error [px]
    int max_age = 0;                // drop tracks older than this many frames (0 = unlimited)
};

struct Feature {
    int id;
    Vec2 px;        // pixel position in the current image
    int age;        // number of frames tracked
};

/// Sparse feature tracker: Shi-Tomasi corners + pyramidal Lucas-Kanade with gyro-predicted initial
/// flow and a forward-backward consistency check (KLT front end, as in VINS-Mono / OKVIS).
class FeatureTracker {
public:
    explicit FeatureTracker(const TrackerConfig& cfg = {});

    /// Track into `image`. R_cur_prev rotates rays of the previous camera into the current one
    /// (from the gyro); it only seeds the search, any value works.
    const std::vector<Feature>& track(const GrayImage& image, const PinholeCamera& cam, const Mat3& R_cur_prev);

    const std::vector<Feature>& features() const { return features_; }
    /// Stop tracking features rejected by the back end.
    void remove(const std::unordered_set<int>& ids);

private:
    using Pyramid = std::vector<GrayImage>;
    /// Builds into `pyr`, reusing its storage (no allocation once sizes are stable).
    static void buildPyramid(const GrayImage& img, int levels, Pyramid& pyr);
    bool trackPoint(const Pyramid& from, const Pyramid& to, const Vec2& p, Vec2& q) const;
    void detect(const GrayImage& img);

    TrackerConfig cfg_;
    Pyramid prev_, cur_;                  // swapped every frame, storage reused
    std::vector<Feature> features_, scratch_;
    int next_id_ = 0;
    // Detection scratch buffers, allocated once per image size.
    std::vector<float> gxx_, gxy_, gyy_, tmp_, resp_;
    std::vector<uint8_t> occupied_;
};

}  // namespace slamcore
