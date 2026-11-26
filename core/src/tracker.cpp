#include "slamcore/tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace slamcore {
namespace {

/// Bilinear sample of an 8-bit image.
inline float sample(const GrayImage& im, float x, float y) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float ax = x - x0, ay = y - y0;
    const int xc = std::max(x0, 0), yc = std::max(y0, 0);
    const int x1 = std::min(x0 + 1, im.width - 1), y1 = std::min(y0 + 1, im.height - 1);
    const uint8_t* r0 = im.data.data() + static_cast<size_t>(yc) * im.width;
    const uint8_t* r1 = im.data.data() + static_cast<size_t>(y1) * im.width;
    return (1 - ay) * ((1 - ax) * r0[xc] + ax * r0[x1]) + ay * ((1 - ax) * r1[xc] + ax * r1[x1]);
}

constexpr int kMaxWindow = 31 * 31;   // supports window_half <= 15

/// Box sum over a (2r+1)^2 window, separable running sums: O(1) per pixel. `tmp` is scratch.
void boxSum(const std::vector<float>& in, std::vector<float>& out, std::vector<float>& tmp, int W, int H, int r) {
    for (int y = 0; y < H; ++y) {                       // horizontal pass
        const float* src = in.data() + static_cast<size_t>(y) * W;
        float* dst = tmp.data() + static_cast<size_t>(y) * W;
        float acc = 0;
        for (int x = 0; x <= std::min(r, W - 1); ++x) acc += src[x];
        for (int x = 0; x < W; ++x) {
            dst[x] = acc;
            if (x + r + 1 < W) acc += src[x + r + 1];
            if (x - r >= 0) acc -= src[x - r];
        }
    }
    for (int x = 0; x < W; ++x) {                       // vertical pass
        float acc = 0;
        for (int y = 0; y <= std::min(r, H - 1); ++y) acc += tmp[static_cast<size_t>(y) * W + x];
        for (int y = 0; y < H; ++y) {
            out[static_cast<size_t>(y) * W + x] = acc;
            if (y + r + 1 < H) acc += tmp[static_cast<size_t>(y + r + 1) * W + x];
            if (y - r >= 0) acc -= tmp[static_cast<size_t>(y - r) * W + x];
        }
    }
}

}  // namespace

FeatureTracker::FeatureTracker(const TrackerConfig& cfg) : cfg_(cfg) {}

void FeatureTracker::buildPyramid(const GrayImage& img, int levels, Pyramid& pyr) {
    pyr.resize(levels);
    pyr[0].width = img.width;
    pyr[0].height = img.height;
    pyr[0].data.assign(img.data.begin(), img.data.end());       // reuses capacity
    for (int l = 1; l < levels; ++l) {
        const GrayImage& s = pyr[l - 1];
        GrayImage& d = pyr[l];
        d.width = s.width / 2;
        d.height = s.height / 2;
        d.data.resize(static_cast<size_t>(d.width) * d.height);
        for (int y = 0; y < d.height; ++y) {
            const uint8_t* a = s.data.data() + static_cast<size_t>(2 * y) * s.width;
            const uint8_t* b = a + s.width;
            uint8_t* o = d.data.data() + static_cast<size_t>(y) * d.width;
            for (int x = 0; x < d.width; ++x)
                o[x] = static_cast<uint8_t>((a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1] + 2) / 4);
        }
    }
}

/// Pyramidal inverse-compositional Lucas-Kanade (translation only). `q` holds the initial guess
/// at level 0 and receives the result. All scratch lives on the stack.
bool FeatureTracker::trackPoint(const Pyramid& from, const Pyramid& to, const Vec2& p, Vec2& q) const {
    const int L = static_cast<int>(from.size());
    const int hw = std::min(cfg_.window_half, 15);
    std::array<float, kMaxWindow> T, gx, gy;
    float dx = static_cast<float>((q.x() - p.x()) / std::pow(2.0, L - 1));
    float dy = static_cast<float>((q.y() - p.y()) / std::pow(2.0, L - 1));
    for (int l = L - 1; l >= 0; --l) {
        const GrayImage& I = from[l];
        const GrayImage& J = to[l];
        const float s = 1.0f / static_cast<float>(1 << l);
        const float px = static_cast<float>(p.x()) * s, py = static_cast<float>(p.y()) * s;
        if (px < hw + 1 || py < hw + 1 || px > I.width - hw - 2 || py > I.height - hw - 2) return false;

        float h00 = 0, h01 = 0, h11 = 0;
        int k = 0;
        for (int v = -hw; v <= hw; ++v)
            for (int u = -hw; u <= hw; ++u, ++k) {
                const float x = px + u, y = py + v;
                T[k] = sample(I, x, y);
                gx[k] = 0.5f * (sample(I, x + 1, y) - sample(I, x - 1, y));
                gy[k] = 0.5f * (sample(I, x, y + 1) - sample(I, x, y - 1));
                h00 += gx[k] * gx[k];
                h01 += gx[k] * gy[k];
                h11 += gy[k] * gy[k];
            }
        const float det = h00 * h11 - h01 * h01;
        if (det < 1e-6f) return false;
        const float i00 = h11 / det, i01 = -h01 / det, i11 = h00 / det;

        for (int it = 0; it < cfg_.max_iterations; ++it) {
            const float cx = px + dx, cy = py + dy;
            if (cx < hw + 1 || cy < hw + 1 || cx > J.width - hw - 2 || cy > J.height - hw - 2) return false;
            float bx = 0, by = 0;
            k = 0;
            for (int v = -hw; v <= hw; ++v)
                for (int u = -hw; u <= hw; ++u, ++k) {
                    const float e = sample(J, cx + u, cy + v) - T[k];
                    bx += gx[k] * e;
                    by += gy[k] * e;
                }
            const float ddx = i00 * bx + i01 * by, ddy = i01 * bx + i11 * by;
            dx -= ddx;
            dy -= ddy;
            if (ddx * ddx + ddy * ddy < 1e-4f) break;
        }
        if (l > 0) {
            dx *= 2;
            dy *= 2;
        }
    }
    q = p + Vec2(dx, dy);
    return std::isfinite(q.x()) && std::isfinite(q.y());
}

const std::vector<Feature>& FeatureTracker::track(const GrayImage& image, const PinholeCamera& cam,
                                                  const Mat3& R_cur_prev) {
    buildPyramid(image, cfg_.pyramid_levels, cur_);
    if (!prev_.empty() && !features_.empty()) {
        scratch_.clear();
        for (const Feature& f : features_) {
            if (cfg_.max_age > 0 && f.age >= cfg_.max_age) continue;
            // Gyro-predicted position (pure rotation homography K R K^-1).
            const Vec2 n = cam.normalize(f.px);
            const Vec3 r = R_cur_prev * Vec3(n.x(), n.y(), 1.0);
            Vec2 q = r.z() > 0.1 ? cam.project(r) : f.px;
            if (!trackPoint(prev_, cur_, f.px, q)) continue;
            Vec2 back = f.px;
            if (!trackPoint(cur_, prev_, q, back)) continue;
            if ((back - f.px).norm() > cfg_.fb_threshold) continue;
            if (q.x() < 2 || q.y() < 2 || q.x() > image.width - 3 || q.y() > image.height - 3) continue;
            scratch_.push_back({f.id, q, f.age + 1});
        }
        features_.swap(scratch_);
    }
    detect(image);
    std::swap(prev_, cur_);
    return features_;
}

void FeatureTracker::remove(const std::unordered_set<int>& ids) {
    if (ids.empty()) return;
    features_.erase(std::remove_if(features_.begin(), features_.end(),
                                   [&](const Feature& f) { return ids.count(f.id) > 0; }),
                    features_.end());
}

void FeatureTracker::detect(const GrayImage& img) {
    if (static_cast<int>(features_.size()) >= cfg_.max_features) return;
    const int W = img.width, H = img.height;
    const size_t N = static_cast<size_t>(W) * H;
    if (gxx_.size() != N) {
        for (auto* v : {&gxx_, &gxy_, &gyy_, &tmp_, &resp_}) v->assign(N, 0.f);
        occupied_.assign(N, 0);
    }
    // Gradient products.
    for (int y = 1; y < H - 1; ++y) {
        const uint8_t* r = img.data.data() + static_cast<size_t>(y) * W;
        for (int x = 1; x < W - 1; ++x) {
            const float gx = 0.5f * (r[x + 1] - r[x - 1]);
            const float gy = 0.5f * (r[x + W] - r[x - W]);
            const size_t i = static_cast<size_t>(y) * W + x;
            gxx_[i] = gx * gx;
            gxy_[i] = gx * gy;
            gyy_[i] = gy * gy;
        }
    }
    // 5x5 structure tensor via O(1) box sums, then Shi-Tomasi minimum eigenvalue (in place).
    const int r = 2, border = 10;
    boxSum(gxx_, resp_, tmp_, W, H, r);
    boxSum(gxy_, gxx_, tmp_, W, H, r);         // gxx_ now holds sum(gxy) (gxx sum lives in resp_)
    boxSum(gyy_, gxy_, tmp_, W, H, r);         // gxy_ now holds sum(gyy)
    float maxr = 0;
    for (int y = border; y < H - border; ++y)
        for (int x = border; x < W - border; ++x) {
            const size_t i = static_cast<size_t>(y) * W + x;
            const float a = resp_[i], b = gxx_[i], c = gxy_[i];
            const float l = 0.5f * (a + c - std::sqrt((a - c) * (a - c) + 4 * b * b));
            gyy_[i] = l;                        // gyy_ now holds the response
            maxr = std::max(maxr, l);
        }
    const std::vector<float>& resp = gyy_;
    if (maxr <= 0) return;

    std::fill(occupied_.begin(), occupied_.end(), 0);
    const int md = cfg_.min_distance;
    auto occupy = [&](const Vec2& p) {
        const int cx = static_cast<int>(p.x()), cy = static_cast<int>(p.y());
        for (int y = std::max(0, cy - md); y <= std::min(H - 1, cy + md); ++y)
            for (int x = std::max(0, cx - md); x <= std::min(W - 1, cx + md); ++x)
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= md * md) occupied_[static_cast<size_t>(y) * W + x] = 1;
    };
    for (const Feature& f : features_) occupy(f.px);

    struct Cand { float r; int x, y; };
    std::vector<Cand> cands;
    cands.reserve(2048);
    const float thr = static_cast<float>(cfg_.min_quality) * maxr;
    for (int y = border; y < H - border; ++y)
        for (int x = border; x < W - border; ++x) {
            const size_t i = static_cast<size_t>(y) * W + x;
            const float v = resp[i];
            if (v < thr || occupied_[i]) continue;
            if (v < resp[i - 1] || v < resp[i + 1] || v < resp[i - W] || v < resp[i + W] || v < resp[i - W - 1] ||
                v < resp[i - W + 1] || v < resp[i + W - 1] || v < resp[i + W + 1])
                continue;
            cands.push_back({v, x, y});
        }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.r > b.r; });

    const int cells = cfg_.grid_cols * cfg_.grid_rows;
    const int quota = std::max(1, cfg_.max_features / cells + 1);
    std::array<int, 256> perCell{};
    auto cellOf = [&](double x, double y) {
        return std::min(cfg_.grid_cols - 1, static_cast<int>(x * cfg_.grid_cols / W)) +
               cfg_.grid_cols * std::min(cfg_.grid_rows - 1, static_cast<int>(y * cfg_.grid_rows / H));
    };
    for (const Feature& f : features_) perCell[std::min(cellOf(f.px.x(), f.px.y()), 255)]++;
    for (const Cand& c : cands) {
        if (static_cast<int>(features_.size()) >= cfg_.max_features) break;
        if (occupied_[static_cast<size_t>(c.y) * W + c.x]) continue;
        const int cell = std::min(cellOf(c.x, c.y), 255);
        if (perCell[cell] >= quota) continue;
        perCell[cell]++;
        const Vec2 p(c.x, c.y);
        features_.push_back({next_id_++, p, 1});
        occupy(p);
    }
}

}  // namespace slamcore
