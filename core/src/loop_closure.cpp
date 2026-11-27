// Loop closure for the post-scan pass:
//  1. every keyframe keeps ~300 ORB-style features (FAST on a 4-level pyramid, intensity-centroid
//     orientation, 256-bit rotated binary descriptor) with LiDAR depth at each corner,
//  2. after the scan, revisit candidates come from the trajectory estimate (close camera centres,
//     similar viewing direction, overlapping look-at points, > min_time_gap apart),
//  3. each candidate pair is matched (Hamming, ratio + mutual test) and verified geometrically:
//     3-point RANSAC on LiDAR 3D-3D matches, then Gauss-Newton on two-sided reprojection error,
//  4. a 4-DoF pose graph (position + yaw per keyframe; roll/pitch are fixed by gravity) combines the
//     VLIO odometry with the verified loops; loops inconsistent with the rest are dropped, and the
//     detect -> optimise cycle repeats with the corrected trajectory to find loops drift had hidden.
#include "slamcore/loop_closure.hpp"

#include "slamcore/so3.hpp"

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace slamcore {

namespace {

constexpr int kBorder = 20;   // rotated descriptor pattern reaches 13 * sqrt(2) ~ 18.4 px
constexpr int kPatchRadius = 15;

// FAST circle (radius 3, 16 pixels), clockwise from the top.
constexpr int kCircle[16][2] = {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
                                {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

/// 256 point pairs, Gaussian around the keypoint (BRIEF G II, sigma = patch/5), from a fixed-seed
/// generator that gives the same pattern on every platform (std:: distributions do not).
struct Pattern {
    int8_t p[256][4];
    Pattern() {
        uint64_t s = 0x9E3779B97F4A7C15ull;
        auto uniform = [&]() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            return ((s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        };
        auto gauss = [&]() { return std::sqrt(-2 * std::log(uniform())) * std::cos(2 * M_PI * uniform()); };
        for (auto& q : p)
            for (int k = 0; k < 4; ++k)
                q[k] = static_cast<int8_t>(std::clamp<long>(std::lround(gauss() * 31.0 / 5.0), -13, 13));
    }
};
const Pattern& pattern() {
    static const Pattern pat;
    return pat;
}

/// Half-widths of the circular orientation patch per row.
const std::array<int, kPatchRadius + 1>& umax() {
    static const auto u = [] {
        std::array<int, kPatchRadius + 1> a{};
        for (int v = 0; v <= kPatchRadius; ++v)
            a[v] = static_cast<int>(std::floor(std::sqrt(double(kPatchRadius * kPatchRadius - v * v)) + 1e-9));
        return a;
    }();
    return u;
}

void resizeBilinear(const GrayImage& src, int w, int h, GrayImage& dst) {
    dst.width = w;
    dst.height = h;
    dst.data.resize(static_cast<size_t>(w) * h);
    const double sx = double(src.width) / w, sy = double(src.height) / h;
    for (int y = 0; y < h; ++y) {
        const double fy = std::min((y + 0.5) * sy - 0.5, src.height - 1.001);
        const int y0 = std::max(0, static_cast<int>(fy));
        const double wy = std::max(0.0, fy - y0);
        for (int x = 0; x < w; ++x) {
            const double fx = std::min((x + 0.5) * sx - 0.5, src.width - 1.001);
            const int x0 = std::max(0, static_cast<int>(fx));
            const double wx = std::max(0.0, fx - x0);
            const double v = (1 - wy) * ((1 - wx) * src.at(x0, y0) + wx * src.at(x0 + 1, y0)) +
                             wy * ((1 - wx) * src.at(x0, y0 + 1) + wx * src.at(x0 + 1, y0 + 1));
            dst.data[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(std::lround(v));
        }
    }
}

/// Separable [1 4 6 4 1] / 16 blur (the descriptor compares smoothed intensities).
void blur5(const GrayImage& src, GrayImage& dst, std::vector<uint16_t>& tmp) {
    const int W = src.width, H = src.height;
    dst.width = W;
    dst.height = H;
    dst.data.resize(src.data.size());
    tmp.resize(src.data.size());
    auto clampi = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint8_t* r = &src.data[static_cast<size_t>(y) * W];
            tmp[static_cast<size_t>(y) * W + x] = static_cast<uint16_t>(
                r[clampi(x - 2, W - 1)] + 4 * r[clampi(x - 1, W - 1)] + 6 * r[x] + 4 * r[clampi(x + 1, W - 1)] +
                r[clampi(x + 2, W - 1)]);
        }
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            auto at = [&](int yy) { return tmp[static_cast<size_t>(clampi(yy, H - 1)) * W + x]; };
            const int v = at(y - 2) + 4 * at(y - 1) + 6 * at(y) + 4 * at(y + 1) + at(y + 2);
            dst.data[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>((v + 128) >> 8);
        }
}

inline int hamming(const std::array<uint64_t, 4>& a, const std::array<uint64_t, 4>& b) {
    return __builtin_popcountll(a[0] ^ b[0]) + __builtin_popcountll(a[1] ^ b[1]) +
           __builtin_popcountll(a[2] ^ b[2]) + __builtin_popcountll(a[3] ^ b[3]);
}

inline Mat3 rotZ(double a) {
    const double c = std::cos(a), s = std::sin(a);
    Mat3 R;
    R << c, -s, 0, s, c, 0, 0, 0, 1;
    return R;
}

/// Yaw (about world z) of a rotation that is close to a pure z rotation.
inline double yawOf(const Mat3& A) { return std::atan2(A(1, 0) - A(0, 1), A(0, 0) + A(1, 1)); }

inline double wrap(double a) {
    while (a > M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
}

Pose inverse(const Pose& T) { return {T.R.transpose(), -T.R.transpose() * T.p}; }
Pose compose(const Pose& A, const Pose& B) { return {A.R * B.R, A.p + A.R * B.p}; }

/// Least-squares rigid transform (Kabsch): A ~= R * B + t.
bool kabsch(const std::vector<Vec3>& A, const std::vector<Vec3>& B, Mat3& R, Vec3& t) {
    const size_t n = A.size();
    if (n < 3) return false;
    Vec3 ca = Vec3::Zero(), cb = Vec3::Zero();
    for (size_t i = 0; i < n; ++i) { ca += A[i]; cb += B[i]; }
    ca /= double(n);
    cb /= double(n);
    Mat3 H = Mat3::Zero();
    for (size_t i = 0; i < n; ++i) H += (B[i] - cb) * (A[i] - ca).transpose();
    Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()(1) < 1e-9) return false;   // collinear
    Mat3 D = Mat3::Identity();
    if ((svd.matrixV() * svd.matrixU().transpose()).determinant() < 0) D(2, 2) = -1;
    R = svd.matrixV() * D * svd.matrixU().transpose();
    t = ca - R * cb;
    return true;
}

}  // namespace

LoopCloser::LoopCloser(const LoopConfig& cfg) : cfg_(cfg) { (void)pattern(); }

void LoopCloser::addKeyframe(double t, int segment, const GrayImage& image, const PinholeCamera& cam,
                             const DepthMap* depth) {
    cam_ = cam;
    LoopKeyframe kf;
    kf.t = t;
    kf.segment = segment;
    extract(image, depth, kf);
    kfs_.push_back(std::move(kf));
}

void LoopCloser::extract(const GrayImage& img, const DepthMap* depth, LoopKeyframe& kf) {
    const int L = std::max(1, cfg_.pyramid_levels);
    pyr_.resize(L);
    blur_.resize(L);
    std::vector<uint16_t> tmp;
    std::vector<double> scale(L, 1.0);
    pyr_[0] = img;
    for (int l = 1; l < L; ++l) {
        scale[l] = scale[l - 1] * cfg_.pyramid_scale;
        resizeBilinear(img, static_cast<int>(std::lround(img.width / scale[l])),
                       static_cast<int>(std::lround(img.height / scale[l])), pyr_[l]);
    }
    double area = 0;
    for (int l = 0; l < L; ++l) area += double(pyr_[l].width) * pyr_[l].height;

    const Pattern& pat = pattern();
    const auto& um = umax();
    const int thr = cfg_.fast_threshold;
    struct Cand { int x, y; float s; };
    std::vector<Cand> cands;

    for (int l = 0; l < L; ++l) {
        const GrayImage& I = pyr_[l];
        const int W = I.width, H = I.height;
        if (W <= 2 * kBorder || H <= 2 * kBorder) continue;
        blur5(I, blur_[l], tmp);
        const GrayImage& B = blur_[l];

        // FAST-9 corners with a sum-of-differences score.
        score_.assign(static_cast<size_t>(W) * H, 0.f);
        int off[16];
        for (int k = 0; k < 16; ++k) off[k] = kCircle[k][1] * W + kCircle[k][0];
        for (int y = kBorder; y < H - kBorder; ++y) {
            const uint8_t* row = &I.data[static_cast<size_t>(y) * W];
            for (int x = kBorder; x < W - kBorder; ++x) {
                const uint8_t* p = row + x;
                const int c = *p, hi = c + thr, lo = c - thr;
                const int v0 = p[off[0]], v4 = p[off[4]], v8 = p[off[8]], v12 = p[off[12]];
                const int nb = (v0 > hi) + (v4 > hi) + (v8 > hi) + (v12 > hi);
                const int nd = (v0 < lo) + (v4 < lo) + (v8 < lo) + (v12 < lo);
                if (nb < 2 && nd < 2) continue;
                int v[16];
                for (int k = 0; k < 16; ++k) v[k] = p[off[k]];
                for (int side = 0; side < 2; ++side) {
                    int run = 0, best = 0;
                    for (int k = 0; k < 32; ++k) {
                        const bool on = side == 0 ? v[k & 15] > hi : v[k & 15] < lo;
                        run = on ? run + 1 : 0;
                        best = std::max(best, run);
                    }
                    if (best >= 9) {
                        float s = 0;
                        for (int k = 0; k < 16; ++k) s += std::max(0, side == 0 ? v[k] - hi : lo - v[k]);
                        score_[static_cast<size_t>(y) * W + x] = std::max(score_[static_cast<size_t>(y) * W + x], s);
                    }
                }
            }
        }
        cands.clear();
        for (int y = kBorder; y < H - kBorder; ++y)
            for (int x = kBorder; x < W - kBorder; ++x) {
                const float s = score_[static_cast<size_t>(y) * W + x];
                if (s <= 0) continue;
                bool peak = true;
                for (int dy = -1; dy <= 1 && peak; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        if ((dx || dy) && score_[static_cast<size_t>(y + dy) * W + x + dx] > s) { peak = false; break; }
                if (peak) cands.push_back({x, y, s});
            }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.s > b.s; });

        // Spread over the image: a cap per 32 px cell, budget proportional to the level area.
        const int budget = static_cast<int>(std::lround(cfg_.max_keypoints * W * H / area));
        const int cell = 32, gc = (W + cell - 1) / cell, gr = (H + cell - 1) / cell;
        const int cap = std::max(1, (2 * budget + gc * gr - 1) / (gc * gr));
        std::vector<uint8_t> used(static_cast<size_t>(gc) * gr, 0);
        int taken = 0;
        for (const Cand& c : cands) {
            if (taken >= budget) break;
            uint8_t& u = used[static_cast<size_t>(c.y / cell) * gc + c.x / cell];
            if (u >= cap) continue;
            ++u;
            ++taken;

            // Orientation: intensity centroid of the circular patch.
            double m10 = 0, m01 = 0;
            for (int dv = -kPatchRadius; dv <= kPatchRadius; ++dv) {
                const int w = um[std::abs(dv)];
                const uint8_t* r = &I.data[static_cast<size_t>(c.y + dv) * W + c.x];
                for (int du = -w; du <= w; ++du) {
                    m10 += du * r[du];
                    m01 += dv * r[du];
                }
            }
            const double ang = std::atan2(m01, m10), ca = std::cos(ang), sa = std::sin(ang);

            LoopKeyframe::Keypoint kp{};
            kp.x = static_cast<float>((c.x + 0.5) * scale[l] - 0.5);
            kp.y = static_cast<float>((c.y + 0.5) * scale[l] - 0.5);
            kp.level = static_cast<uint8_t>(l);
            const uint8_t* centre = &B.data[static_cast<size_t>(c.y) * W + c.x];
            for (int k = 0; k < 256; ++k) {
                const int8_t* q = pat.p[k];
                const int x1 = static_cast<int>(std::lround(ca * q[0] - sa * q[1]));
                const int y1 = static_cast<int>(std::lround(sa * q[0] + ca * q[1]));
                const int x2 = static_cast<int>(std::lround(ca * q[2] - sa * q[3]));
                const int y2 = static_cast<int>(std::lround(sa * q[2] + ca * q[3]));
                if (centre[y1 * W + x1] < centre[y2 * W + x2]) kp.desc[k >> 6] |= 1ull << (k & 63);
            }
            kf.kps.push_back(kp);
        }
    }

    // LiDAR depth at every corner: confident, and locally consistent (corners sit on depth edges).
    if (depth && depth->valid()) {
        const int W = depth->width, H = depth->height;
        for (auto& kp : kf.kps) {
            const int u = static_cast<int>(std::lround((kp.x + 0.5) * W / img.width - 0.5));
            const int v = static_cast<int>(std::lround((kp.y + 0.5) * H / img.height - 0.5));
            if (u < 1 || v < 1 || u >= W - 1 || v >= H - 1) continue;
            const size_t i = static_cast<size_t>(v) * W + u;
            if (!depth->confidence.empty() && depth->confidence[i] < 1) continue;
            const float d = depth->depth[i];
            if (!(d > 0.15f && d < 6.0f)) continue;
            float lo = d, hi = d;
            for (int dv = -1; dv <= 1; ++dv)
                for (int du = -1; du <= 1; ++du) {
                    const float x = depth->depth[i + dv * W + du];
                    lo = std::min(lo, x);
                    hi = std::max(hi, x);
                }
            if (lo > 0 && hi - lo < 0.05f * d) kp.depth = d;
        }
        std::vector<float> ds;
        for (int v = H / 4; v < 3 * H / 4; v += 4)
            for (int u = W / 4; u < 3 * W / 4; u += 4) {
                const float d = depth->depth[static_cast<size_t>(v) * W + u];
                if (d > 0.15f && d < 8.0f) ds.push_back(d);
            }
        if (!ds.empty()) {
            std::nth_element(ds.begin(), ds.begin() + ds.size() / 2, ds.end());
            kf.median_depth = ds[ds.size() / 2];
        }
    }
    kf.kps.shrink_to_fit();
}

bool LoopCloser::verify(int a, int b, const Pose& prior_ab, LoopEdge& out, LoopStats* stats) const {
    auto reject = [&](int LoopStats::*field) {
        if (stats) ++(stats->*field);
        return false;
    };
    const LoopKeyframe& A = kfs_[a];
    const LoopKeyframe& B = kfs_[b];
    const int na = static_cast<int>(A.kps.size()), nb = static_cast<int>(B.kps.size());
    if (na < cfg_.min_inliers || nb < cfg_.min_inliers) return reject(&LoopStats::rejected_matches);

    // Brute-force Hamming matching, ratio + mutual test.
    std::vector<int> bestA(na, -1), bestAd(na, 1 << 20);
    struct Match { int ia, ib; };
    std::vector<Match> matches;
    std::vector<int> bestB(nb, -1);
    for (int j = 0; j < nb; ++j) {
        int d1 = 1 << 20, d2 = 1 << 20, i1 = -1;
        for (int i = 0; i < na; ++i) {
            const int d = hamming(A.kps[i].desc, B.kps[j].desc);
            if (d < d1) { d2 = d1; d1 = d; i1 = i; }
            else if (d < d2) d2 = d;
            if (d < bestAd[i]) { bestAd[i] = d; bestA[i] = j; }
        }
        if (d1 <= cfg_.max_hamming && d1 < cfg_.ratio * d2) bestB[j] = i1;
    }
    for (int j = 0; j < nb; ++j)
        if (bestB[j] >= 0 && bestA[bestB[j]] == j) matches.push_back({bestB[j], j});

    auto ray = [&](const LoopKeyframe::Keypoint& k) {
        const Vec2 xn = cam_.normalize(Vec2(k.x, k.y));
        return Vec3(xn.x(), xn.y(), 1.0);
    };

    // Guided matching: a's LiDAR corners projected into b with a pose guess land near their match.
    // Finds many more matches in weakly textured rooms. The result is the union with `base`
    // (one match per keypoint on either side, guided matches win).
    auto guidedMatch = [&](const Pose& T_ab, double radius, const std::vector<Match>& base) {
        const double r2 = radius * radius;
        std::vector<int> ownerB(nb, -1), ownerD(nb, 1 << 20);
        for (int i = 0; i < na; ++i) {
            const auto& ka = A.kps[i];
            if (ka.depth <= 0) continue;
            const Vec3 Xb = T_ab.R.transpose() * (ray(ka) * ka.depth - T_ab.p);
            if (Xb.z() < 0.1) continue;
            const Vec2 uv = cam_.project(Xb);
            int d1 = 1 << 20, d2 = 1 << 20, j1 = -1;
            for (int j = 0; j < nb; ++j) {
                const auto& kb = B.kps[j];
                if (std::abs(int(kb.level) - int(ka.level)) > 1) continue;
                const double dx = kb.x - uv.x(), dy = kb.y - uv.y();
                if (dx * dx + dy * dy > r2) continue;
                const int d = hamming(ka.desc, kb.desc);
                if (d < d1) { d2 = d1; d1 = d; j1 = j; }
                else if (d < d2) d2 = d;
            }
            if (j1 >= 0 && d1 <= cfg_.guided_max_hamming && d1 < cfg_.guided_ratio * d2 && d1 < ownerD[j1]) {
                ownerB[j1] = i;
                ownerD[j1] = d1;
            }
        }
        std::vector<Match> out;
        for (int j = 0; j < nb; ++j)
            if (ownerB[j] >= 0) out.push_back({ownerB[j], j});
        std::vector<char> usedA(na, 0), usedB(nb, 0);
        for (const Match& m : out) { usedA[m.ia] = 1; usedB[m.ib] = 1; }
        for (const Match& m : base)
            if (!usedA[m.ia] && !usedB[m.ib]) out.push_back(m);
        return out;
    };
    const std::vector<Match> appearance = matches;
    matches = guidedMatch(prior_ab, cfg_.guided_radius_px, appearance);
    if (static_cast<int>(matches.size()) < cfg_.min_inliers) return reject(&LoopStats::rejected_matches);

    // 3-point RANSAC on matches with LiDAR depth on both sides: X_a = R X_b + t.
    std::vector<Vec3> Xa, Xb;
    for (const Match& m : matches) {
        const auto& ka = A.kps[m.ia];
        const auto& kb = B.kps[m.ib];
        if (ka.depth > 0 && kb.depth > 0) {
            Xa.push_back(ray(ka) * ka.depth);
            Xb.push_back(ray(kb) * kb.depth);
        }
    }
    if (Xa.size() < 8) return reject(&LoopStats::rejected_ransac);
    const int n3 = static_cast<int>(Xa.size());
    uint64_t s = 0x2545F4914F6CDD1Dull ^ (uint64_t(a) << 32) ^ uint64_t(b);
    auto rnd = [&](int n) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<int>(s % uint64_t(n));
    };
    Mat3 Rbest = Mat3::Identity();
    Vec3 tbest = Vec3::Zero();
    int bestCount = 0;
    std::vector<Vec3> sa(3), sb(3);
    for (int it = 0; it < 200; ++it) {
        const int i0 = rnd(n3), i1 = rnd(n3), i2 = rnd(n3);
        if (i0 == i1 || i0 == i2 || i1 == i2) continue;
        sa = {Xa[i0], Xa[i1], Xa[i2]};
        sb = {Xb[i0], Xb[i1], Xb[i2]};
        Mat3 R;
        Vec3 t;
        if (!kabsch(sa, sb, R, t)) continue;
        int count = 0;
        for (int i = 0; i < n3; ++i)
            if ((Xa[i] - (R * Xb[i] + t)).norm() < 0.05 + 0.02 * Xa[i].z()) ++count;
        if (count > bestCount) { bestCount = count; Rbest = R; tbest = t; }
    }
    if (bestCount < 8) return reject(&LoopStats::rejected_ransac);
    {   // re-fit on all 3D inliers
        std::vector<Vec3> ia, ib;
        for (int i = 0; i < n3; ++i)
            if ((Xa[i] - (Rbest * Xb[i] + tbest)).norm() < 0.05 + 0.02 * Xa[i].z()) {
                ia.push_back(Xa[i]);
                ib.push_back(Xb[i]);
            }
        kabsch(ia, ib, Rbest, tbest);
    }
    // Second round of guided matching with the RANSAC pose (tight window): recovers matches the
    // drifted trajectory estimate had put too far away.
    matches = guidedMatch(Pose{Rbest, tbest}, cfg_.refine_radius_px, appearance);

    // Gauss-Newton on two-sided reprojection error (pixels) over every match with depth on either side.
    const double f = cam_.fx;
    Mat3 R = Rbest;
    Vec3 t = tbest;
    auto residuals = [&](const Mat3& Rr, const Vec3& tt, int& inl, Mat<6, 6>* H, Vec<6>* g) {
        double cost = 0;
        inl = 0;
        for (const Match& m : matches) {
            const auto& ka = A.kps[m.ia];
            const auto& kb = B.kps[m.ib];
            const Vec3 ra = ray(ka), rb = ray(kb);
            bool ok = false;
            for (int side = 0; side < 2; ++side) {
                Vec3 X;
                Mat<3, 6> dX;
                Vec2 obs;
                if (side == 0) {            // b's LiDAR point projected into a
                    if (kb.depth <= 0) continue;
                    const Vec3 Pb = rb * kb.depth;
                    X = Rr * Pb + tt;
                    dX << -Rr * so3::hat(Pb), Mat3::Identity();
                    obs = ra.head<2>();
                } else {                    // a's LiDAR point projected into b
                    if (ka.depth <= 0) continue;
                    const Vec3 Pa = ra * ka.depth;
                    X = Rr.transpose() * (Pa - tt);
                    dX << so3::hat(X), -Rr.transpose();
                    obs = rb.head<2>();
                }
                if (X.z() < 0.05) continue;
                const Vec2 r = f * (X.head<2>() / X.z() - obs);
                const double e = r.norm();
                if (e < cfg_.reproj_px) ok = true;
                const double delta = 2.0, w = e < delta ? 1.0 : delta / e;
                cost += e < delta ? 0.5 * e * e : delta * (e - 0.5 * delta);
                if (H) {
                    Mat<2, 3> dp;
                    dp << 1 / X.z(), 0, -X.x() / (X.z() * X.z()), 0, 1 / X.z(), -X.y() / (X.z() * X.z());
                    const Mat<2, 6> J = f * dp * dX;
                    *H += w * J.transpose() * J;
                    *g += w * J.transpose() * r;
                }
            }
            if (ok) ++inl;
        }
        return cost;
    };
    int inliers = 0;
    for (int it = 0; it < 10; ++it) {
        Mat<6, 6> H = Mat<6, 6>::Zero();
        Vec<6> g = Vec<6>::Zero();
        residuals(R, t, inliers, &H, &g);
        H.diagonal().array() += 1e-6;
        const Vec<6> d = -H.ldlt().solve(g);
        if (!d.allFinite()) return reject(&LoopStats::rejected_inliers);
        R = so3::normalize(R * so3::exp(d.head<3>()));
        t += d.tail<3>();
        if (d.norm() < 1e-7) break;
    }
    residuals(R, t, inliers, nullptr, nullptr);
    if (inliers < cfg_.min_inliers) return reject(&LoopStats::rejected_inliers);

    // Must be consistent with the trajectory estimate within the drift we can expect after dt.
    const double dt = std::abs(B.t - A.t);
    const Mat3 dR = prior_ab.R.transpose() * R;
    const double dAng = std::abs(Eigen::AngleAxisd(dR).angle()) * 180 / M_PI;
    const double dPos = (t - prior_ab.p).norm();
    if (dPos > cfg_.drift_m + cfg_.drift_m_per_s * dt || dAng > cfg_.drift_deg + cfg_.drift_deg_per_s * dt)
        return reject(&LoopStats::rejected_prior);

    out.a = a;
    out.b = b;
    out.T_ab = {R, t};
    out.inliers = inliers;
    return true;
}

std::vector<std::pair<int, int>> LoopCloser::candidates(const std::vector<Pose>& poses,
                                                        const std::unordered_set<uint64_t>& tried) const {
    const int n = static_cast<int>(kfs_.size());
    std::vector<Vec3> C(n), F(n), Lk(n);
    for (int i = 0; i < n; ++i) {
        C[i] = poses[i].p;
        F[i] = poses[i].R.col(2);
        Lk[i] = C[i] + kfs_[i].median_depth * F[i];
    }
    const double cosMax = std::cos(cfg_.max_angle_deg * M_PI / 180);
    std::vector<std::pair<int, int>> out;
    struct Scored { int i; double s; };
    std::vector<Scored> sc;
    for (int j = 0; j < n; ++j) {
        sc.clear();
        for (int i = 0; i < j; ++i) {
            if (kfs_[j].t - kfs_[i].t < cfg_.min_time_gap) break;   // keyframes are time ordered
            if (kfs_[i].segment != kfs_[j].segment) continue;
            const double dc = (C[i] - C[j]).norm();
            if (dc > cfg_.max_distance) continue;
            const double cosA = F[i].dot(F[j]);
            if (cosA < cosMax) continue;
            const double dl = (Lk[i] - Lk[j]).norm();
            if (dl > cfg_.max_lookat_distance) continue;
            if (tried.count((uint64_t(i) << 32) | uint64_t(j))) continue;
            sc.push_back({i, dl + 0.5 * dc + (1 - cosA)});
        }
        std::sort(sc.begin(), sc.end(), [](const Scored& x, const Scored& y) { return x.s < y.s; });
        // Best few, from different moments of the earlier visit(s).
        std::vector<double> times;
        for (const Scored& c : sc) {
            if (static_cast<int>(times.size()) >= cfg_.candidates_per_keyframe) break;
            const double ti = kfs_[c.i].t;
            if (std::any_of(times.begin(), times.end(), [&](double x) { return std::abs(x - ti) < 2.0; })) continue;
            times.push_back(ti);
            out.push_back({c.i, j});
        }
    }
    return out;
}

std::vector<Pose> LoopCloser::solve(const std::vector<Pose>& poses, std::vector<LoopEdge>& loops, int& accepted) const {
    const int n = static_cast<int>(poses.size());
    // Variables per keyframe: corrected position x (3) and a yaw correction psi about world z:
    //   R' = Rz(psi) R,  p' = x.
    std::vector<Vec3> x(n);
    std::vector<double> psi(n, 0.0);
    for (int i = 0; i < n; ++i) x[i] = poses[i].p;

    struct Edge {
        int a, b;
        Vec3 w;        // measured a->b offset in world axes at a's original orientation
        double yaw;    // measured yaw a->b
        double sp, sy; // sigmas
        bool loop;
        int loopIndex;
        double weight = 1;
    };
    std::vector<Edge> edges;
    for (int k = 1; k < n; ++k) {
        if (kfs_[k].segment != kfs_[k - 1].segment) continue;
        const Vec3 w = poses[k].p - poses[k - 1].p;
        const double dt = std::max(0.0, kfs_[k].t - kfs_[k - 1].t);
        const double rot = std::abs(Eigen::AngleAxisd(poses[k - 1].R.transpose() * poses[k].R).angle());
        edges.push_back({k - 1, k, w, 0.0, cfg_.odom_sigma_p + cfg_.odom_sigma_p_rel * w.norm(),
                         cfg_.odom_sigma_yaw * std::sqrt(std::max(dt, 0.05) / 0.25) + 0.01 * rot, false, -1});
    }
    const size_t nOdom = edges.size();
    for (size_t l = 0; l < loops.size(); ++l) {
        const LoopEdge& e = loops[l];
        const Mat3 Aw = poses[e.a].R * e.T_ab.R * poses[e.b].R.transpose();
        edges.push_back({e.a, e.b, poses[e.a].R * e.T_ab.p, yawOf(Aw), cfg_.loop_sigma_p, cfg_.loop_sigma_yaw, true,
                         static_cast<int>(l)});
    }
    // Gauge: the first keyframe of every segment stays where the VLIO put it.
    std::vector<int> anchors;
    for (int k = 0; k < n; ++k)
        if (k == 0 || kfs_[k].segment != kfs_[k - 1].segment) anchors.push_back(k);

    auto residual = [&](const Edge& e, Vec<4>& r) {
        r.head<3>() = (x[e.b] - x[e.a] - rotZ(psi[e.a]) * e.w) / e.sp;
        r(3) = wrap(psi[e.a] + e.yaw - psi[e.b]) / e.sy;
    };

    std::vector<char> active(edges.size(), 1);
    accepted = 0;
    for (int round = 0; round < 2; ++round) {
        for (int k = 0; k < n; ++k) { x[k] = poses[k].p; psi[k] = 0; }
        for (int it = 0; it < 15; ++it) {
            std::vector<Eigen::Triplet<double>> trip;
            trip.reserve(edges.size() * 32 + anchors.size() * 4);
            VecX g = VecX::Zero(4 * n);
            for (size_t ei = 0; ei < edges.size(); ++ei) {
                if (!active[ei]) continue;
                Edge& e = edges[ei];
                Vec<4> r;
                residual(e, r);
                // Cauchy weight for loops (IRLS), annealed from wide to c^2 = 4 (whitened 4-DoF residual) so
                // loops that correct a lot of drift are not down-weighted before the graph has moved.
                const double c2 = std::max(4.0, 1e4 / std::pow(4.0, it));
                e.weight = e.loop ? 1.0 / (1.0 + r.squaredNorm() / c2) : 1.0;
                Mat<4, 4> Ja = Mat<4, 4>::Zero(), Jb = Mat<4, 4>::Zero();
                const Vec3 dRw = rotZ(psi[e.a]) * Vec3(-e.w.y(), e.w.x(), 0);
                Ja.block<3, 3>(0, 0) = -Mat3::Identity() / e.sp;
                Ja.block<3, 1>(0, 3) = -dRw / e.sp;
                Ja(3, 3) = 1 / e.sy;
                Jb.block<3, 3>(0, 0) = Mat3::Identity() / e.sp;
                Jb(3, 3) = -1 / e.sy;
                const int ia = 4 * e.a, ib = 4 * e.b;
                const Mat<4, 4> Haa = e.weight * Ja.transpose() * Ja, Hab = e.weight * Ja.transpose() * Jb,
                                Hbb = e.weight * Jb.transpose() * Jb;
                for (int r0 = 0; r0 < 4; ++r0)
                    for (int c0 = 0; c0 < 4; ++c0) {
                        trip.emplace_back(ia + r0, ia + c0, Haa(r0, c0));
                        trip.emplace_back(ib + r0, ib + c0, Hbb(r0, c0));
                        trip.emplace_back(ia + r0, ib + c0, Hab(r0, c0));
                        trip.emplace_back(ib + c0, ia + r0, Hab(r0, c0));
                    }
                g.segment<4>(ia) += e.weight * Ja.transpose() * r;
                g.segment<4>(ib) += e.weight * Jb.transpose() * r;
            }
            for (int k : anchors) {
                const double wp = 1e8, wy = 1e10;
                for (int d = 0; d < 3; ++d) {
                    trip.emplace_back(4 * k + d, 4 * k + d, wp);
                    g(4 * k + d) += wp * (x[k](d) - poses[k].p(d));
                }
                trip.emplace_back(4 * k + 3, 4 * k + 3, wy);
                g(4 * k + 3) += wy * psi[k];
            }
            Eigen::SparseMatrix<double> H(4 * n, 4 * n);
            H.setFromTriplets(trip.begin(), trip.end());
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(H);
            if (ldlt.info() != Eigen::Success) break;
            const VecX d = -ldlt.solve(g);
            if (!d.allFinite()) break;
            for (int k = 0; k < n; ++k) {
                x[k] += d.segment<3>(4 * k);
                psi[k] = wrap(psi[k] + d(4 * k + 3));
            }
            if (it >= 7 && d.lpNorm<Eigen::Infinity>() < 1e-6) break;
        }
        // Drop loops that disagree with the optimised graph, then re-solve once without them.
        int dropped = 0;
        accepted = 0;
        for (size_t ei = nOdom; ei < edges.size(); ++ei) {
            if (!active[ei]) continue;
            Vec<4> r;
            residual(edges[ei], r);
            if (r.squaredNorm() > cfg_.loop_outlier_chi2) { active[ei] = 0; ++dropped; }
            else ++accepted;
        }
        if (dropped == 0) break;
    }
    // Keep only the accepted loops.
    std::vector<LoopEdge> kept;
    for (size_t ei = nOdom; ei < edges.size(); ++ei)
        if (active[ei]) kept.push_back(loops[edges[ei].loopIndex]);
    loops.swap(kept);

    std::vector<Pose> out(n);
    for (int k = 0; k < n; ++k) out[k] = {so3::normalize(rotZ(psi[k]) * poses[k].R), x[k]};
    return out;
}

std::vector<Pose> LoopCloser::optimize(const std::vector<Pose>& poses, LoopStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    LoopStats st;
    st.keyframes = static_cast<int>(kfs_.size());
    std::vector<Pose> cur = poses;
    if (poses.size() != kfs_.size() || kfs_.size() < 2) {
        if (stats) *stats = st;
        return cur;
    }
    std::unordered_set<uint64_t> tried;
    std::vector<LoopEdge> verified;
    int accepted = 0;
    for (int pass = 0; pass < std::max(1, cfg_.passes); ++pass) {
        const auto cand = candidates(cur, tried);
        int found = 0;
        for (const auto& [a, b] : cand) {
            tried.insert((uint64_t(a) << 32) | uint64_t(b));
            // A verified loop between the same two moments already constrains this pair.
            const double ta = kfs_[a].t, tb = kfs_[b].t;
            bool covered = false;
            for (auto it = verified.rbegin(); it != verified.rend() && !covered; ++it)
                covered = std::abs(kfs_[it->a].t - ta) < cfg_.loop_spacing_s && std::abs(kfs_[it->b].t - tb) < cfg_.loop_spacing_s;
            if (covered) continue;
            ++st.candidates;
            LoopEdge e;
            if (verify(a, b, compose(inverse(cur[a]), cur[b]), e, &st)) {
                verified.push_back(e);
                ++found;
            }
        }
        st.verified += found;
        if (found == 0 && pass > 0) break;
        std::vector<LoopEdge> loops = verified;
        cur = solve(poses, loops, accepted);
    }
    // Final graph with every verified loop (outliers removed inside).
    loops_ = verified;
    cur = solve(poses, loops_, accepted);
    st.accepted = accepted;
    for (size_t k = 0; k < poses.size(); ++k) {
        st.max_correction_m = std::max(st.max_correction_m, (cur[k].p - poses[k].p).norm());
        st.max_correction_deg =
            std::max(st.max_correction_deg, std::abs(yawOf(cur[k].R * poses[k].R.transpose())) * 180 / M_PI);
    }
    st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (stats) *stats = st;
    return cur;
}

}  // namespace slamcore
