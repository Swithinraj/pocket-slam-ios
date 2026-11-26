#include "slamcore/estimator.hpp"

#include "slamcore/so3.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace slamcore {

// ------------------------------------------------------------------ NavState
void NavState::plus(const Vec<15>& d) {
    R = so3::normalize(R * so3::exp(d.segment<3>(0)));
    p += d.segment<3>(3);
    v += d.segment<3>(6);
    ba += d.segment<3>(9);
    bg += d.segment<3>(12);
}

Vec<15> NavState::minus(const NavState& x0) const {
    Vec<15> d;
    d.segment<3>(0) = so3::log(x0.R.transpose() * R);
    d.segment<3>(3) = p - x0.p;
    d.segment<3>(6) = v - x0.v;
    d.segment<3>(9) = ba - x0.ba;
    d.segment<3>(12) = bg - x0.bg;
    return d;
}

// ------------------------------------------------------------------ setup
Estimator::Estimator(const EstimatorConfig& cfg, const ImuNoise& noise, const Extrinsics& ext)
    : cfg_(cfg), noise_(noise), ext_(ext) {}

int Estimator::indexOf(int kf_id) const {
    for (size_t i = 0; i < window_.size(); ++i)
        if (window_[i].id == kf_id) return static_cast<int>(i);
    return -1;
}

static void addObservations(std::map<int, Landmark>& lms, int kf_id, const std::vector<FeatureObs>& feats) {
    for (const FeatureObs& f : feats) {
        auto it = lms.find(f.id);
        if (it == lms.end()) {
            Landmark lm;
            lm.id = f.id;
            lm.host_kf = kf_id;
            if (f.depth > 0) {
                lm.inv_depth = 1.0 / f.depth;
                lm.initialized = true;
            }
            lm.obs.push_back({kf_id, f.xn, f.depth, f.depth_sigma});
            lms.emplace(f.id, std::move(lm));
        } else {
            it->second.obs.push_back({kf_id, f.xn, f.depth, f.depth_sigma});
        }
    }
}

void Estimator::initialize(double t, const Mat3& R_wb, const Vec3& bg, const std::vector<FeatureObs>& feats) {
    window_.clear();
    landmarks_.clear();
    KeyFrame kf{next_kf_id_++, t, NavState{}, nullptr};
    kf.x.R = R_wb;
    kf.x.bg = bg;
    // Gauge (position, yaw) + weak priors on the unknown initial velocity and biases.
    Vec<15> sigma;
    sigma << 0.01, 0.01, 0.01, 1e-4, 1e-4, 1e-4, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.02, 0.02, 0.02;
    prior_ = Prior{};
    prior_.kf_ids = {kf.id};
    prior_.x0 = {kf.x};
    prior_.J = sigma.cwiseInverse().asDiagonal();
    prior_.r0 = VecX::Zero(15);
    addObservations(landmarks_, kf.id, feats);
    window_.push_back(std::move(kf));
}

void Estimator::addKeyframe(double t, std::unique_ptr<Preintegration> pre, const std::vector<FeatureObs>& feats) {
    const NavState& a = window_.back().x;
    const double T = pre->dt();
    const Vec3& g = cfg_.gravity;
    KeyFrame kf{next_kf_id_++, t, a, nullptr};
    kf.x.R = so3::normalize(a.R * pre->deltaR(a.bg));
    kf.x.v = a.v + g * T + a.R * pre->deltaV(a.ba, a.bg);
    kf.x.p = a.p + a.v * T + 0.5 * g * T * T + a.R * pre->deltaP(a.ba, a.bg);
    kf.pre = std::move(pre);
    const int id = kf.id;
    window_.push_back(std::move(kf));
    addObservations(landmarks_, id, feats);
    for (auto& [lid, lm] : landmarks_)
        if (!lm.initialized && lm.obs.size() >= 2) triangulate(lm);
}

bool Estimator::usable(const Landmark& lm) const {
    return lm.initialized && lm.inv_depth > 0 && lm.obs.size() >= 2 && indexOf(lm.host_kf) >= 0;
}

static Pose cameraPose(const NavState& x, const Extrinsics& e) {
    return {x.R * e.R_bc, x.p + x.R * e.p_bc};
}

void Estimator::triangulate(Landmark& lm) {
    const int h = indexOf(lm.host_kf);
    if (h < 0) return;
    const Pose ch = cameraPose(window_[h].x, ext_);
    double best = 0;
    const Observation* bo = nullptr;
    for (const Observation& o : lm.obs) {
        const int j = indexOf(o.kf_id);
        if (j < 0 || j == h) continue;
        const double b = (cameraPose(window_[j].x, ext_).p - ch.p).norm();
        if (b > best) { best = b; bo = &o; }
    }
    if (!bo || best < 0.03) return;
    const Pose cj = cameraPose(window_[indexOf(bo->kf_id)].x, ext_);
    auto rows = [](const Pose& c, const Vec2& x, Eigen::Matrix<double, 2, 4>& A) {
        Eigen::Matrix<double, 3, 4> P;
        P.leftCols<3>() = c.R.transpose();
        P.col(3) = -c.R.transpose() * c.p;
        A.row(0) = x.x() * P.row(2) - P.row(0);
        A.row(1) = x.y() * P.row(2) - P.row(1);
    };
    Eigen::Matrix4d A;
    Eigen::Matrix<double, 2, 4> a;
    rows(ch, lm.obs.front().xn, a);
    A.topRows<2>() = a;
    rows(cj, bo->xn, a);
    A.bottomRows<2>() = a;
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    const Eigen::Vector4d X = svd.matrixV().col(3);
    if (std::abs(X(3)) < 1e-12) return;
    const Vec3 Pw = X.head<3>() / X(3);
    const double z = (ch.R.transpose() * (Pw - ch.p)).z();
    if (z < cfg_.depth_min || z > cfg_.depth_max) return;
    lm.inv_depth = 1.0 / z;
    lm.initialized = true;
}

Vec3 Estimator::hostPoint(const Landmark& lm) const {
    const NavState& h = window_[indexOf(lm.host_kf)].x;
    const Vec3 Pc = Vec3(lm.obs.front().xn.x(), lm.obs.front().xn.y(), 1.0) / lm.inv_depth;
    return h.R * (ext_.R_bc * Pc + ext_.p_bc) + h.p;
}

// ------------------------------------------------------------------ residuals
VecX Estimator::imuResidual(const NavState& a, const NavState& b, const Preintegration& pre) const {
    const double T = pre.dt();
    const Vec3& g = cfg_.gravity;
    VecX r(15);
    const Mat3 dR = pre.deltaR(a.bg);
    r.segment<3>(0) = so3::log(dR.transpose() * a.R.transpose() * b.R);
    r.segment<3>(3) = a.R.transpose() * (b.v - a.v - g * T) - pre.deltaV(a.ba, a.bg);
    r.segment<3>(6) = a.R.transpose() * (b.p - a.p - a.v * T - 0.5 * g * T * T) - pre.deltaP(a.ba, a.bg);
    r.segment<3>(9) = b.ba - a.ba;
    r.segment<3>(12) = b.bg - a.bg;
    return r;
}

Mat<15, 15> Estimator::imuSqrtInfo(const Preintegration& pre) const {
    Mat<15, 15> cov = Mat<15, 15>::Zero();
    cov.block<9, 9>(0, 0) = pre.covariance();
    const double T = std::max(pre.dt(), 1e-3);
    cov.block<3, 3>(9, 9) = Mat3::Identity() * noise_.acc_walk * noise_.acc_walk * T;
    cov.block<3, 3>(12, 12) = Mat3::Identity() * noise_.gyr_walk * noise_.gyr_walk * T;
    cov.diagonal().array() += 1e-12;
    const Mat<15, 15> info = cov.inverse();
    Eigen::LLT<Mat<15, 15>> llt(0.5 * (info + info.transpose()));
    return llt.matrixL().transpose();   // U with U^T U = info
}

bool Estimator::visualResidual(const NavState& host, const NavState& tgt, double inv_depth, const Vec2& xn_host,
                               const Observation& o, VecX& r) const {
    const Vec3 Pch = Vec3(xn_host.x(), xn_host.y(), 1.0) / inv_depth;
    const Vec3 Pw = host.R * (ext_.R_bc * Pch + ext_.p_bc) + host.p;
    const Vec3 Pb = tgt.R.transpose() * (Pw - tgt.p);
    const Vec3 Pc = ext_.R_bc.transpose() * (Pb - ext_.p_bc);
    if (Pc.z() < 0.05) return false;
    const double s = cfg_.focal / cfg_.pixel_sigma;
    const bool useDepth = o.depth > 0 && cfg_.depth_on_targets;
    r.resize(useDepth ? 3 : 2);
    r(0) = s * (Pc.x() / Pc.z() - o.xn.x());
    r(1) = s * (Pc.y() / Pc.z() - o.xn.y());
    if (useDepth) r(2) = (Pc.z() - o.depth) / o.depth_sigma;
    return true;
}

double Estimator::robustify(VecX& r, std::vector<Block>* blocks) const {
    const double s = r.squaredNorm();
    const double k = cfg_.huber;
    if (s <= k * k) return s;
    const double n = std::sqrt(s);
    const double w = std::sqrt(k / n);
    r *= w;
    if (blocks)
        for (Block& b : *blocks) b.J *= w;
    return 2 * k * n - k * k;
}

// ------------------------------------------------------------------ linearisation
static constexpr double kEps = 1e-6;

void Estimator::linearizeImu(int j, std::vector<Lin>& out, double& cost, bool jac) const {
    const NavState& a = window_[j - 1].x;
    const NavState& b = window_[j].x;
    const Preintegration& pre = *window_[j].pre;
    const Mat<15, 15> U = imuSqrtInfo(pre);
    VecX r = U * imuResidual(a, b, pre);
    cost += r.squaredNorm();
    if (!jac) return;
    MatX Ja(15, 15), Jb(15, 15);
    for (int k = 0; k < 15; ++k) {
        Vec<15> d = Vec<15>::Zero();
        d(k) = kEps;
        NavState ap = a, am = a, bp = b, bm = b;
        ap.plus(d);
        am.plus(-d);
        bp.plus(d);
        bm.plus(-d);
        Ja.col(k) = U * (imuResidual(ap, b, pre) - imuResidual(am, b, pre)) / (2 * kEps);
        Jb.col(k) = U * (imuResidual(a, bp, pre) - imuResidual(a, bm, pre)) / (2 * kEps);
    }
    out.push_back({r, {{j - 1, Ja}, {j, Jb}}});
}

void Estimator::linearizeLandmark(const Landmark& lm, int li, std::vector<Lin>& out, double& cost, bool jac) const {
    const int h = indexOf(lm.host_kf);
    const NavState& host = window_[h].x;
    const Vec2& xnh = lm.obs.front().xn;
    const double lam = lm.inv_depth;
    const int lv = -(li + 1);
    for (const Observation& o : lm.obs) {
        const int j = indexOf(o.kf_id);
        if (j < 0) continue;
        if (j == h) {
            if (o.depth <= 0 || !cfg_.depth_on_host) continue;
            VecX r(1);
            r(0) = (1.0 / lam - o.depth) / o.depth_sigma;
            std::vector<Block> blocks;
            if (jac) {
                MatX J(1, 1);
                J(0, 0) = -1.0 / (lam * lam * o.depth_sigma);
                blocks.push_back({lv, J});
            }
            cost += robustify(r, jac ? &blocks : nullptr);
            if (jac) out.push_back({r, std::move(blocks)});
            continue;
        }
        const NavState& tgt = window_[j].x;
        VecX r;
        if (!visualResidual(host, tgt, lam, xnh, o, r)) continue;
        std::vector<Block> blocks;
        if (jac) {
            const int m = static_cast<int>(r.size());
            MatX Jh = MatX::Zero(m, 15), Jt = MatX::Zero(m, 15), Jl(m, 1);
            VecX rp, rm;
            for (int k = 0; k < 6; ++k) {
                Vec<15> d = Vec<15>::Zero();
                d(k) = kEps;
                NavState p1 = host, m1 = host;
                p1.plus(d);
                m1.plus(-d);
                if (visualResidual(p1, tgt, lam, xnh, o, rp) && visualResidual(m1, tgt, lam, xnh, o, rm))
                    Jh.col(k) = (rp - rm) / (2 * kEps);
                NavState p2 = tgt, m2 = tgt;
                p2.plus(d);
                m2.plus(-d);
                if (visualResidual(host, p2, lam, xnh, o, rp) && visualResidual(host, m2, lam, xnh, o, rm))
                    Jt.col(k) = (rp - rm) / (2 * kEps);
            }
            const double el = kEps * std::max(lam, 1e-3);
            if (visualResidual(host, tgt, lam + el, xnh, o, rp) && visualResidual(host, tgt, lam - el, xnh, o, rm))
                Jl = (rp - rm) / (2 * el);
            else
                Jl.setZero();
            blocks = {{h, Jh}, {j, Jt}, {lv, Jl}};
        }
        cost += robustify(r, jac ? &blocks : nullptr);
        if (jac) out.push_back({r, std::move(blocks)});
    }
}

void Estimator::linearizePrior(std::vector<Lin>& out, double& cost, bool jac) const {
    if (!prior_.valid()) return;
    const int m = static_cast<int>(prior_.kf_ids.size());
    VecX dx(15 * m);
    std::vector<int> idx(m);
    for (int k = 0; k < m; ++k) {
        idx[k] = indexOf(prior_.kf_ids[k]);
        dx.segment<15>(15 * k) = window_[idx[k]].x.minus(prior_.x0[k]);
    }
    VecX r = prior_.r0 + prior_.J * dx;
    cost += r.squaredNorm();
    if (!jac) return;
    std::vector<Block> blocks;
    for (int k = 0; k < m; ++k) {
        Mat<15, 15> D = Mat<15, 15>::Identity();
        D.block<3, 3>(0, 0) = so3::rightJacobianInv(dx.segment<3>(15 * k));
        blocks.push_back({idx[k], prior_.J.middleCols(15 * k, 15) * D});
    }
    out.push_back({r, std::move(blocks)});
}

// ------------------------------------------------------------------ system assembly
namespace {
struct NormalEq {
    MatX Hss, Hsl;
    VecX bs, Hll, bl;
    NormalEq(int S, int L) : Hss(MatX::Zero(S, S)), Hsl(MatX::Zero(S, L)), bs(VecX::Zero(S)), Hll(VecX::Zero(L)), bl(VecX::Zero(L)) {}
};
}  // namespace

template <class LinT>
static void assemble(const std::vector<LinT>& lins, NormalEq& ne) {
    for (const LinT& f : lins) {
        for (const auto& bi : f.blocks) {
            if (bi.var >= 0) ne.bs.segment(15 * bi.var, 15) -= bi.J.transpose() * f.r;
            else ne.bl(-bi.var - 1) -= (bi.J.transpose() * f.r)(0);
            for (const auto& bj : f.blocks) {
                if (bi.var >= 0 && bj.var >= 0)
                    ne.Hss.block(15 * bi.var, 15 * bj.var, 15, 15) += bi.J.transpose() * bj.J;
                else if (bi.var >= 0 && bj.var < 0)
                    ne.Hsl.block(15 * bi.var, -bj.var - 1, 15, 1) += bi.J.transpose() * bj.J;
                else if (bi.var < 0 && bj.var < 0)
                    ne.Hll(-bi.var - 1) += (bi.J.transpose() * bj.J)(0);
            }
        }
    }
}

double Estimator::evaluate() {
    std::vector<Lin> dummy;
    double cost = 0;
    linearizePrior(dummy, cost, false);
    for (size_t j = 1; j < window_.size(); ++j) linearizeImu(static_cast<int>(j), dummy, cost, false);
    for (size_t l = 0; l < opt_landmarks_.size(); ++l) linearizeLandmark(*opt_landmarks_[l], static_cast<int>(l), dummy, cost, false);
    return cost;
}

bool Estimator::optimize() {
    opt_landmarks_.clear();
    for (auto& [id, lm] : landmarks_)
        if (usable(lm)) opt_landmarks_.push_back(&lm);
    const int n = static_cast<int>(window_.size());
    const int S = 15 * n, L = static_cast<int>(opt_landmarks_.size());
    double mu = 1e-4;
    double cost = evaluate();
    for (int it = 0; it < cfg_.max_iterations; ++it) {
        std::vector<Lin> lins;
        double c = 0;
        linearizePrior(lins, c, true);
        for (int j = 1; j < n; ++j) linearizeImu(j, lins, c, true);
        for (int l = 0; l < L; ++l) linearizeLandmark(*opt_landmarks_[l], l, lins, c, true);
        NormalEq ne(S, L);
        assemble(lins, ne);

        std::vector<NavState> saved;
        for (const auto& kf : window_) saved.push_back(kf.x);
        std::vector<double> savedL;
        for (auto* lm : opt_landmarks_) savedL.push_back(lm->inv_depth);

        bool accepted = false;
        double stepNorm = 0;
        for (int tries = 0; tries < 6 && !accepted; ++tries) {
            MatX H = ne.Hss;
            H.diagonal() += mu * ne.Hss.diagonal().cwiseMax(1e-6);
            VecX b = ne.bs;
            VecX Hll = ne.Hll * (1 + mu);
            Hll.array() += 1e-9;
            for (int l = 0; l < L; ++l) {
                const auto c2 = ne.Hsl.col(l);
                H.noalias() -= c2 * c2.transpose() / Hll(l);
                b.noalias() -= c2 * (ne.bl(l) / Hll(l));
            }
            const VecX dx = H.ldlt().solve(b);
            if (!dx.allFinite()) { mu *= 10; continue; }
            for (int i = 0; i < n; ++i) window_[i].x.plus(dx.segment<15>(15 * i));
            bool ok = true;
            for (int l = 0; l < L; ++l) {
                const double dl = (ne.bl(l) - ne.Hsl.col(l).dot(dx)) / Hll(l);
                opt_landmarks_[l]->inv_depth += dl;
                if (!(opt_landmarks_[l]->inv_depth > 1e-4)) ok = false;
            }
            const double newCost = ok ? evaluate() : 1e300;
            if (newCost < cost) {
                accepted = true;
                cost = newCost;
                mu = std::max(mu / 3, 1e-8);
                stepNorm = dx.norm();
            } else {
                for (int i = 0; i < n; ++i) window_[i].x = saved[i];
                for (int l = 0; l < L; ++l) opt_landmarks_[l]->inv_depth = savedL[l];
                mu *= 4;
            }
        }
        if (!accepted || stepNorm < 1e-6) break;
    }
    last_cost_ = cost;
    return true;
}

// ------------------------------------------------------------------ outliers, marginalisation
void Estimator::rejectOutliers() {
    const double thr = cfg_.outlier_px / cfg_.focal;
    for (auto it = landmarks_.begin(); it != landmarks_.end();) {
        Landmark& lm = it->second;
        if (!usable(lm)) { ++it; continue; }
        const double d = 1.0 / lm.inv_depth;
        if (!std::isfinite(d) || d < cfg_.depth_min || d > cfg_.depth_max) {
            rejected_.insert(lm.id);
            it = landmarks_.erase(it);
            continue;
        }
        const int h = indexOf(lm.host_kf);
        const NavState& host = window_[h].x;
        Observation& ho = lm.obs.front();
        if (ho.depth > 0 && std::abs(d - ho.depth) / ho.depth_sigma > cfg_.depth_outlier_sigma) ho.depth = 0;
        int bad = 0;
        for (size_t k = 1; k < lm.obs.size();) {
            Observation& o = lm.obs[k];
            const int j = indexOf(o.kf_id);
            const Vec3 Pw = hostPoint(lm);
            const NavState& t = window_[j].x;
            const Vec3 Pc = ext_.R_bc.transpose() * (t.R.transpose() * (Pw - t.p) - ext_.p_bc);
            const bool front = Pc.z() > 0.05;
            const double e = front ? (Pc.head<2>() / Pc.z() - o.xn).norm() : 1e9;
            if (e > thr) {
                lm.obs.erase(lm.obs.begin() + static_cast<long>(k));
                ++bad;
                continue;
            }
            if (o.depth > 0 && std::abs(Pc.z() - o.depth) / o.depth_sigma > cfg_.depth_outlier_sigma) o.depth = 0;
            ++k;
        }
        if (bad > 0 && lm.obs.size() < 2) rejected_.insert(lm.id);   // stop tracking repeat offenders
        (void)host;
        ++it;
    }
}

void Estimator::marginalizeOldest() {
    const int n = static_cast<int>(window_.size());
    const KeyFrame& k0 = window_.front();
    finalized_.push_back({k0.t, k0.x});

    std::vector<Landmark*> marg;
    for (auto& [id, lm] : landmarks_)
        if (lm.host_kf == k0.id && usable(lm)) marg.push_back(&lm);
    const int M = static_cast<int>(marg.size());

    std::vector<Lin> lins;
    double c = 0;
    linearizePrior(lins, c, true);
    if (n > 1) linearizeImu(1, lins, c, true);
    for (int l = 0; l < M; ++l) linearizeLandmark(*marg[l], l, lins, c, true);
    NormalEq ne(15 * n, M);
    assemble(lins, ne);

    const int m = 15 + M, k = 15 * (n - 1);
    MatX Hmm = MatX::Zero(m, m), Hmk = MatX::Zero(m, k);
    VecX bm(m);
    Hmm.topLeftCorner(15, 15) = ne.Hss.topLeftCorner(15, 15);
    if (M > 0) {
        Hmm.block(0, 15, 15, M) = ne.Hsl.topRows(15);
        Hmm.block(15, 0, M, 15) = ne.Hsl.topRows(15).transpose();
        Hmm.bottomRightCorner(M, M) = ne.Hll.asDiagonal();
        Hmk.bottomRows(M) = ne.Hsl.bottomRows(k).transpose();
    }
    Hmk.topRows(15) = ne.Hss.block(0, 15, 15, k);
    bm.head(15) = ne.bs.head(15);
    if (M > 0) bm.tail(M) = ne.bl;
    const MatX Hkk = ne.Hss.bottomRightCorner(k, k);
    const VecX bk = ne.bs.tail(k);

    Eigen::SelfAdjointEigenSolver<MatX> es(0.5 * (Hmm + Hmm.transpose()));
    const VecX ev = es.eigenvalues();
    VecX evInv = VecX::Zero(m);
    for (int i = 0; i < m; ++i) evInv(i) = ev(i) > 1e-8 ? 1.0 / ev(i) : 0.0;
    const MatX HmmInv = es.eigenvectors() * evInv.asDiagonal() * es.eigenvectors().transpose();
    MatX Hs = Hkk - Hmk.transpose() * HmmInv * Hmk;
    const VecX bs = bk - Hmk.transpose() * HmmInv * bm;

    Eigen::SelfAdjointEigenSolver<MatX> es2(0.5 * (Hs + Hs.transpose()));
    const VecX lam = es2.eigenvalues();
    std::vector<int> keep;
    for (int i = 0; i < k; ++i)
        if (lam(i) > 1e-8) keep.push_back(i);
    Prior p;
    p.J = MatX(keep.size(), k);
    p.r0 = VecX(keep.size());
    for (size_t r = 0; r < keep.size(); ++r) {
        const double s = std::sqrt(lam(keep[r]));
        const VecX v = es2.eigenvectors().col(keep[r]);
        p.J.row(static_cast<long>(r)) = s * v.transpose();
        p.r0(static_cast<long>(r)) = -v.dot(bs) / s;
    }
    for (int i = 1; i < n; ++i) {
        p.kf_ids.push_back(window_[i].id);
        p.x0.push_back(window_[i].x);
    }
    prior_ = std::move(p);

    const int id0 = k0.id;
    for (auto it = landmarks_.begin(); it != landmarks_.end();)
        it = (it->second.host_kf == id0) ? landmarks_.erase(it) : std::next(it);
    window_.pop_front();
    window_.front().pre.reset();
}

void Estimator::repropagate() {
    for (size_t j = 1; j < window_.size(); ++j) {
        Preintegration& pre = *window_[j].pre;
        const NavState& a = window_[j - 1].x;
        if ((a.ba - pre.ba()).norm() > 0.05 || (a.bg - pre.bg()).norm() > 0.005) pre.reintegrate(a.ba, a.bg);
    }
}

void Estimator::process() {
    const auto t0 = std::chrono::steady_clock::now();
    repropagate();
    optimize();
    rejectOutliers();
    if (static_cast<int>(window_.size()) > cfg_.window) marginalizeOldest();
    last_solve_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::unordered_set<int> Estimator::takeRejected() {
    std::unordered_set<int> out;
    out.swap(rejected_);
    return out;
}

std::vector<std::pair<double, NavState>> Estimator::takeFinalized() {
    std::vector<std::pair<double, NavState>> out;
    out.swap(finalized_);
    return out;
}

std::vector<Vec3> Estimator::landmarkPositions() const {
    std::vector<Vec3> out;
    for (const auto& [id, lm] : landmarks_)
        if (usable(lm)) out.push_back(hostPoint(lm));
    return out;
}

int Estimator::activeLandmarks() const {
    int n = 0;
    for (const auto& [id, lm] : landmarks_) n += usable(lm) ? 1 : 0;
    return n;
}

}  // namespace slamcore
