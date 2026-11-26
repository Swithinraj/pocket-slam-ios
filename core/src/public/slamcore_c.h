/* SLAM to Go - C API of the VLIO core (used from Swift on iOS and JNI on Android). */
#ifndef SLAMCORE_C_H
#define SLAMCORE_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SlamHandle SlamHandle;

typedef struct {
    double t;
    int32_t initialized;
    int32_t keyframe;
    double R[9];          /* camera -> world rotation, row-major (OpenCV camera, world z-up) */
    double p[3];          /* camera position in world [m] */
    double v[3];          /* body velocity in world [m/s] */
    int32_t tracked;      /* features tracked in this frame */
    int32_t landmarks;    /* landmarks in the sliding window */
    double solve_ms;      /* back-end time (keyframes only) */
    double total_ms;      /* whole addFrame time */
    int32_t reinitialized; /* a new map / world frame started at this frame (after an interruption) */
} SlamFrameResult;

/* R_bc: body(IMU) <- camera rotation, row-major. p_bc: camera position in the body frame [m].
   focal: focal length [px] of the tracking image (full-resolution focal / downscale). */
SlamHandle* slam_create(const double R_bc[9], const double p_bc[3], double focal);
void slam_destroy(SlamHandle* h);

/* IMU sample in the body frame: specific force [m/s^2] (+9.81 "up" at rest) and angular rate [rad/s]. */
void slam_add_imu(SlamHandle* h, double t, double ax, double ay, double az, double gx, double gy, double gz);

/* Downscale a luma plane by an integer factor (box filter) into dst (width/factor x height/factor). */
void slam_downsample_luma(const uint8_t* src, int32_t width, int32_t height, int32_t stride, int32_t factor, uint8_t* dst);

/* Process one frame. gray: tracking image (tightly packed). fx..cy: intrinsics of that image.
   depth/conf: optional metric depth [m] and confidence (0..2) with the same field of view (may be NULL).
   Call only after IMU samples beyond t were added. Returns 1 when a pose is available. */
int32_t slam_add_frame(SlamHandle* h, double t, const uint8_t* gray, int32_t width, int32_t height, double fx,
                       double fy, double cx, double cy, const float* depth, const uint8_t* conf, int32_t depth_width,
                       int32_t depth_height, SlamFrameResult* out);

/* Landmarks of the current window as xyz triples. Returns the number written. */
int32_t slam_map_points(SlamHandle* h, float* xyz, int32_t max_points);
/* Currently tracked feature positions (pixels of the tracking image) as uv pairs. */
int32_t slam_features(SlamHandle* h, float* uv, int32_t max_features);
/* Keyframes that left the window since the last call: 13 doubles each (t, R row-major 9, p 3). */
int32_t slam_take_finalized(SlamHandle* h, double* out, int32_t max_keyframes);
/* Keyframes still in the sliding window (latest estimates), same layout. Use at the end of a sequence. */
int32_t slam_window_keyframes(SlamHandle* h, double* out, int32_t max_keyframes);

#ifdef __cplusplus
}
#endif

#endif /* SLAMCORE_C_H */
