// swift-tools-version: 6.0
import PackageDescription

// The VLIO core as a SwiftPM package (used by the iOS app). Linux tools / Android use CMakeLists.txt.
// Swift sees only the C API (src/public/slamcore_c.h); the C++ implementation stays internal.
let package = Package(
    name: "SlamCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "SlamCore", targets: ["SlamCore"])
    ],
    targets: [
        .target(
            name: "SlamCore",
            path: "src",
            publicHeadersPath: "public",
            cxxSettings: [
                .headerSearchPath("../include"),
                .headerSearchPath("../third_party/eigen"),
                .define("EIGEN_MPL2_ONLY"),
                .define("NDEBUG"),
                // The solver must be optimised even in debug app builds (Eigen at -O0 is ~30x slower).
                .unsafeFlags(["-O3"]),
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)
