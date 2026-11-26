// swift-tools-version: 6.0
import PackageDescription

// xtool builds the `SlamToGo` library product into an iOS .app (see xtool.yml).
let package = Package(
    name: "SlamToGo",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "SlamToGo", targets: ["SlamToGo"])
    ],
    dependencies: [
        .package(path: "../core")             // shared C++ VLIO core (C API)
    ],
    targets: [
        .target(
            name: "SlamToGo",
            dependencies: [.product(name: "SlamCore", package: "core")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
