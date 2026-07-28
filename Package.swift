// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CapacitorFaceRecognitionTflite",
    platforms: [.iOS(.v15)],
    products: [
        .library(
            name: "CapacitorFaceRecognitionTflite",
            targets: ["FaceRecognitionPlugin"])
    ],
    dependencies: [
        .package(url: "https://github.com/ionic-team/capacitor-swift-pm.git", from: "8.0.0"),
        .package(url: "https://github.com/tensorflow/tensorflow.git", from: "2.14.0"),
    ],
    targets: [
        .target(
            name: "FaceRecognitionPlugin",
            dependencies: [
                .product(name: "Capacitor", package: "capacitor-swift-pm"),
                .product(name: "Cordova", package: "capacitor-swift-pm"),
                .product(name: "TensorFlowLite", package: "tensorflow"),
            ],
            path: "ios/Sources/FaceRecognitionPlugin"),
        .testTarget(
            name: "FaceRecognitionPluginTests",
            dependencies: ["FaceRecognitionPlugin"],
            path: "ios/Tests/FaceRecognitionPluginTests")
    ]
)