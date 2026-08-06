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
        .package(url: "https://github.com/kewlbear/TensorFlowLiteSwift.git", branch: "master"),
    ],
    targets: [
        .target(
            name: "FaceRecognitionPlugin",
            dependencies: [
                .product(name: "Capacitor", package: "capacitor-swift-pm"),
                .product(name: "Cordova", package: "capacitor-swift-pm"),
                .product(name: "TensorFlowLiteSwift", package: "TensorFlowLiteSwift"),
            ],
            path: "ios/Sources/FaceRecognitionPlugin",
            resources: [
                // Model TFLite di-bundle otomatis ke dalam iOS app
                .process("Resources/mobile_face_net.tflite"),
                .process("Resources/anti_spoof.tflite"),
            ]),
        .testTarget(
            name: "FaceRecognitionPluginTests",
            dependencies: ["FaceRecognitionPlugin"],
            path: "ios/Tests/FaceRecognitionPluginTests")
    ]
)