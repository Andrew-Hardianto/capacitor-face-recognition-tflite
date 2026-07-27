import Capacitor
import Foundation
// import TensorFlowLite
import Vision

/// Please read the Capacitor iOS Plugin Development Guide
/// here: https://capacitorjs.com/docs/plugins/ios
@objc(FaceRecognitionPlugin)
public class FaceRecognitionPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "FaceRecognitionPlugin"
    public let jsName = "FaceRecognition"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "extractFaceFeature", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "compareFaces", returnType: CAPPluginReturnPromise),
    ]

    // Implementasi utama dipisah ke file FaceRecognition.swift (opsional),
    // tapi kita bisa tulis langsung di sini untuk mempermudah:
    private let implementation = FaceRecognition()

    @objc func extractFaceFeature(_ call: CAPPluginCall) {
        guard let data = Data(base64Encoded: imageBase64, options: .ignoreUnknownCharacters),
            let image = UIImage(data: data),
            let cgImage = image.cgImage
        else {
            call.reject("Gagal decode Base64 menjadi gambar")
            return
        }

        // 1. Bersihkan path dan ubah menjadi UIImage
        let cleanPath = imagePath.replacingOccurrences(of: "file://", with: "")
        guard let image = UIImage(contentsOfFile: cleanPath),
            let cgImage = image.cgImage
        else {
            call.reject("Gagal memuat gambar dari path yang diberikan")
            return
        }

        // 2. Siapkan Apple Vision untuk Face Detection
        let request = VNDetectFaceRectanglesRequest { (req, error) in
            if let err = error {
                call.reject("Vision gagal mendeteksi: \(err.localizedDescription)")
                return
            }

            guard let results = req.results as? [VNFaceObservation],
                let face = results.first
            else {
                call.reject("Tidak ada wajah yang terdeteksi")
                return
            }

            // 3. Hitung Bounding Box (Vision koordinatnya dari kiri bawah)
            let width = image.size.width
            let height = image.size.height
            let boundingBox = face.boundingBox
            let rect = CGRect(
                x: boundingBox.origin.x * width,
                y: (1 - boundingBox.origin.y - boundingBox.size.height) * height,
                width: boundingBox.size.width * width,
                height: boundingBox.size.height * height
            )

            // 4. Potong (Crop) Gambar
            guard let croppedCgImage = cgImage.cropping(to: rect) else {
                call.reject("Gagal memotong gambar wajah")
                return
            }
            let croppedFace = UIImage(cgImage: croppedCgImage)

            // 5. Resize ke 112x112 pixel (Tanpa scale retina)
            let targetSize = CGSize(width: 112, height: 112)
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)

            let _ = renderer.image { _ in
                croppedFace.draw(in: CGRect(origin: .zero, size: targetSize))
            }

            // 6. TODO: TFLite Inference di sini
            let embeddings: [Float] = Array(repeating: 0.0, count: 128)

            // 7. Kembalikan data
            call.resolve([
                "embedding": embeddings
            ])
        }

        // Eksekusi Vision Request
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            call.reject("Gagal menjalankan Vision Request: \(error.localizedDescription)")
        }
    }

    @objc func compareFaces(_ call: CAPPluginCall) {
        // Ambil array dari Angular (Float)
        guard let vec1 = call.getArray("vector1", Float.self),
            let vec2 = call.getArray("vector2", Float.self),
            vec1.count == vec2.count
        else {
            call.reject("Vector tidak valid atau panjangnya tidak sama")
            return
        }

        var sum: Float = 0
        for i in 0..<vec1.count {
            let diff = vec1[i] - vec2[i]
            sum += diff * diff
        }

        let distance = sqrt(sum)
        var similarityPercentage = 100.0 - (distance * 30.0)
        if similarityPercentage < 0 { similarityPercentage = 0 }

        call.resolve([
            "distance": distance,
            "similarityPercentage": similarityPercentage,
        ])
    }
}
