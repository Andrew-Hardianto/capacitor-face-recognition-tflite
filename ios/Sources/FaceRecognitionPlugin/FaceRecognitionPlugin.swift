import Capacitor
import CoreImage
import Foundation
import TensorFlowLite
import UIKit
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

    // ============================
    // Konfigurasi
    // ============================
    // Nama file model tanpa ekstensi (harus ada di bundle: mobile_face_net.tflite)
    private let modelName = "mobile_face_net"
    // Ukuran input model MobileFaceNet
    private let inputSize = 112
    // Ukuran output embedding
    private let embeddingSize = 128

    // TFLite Interpreter (lazy agar dimuat sekali)
    private lazy var interpreter: Interpreter? = {
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "tflite") else {
            print("[FaceRecognitionPlugin] ERROR: File '\(modelName).tflite' tidak ditemukan di bundle.")
            return nil
        }
        do {
            let interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
            return interpreter
        } catch {
            print("[FaceRecognitionPlugin] ERROR: Gagal membuat Interpreter: \(error)")
            return nil
        }
    }()

    // ============================
    // extractFaceFeature
    // ============================
    @objc func extractFaceFeature(_ call: CAPPluginCall) {
        // Pastikan interpreter berhasil dimuat
        guard let interpreter = interpreter else {
            call.reject("Model TFLite gagal dimuat. Pastikan file '\(modelName).tflite' ada di bundle iOS.")
            return
        }

        // 1. Decode Base64 ke UIImage
        guard let imageBase64 = call.getString("imageBase64"),
              let data = Data(base64Encoded: imageBase64, options: .ignoreUnknownCharacters),
              let image = UIImage(data: data),
              let cgImage = image.cgImage
        else {
            call.reject("Gagal decode Base64 menjadi gambar")
            return
        }

        // 2. Deteksi wajah menggunakan Apple Vision
        let request = VNDetectFaceRectanglesRequest { [weak self] (req, error) in
            guard let self = self else { return }

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

            // 3. Konversi bounding box Vision (koordinat dari kiri bawah → kiri atas)
            let imgWidth = image.size.width
            let imgHeight = image.size.height
            let bb = face.boundingBox
            let faceRect = CGRect(
                x: bb.origin.x * imgWidth,
                y: (1 - bb.origin.y - bb.size.height) * imgHeight,
                width: bb.size.width * imgWidth,
                height: bb.size.height * imgHeight
            )

            // 4. Crop wajah dari gambar asli
            guard let croppedCgImage = cgImage.cropping(to: faceRect) else {
                call.reject("Gagal memotong gambar wajah")
                return
            }

            // 5. Resize ke 112x112 (scale = 1 agar tidak jadi 2x/3x di Retina)
            let targetSize = CGSize(width: self.inputSize, height: self.inputSize)
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
            let scaledImage = renderer.image { _ in
                UIImage(cgImage: croppedCgImage).draw(in: CGRect(origin: .zero, size: targetSize))
            }

            // 6. Konversi UIImage ke Data input TFLite (normalisasi ke [-1, 1])
            guard let inputData = self.imageToInputData(scaledImage) else {
                call.reject("Gagal mengkonversi gambar ke format TFLite input")
                return
            }

            // 7. Set input tensor & jalankan inference
            do {
                try interpreter.copy(inputData, toInputAt: 0)
                try interpreter.invoke()

                // 8. Ambil output tensor
                let outputTensor = try interpreter.output(at: 0)
                let outputData = outputTensor.data

                // 9. Konversi output bytes ke [Float]
                let floatCount = outputData.count / MemoryLayout<Float>.size
                var embeddings = [Float](repeating: 0, count: floatCount)
                _ = embeddings.withUnsafeMutableBytes { ptr in
                    outputData.copyBytes(to: ptr)
                }

                call.resolve(["embedding": embeddings])
            } catch {
                call.reject("TFLite inference gagal: \(error.localizedDescription)")
            }
        }

        // Eksekusi Vision Request
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            call.reject("Gagal menjalankan Vision Request: \(error.localizedDescription)")
        }
    }

    // ============================
    // compareFaces
    // ============================
    @objc func compareFaces(_ call: CAPPluginCall) {
        guard let vec1 = call.getArray("vector1", Double.self),
              let vec2 = call.getArray("vector2", Double.self),
              vec1.count == vec2.count
        else {
            call.reject("Vector tidak valid atau panjangnya tidak sama")
            return
        }

        var dotProduct: Double = 0
        var norm1: Double = 0
        var norm2: Double = 0

        // Hitung Cosine Similarity (konsisten dengan Android)
        for i in 0..<vec1.count {
            dotProduct += vec1[i] * vec2[i]
            norm1 += vec1[i] * vec1[i]
            norm2 += vec2[i] * vec2[i]
        }

        guard norm1 > 0, norm2 > 0 else {
            call.reject("Data vektor ada yang bernilai nol semua, cek apakah embedding berhasil")
            return
        }

        let cosineSimilarity = dotProduct / (norm1.squareRoot() * norm2.squareRoot())

        // THRESHOLD / AMBANG BATAS
        // 0.75 adalah standar yang baik.
        // Jika masih tembus wajah orang lain, naikkan ke 0.80 atau 0.85
        let isMatch = cosineSimilarity > 0.75
        let similarityPercentage = max(0, cosineSimilarity * 100.0)

        call.resolve([
            "isMatch": isMatch,
            "similarityPercentage": similarityPercentage,
            "score": cosineSimilarity,
        ])
    }

    // ============================
    // Helper: UIImage → ByteData TFLite
    // ============================
    /// Konversi UIImage 112x112 ke Data berformat [R, G, B, R, G, B, ...]
    /// dengan normalisasi ke [-1, 1]: normalized = (pixel / 128.0) - 1.0
    private func imageToInputData(_ image: UIImage) -> Data? {
        guard let cgImage = image.cgImage else { return nil }

        let width = inputSize
        let height = inputSize
        let bytesPerPixel = 4 // RGBA
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow

        var rawPixels = [UInt8](repeating: 0, count: totalBytes)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                  data: &rawPixels,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: bytesPerRow,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              )
        else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Buat buffer float: [R, G, B] per pixel, dinormalisasi ke [-1, 1]
        var floatBuffer = [Float](repeating: 0, count: width * height * 3)
        var floatIndex = 0

        for i in stride(from: 0, to: totalBytes, by: bytesPerPixel) {
            let r = Float(rawPixels[i])
            let g = Float(rawPixels[i + 1])
            let b = Float(rawPixels[i + 2])

            floatBuffer[floatIndex]     = (r / 128.0) - 1.0
            floatBuffer[floatIndex + 1] = (g / 128.0) - 1.0
            floatBuffer[floatIndex + 2] = (b / 128.0) - 1.0
            floatIndex += 3
        }

        return floatBuffer.withUnsafeBytes { Data($0) }
    }
}
