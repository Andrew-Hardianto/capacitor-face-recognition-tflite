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
        CAPPluginMethod(name: "checkLiveness", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "detectFaces", returnType: CAPPluginReturnPromise),
    ]

    // ============================
    // Konfigurasi — Face Recognition
    // ============================
    // Nama file model tanpa ekstensi (harus ada di bundle: mobile_face_net.tflite)
    private let modelName = "mobile_face_net"
    // Ukuran input model MobileFaceNet
    private let inputSize = 112
    // Ukuran output embedding
    private let embeddingSize = 128

    // ============================
    // Konfigurasi — Anti-Spoofing (Liveness Detection)
    // ============================
    // Nama file model anti-spoofing tanpa ekstensi (harus ada di bundle: anti_spoof.tflite)
    // Model: MiniFASNetV1 dari shubham0204/OnDevice-Face-Recognition-Android
    // Source: https://github.com/shubham0204/OnDevice-Face-Recognition-Android
    private let antiSpoofModelName = "anti_spoof"
    // MiniFASNet menggunakan input 80x80 (bukan 128x128)
    private let antiSpoofInputSize = 80
    // Output model: 3 kelas softmax → [live_score, print_spoof_score, replay_spoof_score]
    // Index 0 = live (wajah asli)
    private let antiSpoofOutputSize = 3
    // Threshold liveness: skor live (index 0) di atas ini dianggap Live
    private let livenessThreshold: Float = 0.5

    /// Menentukan bundle yang tepat untuk mencari model TFLite:
    /// - Swift Package Manager (SPM): gunakan `Bundle.module` (bundle plugin itu sendiri)
    /// - CocoaPods: gunakan `Bundle.main` (model dikopi ke app bundle)
    private var modelBundle: Bundle {
        // Bundle.module hanya tersedia ketika plugin dipakai via SPM
        // Saat CocoaPods, Swift Package resources tidak tersedia — fallback ke Bundle.main
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle.main
        #endif
    }

    // TFLite Interpreter — Face Recognition (lazy agar dimuat sekali)
    private lazy var interpreter: Interpreter? = {
        // modelBundle otomatis memilih Bundle.module (SPM) atau Bundle.main (CocoaPods)
        guard let modelPath = modelBundle.path(forResource: modelName, ofType: "tflite") else {
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

    // TFLite Interpreter — Anti-Spoofing (lazy, nullable — tidak crash jika model tidak ada)
    private lazy var antiSpoofInterpreter: Interpreter? = {
        // modelBundle otomatis memilih Bundle.module (SPM) atau Bundle.main (CocoaPods)
        guard let modelPath = modelBundle.path(forResource: antiSpoofModelName, ofType: "tflite") else {
            print("[FaceRecognitionPlugin] INFO: File '\(antiSpoofModelName).tflite' tidak ditemukan di bundle. checkLiveness tidak akan tersedia.")
            return nil
        }
        do {
            let interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
            return interpreter
        } catch {
            print("[FaceRecognitionPlugin] ERROR: Gagal membuat antiSpoofInterpreter: \(error)")
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
            guard let inputData = self.imageToInputData(scaledImage, size: self.inputSize) else {
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
    // detectFaces
    // ============================
    /// Mendeteksi semua wajah dalam gambar menggunakan Apple Vision.
    /// Mengembalikan jumlah wajah dan bounding box tiap wajah (koordinat piksel).
    ///
    /// Berguna untuk validasi awal:
    ///   - Pastikan tepat 1 wajah terdeteksi sebelum extractFaceFeature
    ///   - Dapatkan koordinat wajah untuk UI feedback / overlay
    @objc func detectFaces(_ call: CAPPluginCall) {
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
        let request = VNDetectFaceRectanglesRequest { (req, error) in
            if let err = error {
                call.reject("Vision gagal mendeteksi: \(err.localizedDescription)")
                return
            }

            let results = req.results as? [VNFaceObservation] ?? []
            let imgWidth  = image.size.width
            let imgHeight = image.size.height

            // 3. Konversi setiap bounding box dari Vision (kiri-bawah) ke koordinat piksel (kiri-atas)
            var facesArray: [[String: Any]] = []
            for face in results {
                let bb = face.boundingBox
                let x      = bb.origin.x * imgWidth
                let y      = (1 - bb.origin.y - bb.size.height) * imgHeight
                let width  = bb.size.width  * imgWidth
                let height = bb.size.height * imgHeight

                facesArray.append([
                    "x":      Int(max(0, x)),
                    "y":      Int(max(0, y)),
                    "width":  Int(min(width,  imgWidth  - max(0, x))),
                    "height": Int(min(height, imgHeight - max(0, y))),
                ])
            }

            call.resolve([
                "count": results.count,
                "faces": facesArray,
            ])
        }

        // 4. Eksekusi Vision Request
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            call.reject("Gagal menjalankan Vision Request: \(error.localizedDescription)")
        }
    }

    // ============================
    // checkLiveness (Anti-Spoofing)
    // ============================
    /// Memeriksa apakah wajah dalam gambar adalah wajah asli (live) atau spoofing.
    ///
    /// Alur kerja:
    ///   1. Decode Base64 → UIImage
    ///   2. Apple Vision → deteksi wajah → crop region wajah (dengan margin)
    ///   3. Resize ke 128×128 (input MiniFASNet)
    ///   4. TFLite inference model anti-spoofing
    ///   5. Output: [spoof_score, live_score] → softmax → livenessScore
    ///   6. Return isLive, score, confidence ke JavaScript
    ///
    /// Memerlukan file 'anti_spoof.tflite' di bundle iOS.
    /// Download: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    @objc func checkLiveness(_ call: CAPPluginCall) {
        // Periksa apakah model anti-spoofing tersedia
        guard let antiSpoofInterpreter = antiSpoofInterpreter else {
            call.reject(
                "Model anti-spoofing tidak ditemukan. " +
                "Letakkan file '\(antiSpoofModelName).tflite' di bundle iOS Xcode project. " +
                "Download model dari: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
            )
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
                call.reject("Tidak ada wajah yang terdeteksi untuk pemeriksaan liveness")
                return
            }

            // 3. Konversi bounding box Vision (koordinat dari kiri bawah → kiri atas)
            let imgWidth = image.size.width
            let imgHeight = image.size.height
            let bb = face.boundingBox

            // Tambahkan margin 20% agar konteks kulit sekitar wajah ikut tertangkap
            // (membantu model mendeteksi tepi foto / layar)
            let marginW = bb.size.width * 0.2
            let marginH = bb.size.height * 0.2

            let faceRect = CGRect(
                x: max(0, bb.origin.x * imgWidth - marginW * imgWidth),
                y: max(0, (1 - bb.origin.y - bb.size.height) * imgHeight - marginH * imgHeight),
                width: min(imgWidth, (bb.size.width + 2 * marginW) * imgWidth),
                height: min(imgHeight, (bb.size.height + 2 * marginH) * imgHeight)
            )

            // 4. Crop wajah (dengan margin)
            guard let croppedCgImage = cgImage.cropping(to: faceRect) else {
                call.reject("Gagal memotong gambar wajah untuk liveness check")
                return
            }

            // 5. Resize ke antiSpoofInputSize x antiSpoofInputSize (80x80)
            let targetSize = CGSize(width: self.antiSpoofInputSize, height: self.antiSpoofInputSize)
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
            let scaledImage = renderer.image { _ in
                UIImage(cgImage: croppedCgImage).draw(in: CGRect(origin: .zero, size: targetSize))
            }

            // 6. Konversi UIImage ke Data input TFLite (normalisasi ke [0, 1] — MiniFASNet standard)
            guard let inputData = self.imageToInputDataAntiSpoof(scaledImage, size: self.antiSpoofInputSize) else {
                call.reject("Gagal mengkonversi gambar ke format TFLite input anti-spoofing")
                return
            }

            // 7. Set input tensor & jalankan inference
            do {
                try antiSpoofInterpreter.copy(inputData, toInputAt: 0)
                try antiSpoofInterpreter.invoke()

                // 8. Ambil output tensor — shape: [1, 3] → [live_score, print_spoof, replay_spoof]
                let outputTensor = try antiSpoofInterpreter.output(at: 0)
                let outputData = outputTensor.data

                // 9. Konversi output bytes ke [Float]
                var scores = [Float](repeating: 0, count: 3)
                _ = scores.withUnsafeMutableBytes { ptr in
                    outputData.copyBytes(to: ptr)
                }

                // 10. Terapkan softmax manual karena model mungkin output raw logits
                // Original repo (minivision-ai): Label 1 adalah Real Face
                // Index 0 = spoof, Index 1 = live, Index 2 = spoof
                let probs = self.softmax(scores)
                let livenessScore = probs[1]

                // 11. Tentukan apakah live berdasarkan threshold
                let isLive = livenessScore > self.livenessThreshold

                // 12. Tentukan tingkat kepercayaan
                let confidence: String
                if livenessScore > 0.85 || livenessScore < 0.15 {
                    confidence = "HIGH"
                } else if livenessScore > 0.6 || livenessScore < 0.4 {
                    confidence = "MEDIUM"
                } else {
                    confidence = "LOW"
                }

                // 13. Kirim hasil ke JavaScript
                call.resolve([
                    "isLive": isLive,
                    "score": livenessScore,
                    "confidence": confidence,
                ])
            } catch {
                call.reject("TFLite anti-spoofing inference gagal: \(error.localizedDescription)")
            }
        }

        // Eksekusi Vision Request
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            call.reject("Gagal menjalankan Vision Request untuk liveness: \(error.localizedDescription)")
        }
    }

    // ============================
    // Helper: UIImage → ByteData TFLite — Face Recognition (MobileFaceNet)
    // Normalisasi ke [-1, 1]
    // ============================
    /// Konversi UIImage ke Data berformat [R, G, B, R, G, B, ...]
    /// dengan normalisasi ke [-1, 1]: normalized = (pixel / 128.0) - 1.0
    ///
    /// - Parameter image: UIImage yang sudah di-resize ke ukuran yang sesuai
    /// - Parameter size: Ukuran input model (112 untuk face recognition)
    private func imageToInputData(_ image: UIImage, size: Int) -> Data? {
        guard let cgImage = image.cgImage else { return nil }

        let width = size
        let height = size
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

    // ============================
    // Helper: UIImage → ByteData TFLite — Anti-Spoofing (MiniFASNet)
    // Normalisasi ke [0, 1] — sesuai training MiniFASNet
    // ============================
    private func imageToInputDataAntiSpoof(_ image: UIImage, size: Int) -> Data? {
        guard let cgImage = image.cgImage else { return nil }

        let width = size
        let height = size
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

        // MiniFASNet dilatih dengan OpenCV (BGR), normalisasi HILANGKAN (model asli pakai [0, 255])
        // Urutan channel: B, G, R (BUKAN R, G, B)
        var floatBuffer = [Float](repeating: 0, count: width * height * 3)
        var floatIndex = 0

        for i in stride(from: 0, to: totalBytes, by: bytesPerPixel) {
            let r = Float(rawPixels[i])
            let g = Float(rawPixels[i + 1])
            let b = Float(rawPixels[i + 2])

            floatBuffer[floatIndex]     = b
            floatBuffer[floatIndex + 1] = g
            floatBuffer[floatIndex + 2] = r
            floatIndex += 3
        }

        return floatBuffer.withUnsafeBytes { Data($0) }
    }

    // ============================
    // Softmax — konversi raw logits ke probability
    // ============================
    private func softmax(_ logits: [Float]) -> [Float] {
        let maxVal = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxVal) }
        let sumExps = exps.reduce(0, +)
        return exps.map { $0 / sumExps }
    }
}
