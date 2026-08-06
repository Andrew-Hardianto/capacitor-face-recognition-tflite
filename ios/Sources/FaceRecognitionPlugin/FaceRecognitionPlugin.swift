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
    // Threshold liveness (HRIS production): skor live di atas ini dianggap Live
    // Dinaikkan ke 0.75 untuk mencegah foto/replay di lingkungan HRIS
    private let livenessThreshold: Float = 0.75
    // Scale factor untuk multi-scale inference (sesuai paper MiniFASNet)
    // Scale 1.0 = crop ketat, Scale 2.7 = crop lebar (tangkap artefak tepi foto/layar)
    private let antiSpoofScales: [CGFloat] = [1.0, 2.7]

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
    /// Mendeteksi semua wajah dalam gambar menggunakan Apple Vision dengan landmarks.
    /// Mengembalikan jumlah wajah, bounding box, eye open probability, dan head angle.
    ///
    /// Berguna untuk:
    ///   - Validasi awal: pastikan tepat 1 wajah sebelum extractFaceFeature
    ///   - Blink detection untuk liveness challenge di layer JS (HRIS production)
    ///   - Koordinat wajah untuk UI overlay
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

        // 2. Deteksi wajah + landmarks menggunakan Apple Vision
        // VNDetectFaceLandmarksRequest mencakup deteksi wajah + landmark (mata, hidung, dll)
        let request = VNDetectFaceLandmarksRequest { (req, error) in
            if let err = error {
                call.reject("Vision gagal mendeteksi: \(err.localizedDescription)")
                return
            }

            let results = req.results as? [VNFaceObservation] ?? []
            let imgWidth  = image.size.width
            let imgHeight = image.size.height

            // 3. Konversi bounding box dan ekstrak data landmark
            var facesArray: [[String: Any]] = []
            for face in results {
                let bb = face.boundingBox
                let x      = bb.origin.x * imgWidth
                let y      = (1 - bb.origin.y - bb.size.height) * imgHeight
                let width  = bb.size.width  * imgWidth
                let height = bb.size.height * imgHeight

                // Head roll angle dari Vision (yaw tidak tersedia)
                let roll = face.roll?.doubleValue ?? 0.0

                // Estimasi eye open probability dari Eye Aspect Ratio (EAR) landmark mata
                var leftEyeOpenProb: Double = -1.0
                var rightEyeOpenProb: Double = -1.0

                if let landmarks = face.landmarks {
                    if let leftEye = landmarks.leftEye {
                        let pts = leftEye.normalizedPoints
                        if pts.count >= 4 {
                            let ys = pts.map { $0.y }
                            let xs = pts.map { $0.x }
                            let eyeHeight = (ys.max() ?? 0) - (ys.min() ?? 0)
                            let eyeWidth  = (xs.max() ?? 0) - (xs.min() ?? 0)
                            if eyeWidth > 0 {
                                let ear = eyeHeight / eyeWidth
                                // EAR: ~0.05 = tutup, ~0.3 = buka penuh
                                leftEyeOpenProb = min(1.0, max(0.0, Double((ear - 0.05) / 0.25)))
                            }
                        }
                    }
                    if let rightEye = landmarks.rightEye {
                        let pts = rightEye.normalizedPoints
                        if pts.count >= 4 {
                            let ys = pts.map { $0.y }
                            let xs = pts.map { $0.x }
                            let eyeHeight = (ys.max() ?? 0) - (ys.min() ?? 0)
                            let eyeWidth  = (xs.max() ?? 0) - (xs.min() ?? 0)
                            if eyeWidth > 0 {
                                let ear = eyeHeight / eyeWidth
                                rightEyeOpenProb = min(1.0, max(0.0, Double((ear - 0.05) / 0.25)))
                            }
                        }
                    }
                }

                var faceDict: [String: Any] = [
                    "x":               Int(max(0, x)),
                    "y":               Int(max(0, y)),
                    "width":           Int(min(width,  imgWidth  - max(0, x))),
                    "height":          Int(min(height, imgHeight - max(0, y))),
                    "headEulerAngleY": 0.0,  // tidak tersedia di Vision Framework
                    "headEulerAngleZ": roll,
                ]
                // Hanya set jika berhasil dihitung
                if leftEyeOpenProb >= 0 { faceDict["leftEyeOpenProbability"]  = leftEyeOpenProb }
                if rightEyeOpenProb >= 0 { faceDict["rightEyeOpenProbability"] = rightEyeOpenProb }

                facesArray.append(faceDict)
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
    // checkLiveness (Anti-Spoofing) — Production HRIS
    // ============================
    /// Memeriksa apakah wajah dalam gambar adalah wajah asli (live) atau spoofing.
    ///
    /// Alur kerja (Multi-Scale MiniFASNet):
    ///   1. Decode Base64 → UIImage
    ///   2. Apple Vision → deteksi wajah → bounding box
    ///   3. Inference DUA KALI dengan scale factor berbeda (1.0x dan 2.7x)
    ///      - Scale 1.0: crop ketat (wajah saja)
    ///      - Scale 2.7: crop lebar (tangkap artefak tepi foto/layar)
    ///   4. Average kedua skor liveness
    ///   5. Threshold 0.75 untuk HRIS production
    ///   6. Return isLive, score, confidence ke JavaScript
    ///
    /// Memerlukan file 'anti_spoof.tflite' di bundle iOS.
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

            let imgWidth = image.size.width
            let imgHeight = image.size.height
            let bb = face.boundingBox

            // Bounding box wajah dalam koordinat piksel (kiri atas, tanpa margin — multi-scale yang atur)
            let baseFaceRect = CGRect(
                x: bb.origin.x * imgWidth,
                y: (1 - bb.origin.y - bb.size.height) * imgHeight,
                width: bb.size.width * imgWidth,
                height: bb.size.height * imgHeight
            )

            // 3. Multi-Scale Inference: jalankan anti-spoof untuk setiap scale
            var scaleScores: [Float] = []

            for scaleFactor in self.antiSpoofScales {
                guard let score = self.runAntiSpoofInference(
                    interpreter: antiSpoofInterpreter,
                    image: image,
                    cgImage: cgImage,
                    baseFaceRect: baseFaceRect,
                    scaleFactor: scaleFactor,
                    imgWidth: imgWidth,
                    imgHeight: imgHeight
                ) else {
                    continue
                }
                scaleScores.append(score)
            }

            guard !scaleScores.isEmpty else {
                call.reject("Gagal menjalankan anti-spoof inference")
                return
            }

            // 4. Average semua score dari berbagai scale
            let livenessScore = scaleScores.reduce(0, +) / Float(scaleScores.count)
            print("[AntiSpoof] Scale scores: \(scaleScores), Final: \(livenessScore)")

            // 5. Tentukan apakah live berdasarkan threshold (0.75 untuk HRIS production)
            let isLive = livenessScore > self.livenessThreshold

            // 6. Tentukan tingkat kepercayaan
            let confidence: String
            if livenessScore > 0.85 || livenessScore < 0.15 {
                confidence = "HIGH"
            } else if livenessScore > 0.6 || livenessScore < 0.4 {
                confidence = "MEDIUM"
            } else {
                confidence = "LOW"
            }

            // 7. Kirim hasil ke JavaScript
            call.resolve([
                "isLive": isLive,
                "score": livenessScore,
                "confidence": confidence,
            ])
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
    // Helper: Multi-Scale Anti-Spoof Inference
    // ============================
    /// Menjalankan satu inferensi anti-spoofing untuk satu scale tertentu.
    ///
    /// - Parameters:
    ///   - interpreter: TFLite interpreter anti-spoof
    ///   - image: UIImage asli (untuk rendering)
    ///   - cgImage: CGImage dari UIImage asli (untuk crop)
    ///   - baseFaceRect: Bounding box wajah dalam koordinat piksel (tanpa margin)
    ///   - scaleFactor: Faktor perbesaran area crop
    ///                  1.0 = crop ketat (wajah saja + sedikit sekitar)
    ///                  2.7 = crop lebar (sesuai paper MiniFASNet, tangkap artefak layar/foto)
    ///   - imgWidth / imgHeight: Dimensi gambar asli
    /// - Returns: Skor liveness (0.0 = spoof, 1.0 = live), nil jika gagal
    private func runAntiSpoofInference(
        interpreter: Interpreter,
        image: UIImage,
        cgImage: CGImage,
        baseFaceRect: CGRect,
        scaleFactor: CGFloat,
        imgWidth: CGFloat,
        imgHeight: CGFloat
    ) -> Float? {
        // 1. Buat bounding box menjadi BUJUR SANGKAR (square) terlebih dahulu
        // Model MiniFASNet sangat sensitif terhadap distorsi aspect ratio
        let cx = baseFaceRect.origin.x + baseFaceRect.width / 2.0
        let cy = baseFaceRect.origin.y + baseFaceRect.height / 2.0
        let side = max(baseFaceRect.width, baseFaceRect.height)

        // 2. Perbesar bujur sangkar berdasarkan scaleFactor
        let newSide = side * scaleFactor
        let expandW = newSide / 2.0
        let expandH = newSide / 2.0

        // 3. Hitung koordinat crop dan pastikan tidak keluar dari batas gambar
        let expandedX = max(0, cx - expandW)
        let expandedY = max(0, cy - expandH)
        let expandedW = min(imgWidth - expandedX, cx + expandW - expandedX)
        let expandedH = min(imgHeight - expandedY, cy + expandH - expandedY)

        let faceRect = CGRect(x: expandedX, y: expandedY, width: expandedW, height: expandedH)

        guard let croppedCgImage = cgImage.cropping(to: faceRect) else { return nil }

        // Resize ke 80x80
        let targetSize = CGSize(width: antiSpoofInputSize, height: antiSpoofInputSize)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let scaledImage = renderer.image { _ in
            UIImage(cgImage: croppedCgImage).draw(in: CGRect(origin: .zero, size: targetSize))
        }

        guard let inputData = imageToInputDataAntiSpoof(scaledImage, size: antiSpoofInputSize) else {
            return nil
        }

        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data

            var scores = [Float](repeating: 0, count: antiSpoofOutputSize)
            _ = scores.withUnsafeMutableBytes { ptr in
                outputData.copyBytes(to: ptr)
            }

            // Model TFLite biasanya sudah memiliki layer Softmax bawaan di akhir (output 0.0 - 1.0).
            // JANGAN lakukan softmax manual lagi, langsung ambil probabilitas aslinya.
            // Index 1 = Wajah Asli (Real Face)
            return scores[1]

        } catch {
            print("[AntiSpoof] Inference error (scale \(scaleFactor)): \(error)")
            return nil
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

        // Model TFLite ini diexport dengan preprocessing bawaan (kemungkinan normalization node ada di dalam graf)
        // sehingga model mengekspektasikan nilai piksel raw [0, 255].
        // Format yang diminta: B, G, R
        var floatBuffer = [Float](repeating: 0, count: width * height * 3)
        var floatIndex = 0

        for i in stride(from: 0, to: totalBytes, by: bytesPerPixel) {
            let r = Float(rawPixels[i])
            let g = Float(rawPixels[i + 1])
            let b = Float(rawPixels[i + 2])

            // BGR order (unnormalized)
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
