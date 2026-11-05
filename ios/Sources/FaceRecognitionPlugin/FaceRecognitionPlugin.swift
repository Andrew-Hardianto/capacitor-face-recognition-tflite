import AVFoundation
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
        CAPPluginMethod(name: "verifyFace", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getLivenessChallenge", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "verifyFaceSequence", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "captureLivenessSession", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getDeviceIntegrity", returnType: CAPPluginReturnPromise),
    ]

    // ============================
    // Konfigurasi — Face Recognition (MobileFaceNet)
    // ============================
    // Nama file model tanpa ekstensi (harus ada di bundle: mobile_face_net.tflite)
    private let modelName = "mobile_face_net"
    // Ukuran input model MobileFaceNet
    private let inputSize = 112
    // Ukuran output embedding — terverifikasi dari tensor output model: shape (1, 192)
    private let embeddingSize = 192
    // Threshold cosine similarity untuk face match
    // CATATAN: default ini BELUM dikalibrasi dengan data sendiri (Fase 5)
    private let matchThreshold: Double = 0.75

    // ============================
    // Konfigurasi — Anti-Spoofing (Silent-Face / minivision-ai)
    // ============================
    // Dua model dengan crop scale masing-masing (sesuai referensi):
    //   anti_spoof_2_7.tflite = spoof_model_scale_2_7.tflite (crop 2.7x)
    //   anti_spoof_4_0.tflite = spoof_model_scale_4_0.tflite (crop 4.0x)
    // Source: https://github.com/shubham0204/OnDevice-Face-Recognition-Android
    private let antiSpoofModelName27 = "anti_spoof_2_7"
    private let antiSpoofModelName40 = "anti_spoof_4_0"
    // Nama file legacy — fallback untuk bundle lama yang masih memakai anti_spoof.tflite
    private let antiSpoofModelNameLegacy = "anti_spoof"
    // MiniFASNet menggunakan input 80x80
    private let antiSpoofInputSize = 80
    // Output model: 3 kelas → [print_spoof, live, replay_spoof]
    // Index 1 = LIVE (wajah asli). Urutan ini hasil verifikasi empiris terhadap
    // model referensi, BUKAN dari komentar kode lama.
    private let antiSpoofOutputSize = 3
    // Crop scale per model (sesuai referensi Silent-Face; TIDAK ada scale 1.0)
    private let cropScale27: CGFloat = 2.7
    private let cropScale40: CGFloat = 4.0

    // ============================
    // Threshold — SATU sumber kebenaran (Fase 1)
    // ============================
    // Keputusan live (Fase 2.4):
    //   p = softmax(out27) + softmax(out40)        (rentang 0..2)
    //   label = argmax(p); liveScore = p[1]/2; spoofScore = max(p[0],p[2])/2
    //   isLive = label==1 && liveScore >= liveThreshold && spoofScore < spoofVeto
    private let liveThreshold: Float = 0.80
    private let spoofVeto: Float = 0.20
    // Versi threshold — disertakan di setiap hasil agar perubahan semantik terlacak
    private let thresholdVersion = "liveness-v2"
    private let matchThresholdVersion = "match-v1"

    // ============================
    // Quality gate (Fase 3.1) — semua bisa di-override via config
    // ============================
    private let minFrameSide = 480
    private let minFaceRatio: Float = 0.25
    private let edgeMargin: Float = 0.05
    private let maxPoseDeg: Float = 20
    private let blurMin: Float = 50           // variance of Laplacian (perlu kalibrasi Fase 5)
    private let exposureMin: Float = 60
    private let exposureMax: Float = 200
    // Konsistensi identitas antar frame (verifyFaceSequence)
    private let identityConsistency: Double = 0.8
    // Active challenge
    private let challengeYawMin: Float = 15
    private let challengeFrontMax: Float = 15
    private let challengeMinFrontalFrames = 2

    // ============================
    // Pertahanan berlapis (Fase 4)
    // ============================
    // Rate limiting percobaan sesi liveness (anti brute-force threshold/serangan beruntun)
    private let maxSessionAttempts = 10
    private let sessionAttemptWindow: TimeInterval = 5 * 60
    private var sessionAttempts: [Date] = []

    /// Output debug hanya diizinkan pada build DEBUG (Fase 4).
    private var debugAllowed: Bool {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }

    private func checkRateLimit() -> Bool {
        let now = Date()
        sessionAttempts.removeAll { now.timeIntervalSince($0) > sessionAttemptWindow }
        if sessionAttempts.count >= maxSessionAttempts { return false }
        sessionAttempts.append(now)
        return true
    }

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

    // TFLite Interpreter — Anti-Spoofing dua model (lazy, nullable — tidak crash jika model tidak ada)
    // Interpreter TIDAK thread-safe → semua akses wajib melalui lock yang sesuai
    private lazy var antiSpoof27: Interpreter? = {
        return loadAntiSpoofModel(names: [antiSpoofModelName27, antiSpoofModelNameLegacy])
    }()

    private lazy var antiSpoof40: Interpreter? = {
        return loadAntiSpoofModel(names: [antiSpoofModelName40])
    }()

    private let antiSpoof27Lock = NSLock()
    private let antiSpoof40Lock = NSLock()
    private let embeddingLock = NSLock()

    private func loadAntiSpoofModel(names: [String]) -> Interpreter? {
        for name in names {
            if let modelPath = modelBundle.path(forResource: name, ofType: "tflite") {
                do {
                    let interpreter = try Interpreter(modelPath: modelPath)
                    try interpreter.allocateTensors()
                    return interpreter
                } catch {
                    print("[FaceRecognitionPlugin] ERROR: Gagal membuat antiSpoofInterpreter (\(name)): \(error)")
                }
            }
        }
        print("[FaceRecognitionPlugin] INFO: Model anti-spoofing \(names) tidak ditemukan di bundle. checkLiveness tidak akan tersedia.")
        return nil
    }

    // ============================
    // extractFaceFeature
    // ============================
    @objc func extractFaceFeature(_ call: CAPPluginCall) {
        // Pastikan interpreter berhasil dimuat
        guard interpreter != nil else {
            call.reject("Model TFLite gagal dimuat. Pastikan file '\(modelName).tflite' ada di bundle iOS.")
            return
        }

        // 1. Decode Base64 ke UIImage (orientasi sudah dinormalisasi)
        guard let image = decodeOrientedImage(call.getString("imageBase64") ?? ""),
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
                  let face = self.pickLargestFace(results)
            else {
                call.reject("Tidak ada wajah yang terdeteksi")
                return
            }

            // 3. Konversi bounding box Vision (koordinat dari kiri bawah → kiri atas)
            let faceRect = self.pixelRect(from: face, imgWidth: image.size.width, imgHeight: image.size.height)

            // 4. Ekstrak embedding dari frame dan bbox yang sama
            guard let embeddings = self.extractEmbedding(cgImage: cgImage, faceRect: faceRect) else {
                call.reject("Gagal menjalankan inference face recognition")
                return
            }

            call.resolve(["embedding": embeddings])
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

        guard let cosineSimilarity = cosineSimilarity(vec1, vec2) else {
            call.reject("Data vektor ada yang bernilai nol semua, cek apakah embedding berhasil")
            return
        }

        let isMatch = cosineSimilarity > matchThreshold
        let similarityPercentage = max(0, cosineSimilarity * 100.0)

        call.resolve([
            "isMatch": isMatch,
            "similarityPercentage": similarityPercentage,
            "score": cosineSimilarity,
            "threshold": matchThreshold,
            "thresholdVersion": matchThresholdVersion,
        ])
    }

    // ============================
    // detectFaces
    // ============================
    /// Mendeteksi semua wajah dalam gambar menggunakan Apple Vision dengan landmarks.
    /// Mengembalikan jumlah wajah, bounding box, euler angles (X/Y/Z, derajat),
    /// dan eye open probability.
    ///
    /// CATATAN: sudut dari Vision dikonversi dari radian ke derajat agar konsisten
    /// dengan dokumentasi API dan platform Android.
    @objc func detectFaces(_ call: CAPPluginCall) {
        // 1. Decode Base64 ke UIImage (orientasi sudah dinormalisasi)
        guard let image = decodeOrientedImage(call.getString("imageBase64") ?? ""),
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

                // Konversi bounding box Vision (kiri-bawah, normalized) → piksel (kiri-atas)
                let x      = bb.origin.x * imgWidth
                let y      = (1 - bb.origin.y - bb.size.height) * imgHeight
                let width  = bb.size.width  * imgWidth
                let height = bb.size.height * imgHeight

                // Head angles dari Vision (radian) → derajat
                let angles = self.headAnglesDeg(face)

                // Estimasi eye open probability dari Eye Aspect Ratio (EAR) landmark mata
                var leftEyeOpenProb: Double = -1.0
                var rightEyeOpenProb: Double = -1.0

                if let landmarks = face.landmarks {
                    if let leftEye = landmarks.leftEye {
                        leftEyeOpenProb = self.eyeOpenProbability(from: leftEye.normalizedPoints)
                    }
                    if let rightEye = landmarks.rightEye {
                        rightEyeOpenProb = self.eyeOpenProbability(from: rightEye.normalizedPoints)
                    }
                }

                var faceDict: [String: Any] = [
                    "x":               Int(max(0, x)),
                    "y":               Int(max(0, y)),
                    "width":           Int(min(width,  imgWidth  - max(0, x))),
                    "height":          Int(min(height, imgHeight - max(0, y))),
                    "headEulerAngleX": angles.pitch,   // pitch (atas-bawah)
                    "headEulerAngleY": angles.yaw,     // yaw (kiri-kanan)
                    "headEulerAngleZ": angles.roll,    // roll (miring)
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

        // 6. Eksekusi Vision Request
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            call.reject("Gagal menjalankan Vision Request: \(error.localizedDescription)")
        }
    }

    /// Menghitung Eye Aspect Ratio (EAR) dari normalized landmark points mata.
    /// EAR = (tinggi atas-bawah rata-rata) / lebar mata
    /// Hasil dinormalisasi ke probabilitas 0.0-1.0 (0=pasti merem, 1=pasti terbuka)
    private func eyeOpenProbability(from points: [CGPoint]) -> Double {
        guard points.count >= 6 else { return 1.0 }
        // Eye landmark order (Vision): 0=kiri, 1=kiri-atas, 2=kanan-atas, 3=kanan, 4=kanan-bawah, 5=kiri-bawah
        let eyeWidth  = hypot(points[3].x - points[0].x, points[3].y - points[0].y)
        guard eyeWidth > 0 else { return 1.0 }

        let height1 = hypot(points[1].x - points[5].x, points[1].y - points[5].y)
        let height2 = hypot(points[2].x - points[4].x, points[2].y - points[4].y)
        let ear = Double((height1 + height2) / (2.0 * eyeWidth))

        // EAR biasanya 0.2-0.4 saat terbuka, < 0.1 saat tertutup
        // Normalisasi ke 0.0-1.0: probability = clamp(ear / 0.3, 0, 1)
        return min(1.0, max(0.0, ear / 0.3))
    }

    // ============================
    // checkLiveness — wrapper backward-compatible (Fase 1.3)
    // ============================
    /// Memeriksa apakah wajah dalam gambar adalah wajah asli (live) atau spoofing.
    ///
    /// CATATAN PERUBAHAN SEMANTIK (liveness-v2):
    ///   Versi lama menghitung rata-rata skor live antar scale (1.0 & 2.7) tanpa
    ///   argmax kelas. Versi ini memakai pipeline referensi Silent-Face:
    ///   dua model (crop 2.7 & 4.0), softmax dijumlahkan, argmax 3 kelas, plus
    ///   syarat liveScore dan veto spoofScore. Field "score" kini dipetakan ke
    ///   liveScore pasca-argmax — NILAINYA TIDAK SEBANDING dengan versi lama.
    ///
    /// Response lama {isLive, score, confidence} tetap ada, ditambah field baru:
    ///   label, liveScore, printScore, replayScore, thresholdVersion.
    @objc func checkLiveness(_ call: CAPPluginCall) {
        // Periksa apakah model anti-spoofing tersedia
        guard antiSpoof27 != nil, antiSpoof40 != nil else {
            call.reject(
                "Model anti-spoofing tidak ditemukan. " +
                "Letakkan file '\(antiSpoofModelName27).tflite' dan '\(antiSpoofModelName40).tflite' di bundle iOS Xcode project. " +
                "Download model dari: https://github.com/shubham0204/OnDevice-Face-Recognition-Android"
            )
            return
        }

        // 1. Decode Base64 ke UIImage (orientasi sudah dinormalisasi)
        guard let image = decodeOrientedImage(call.getString("imageBase64") ?? ""),
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
                  let face = self.pickLargestFace(results)
            else {
                call.reject("Tidak ada wajah yang terdeteksi untuk pemeriksaan liveness")
                return
            }

            let baseFaceRect = self.pixelRect(from: face, imgWidth: image.size.width, imgHeight: image.size.height)

            // 3. Pipeline anti-spoof (dua model, crop shift, argmax + veto)
            guard let result = self.runAntiSpoofPipeline(
                cgImage: cgImage,
                baseFaceRect: baseFaceRect,
                imgWidth: image.size.width,
                imgHeight: image.size.height
            ) else {
                call.reject("Gagal menjalankan anti-spoof inference")
                return
            }

            // 4. Tentukan tingkat kepercayaan (kompatibel dengan response lama)
            let confidence: String
            if result.liveScore > 0.85 || result.liveScore < 0.15 {
                confidence = "HIGH"
            } else if result.liveScore > 0.6 || result.liveScore < 0.4 {
                confidence = "MEDIUM"
            } else {
                confidence = "LOW"
            }

            // 5. Kirim hasil ke JavaScript
            call.resolve([
                "isLive": result.isLive,
                "score": result.liveScore,       // dipetakan ke liveScore
                "confidence": confidence,
                "label": result.label,
                "liveScore": result.liveScore,
                "printScore": result.printScore,
                "replayScore": result.replayScore,
                "thresholdVersion": self.thresholdVersion,
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
    // verifyFace — API terpadu (Fase 3.2, menutup TOCTOU)
    // ============================
    /// Liveness + quality gate + embedding dari FRAME DAN BBOX YANG SAMA.
    /// Embedding hanya dikembalikan jika isLive = true.
    ///
    /// options:
    ///   imageBase64: string
    ///   referenceEmbedding?: number[]   → jika ada, sekaligus compare (match)
    ///   config?: {
    ///     liveThreshold?, spoofVeto?, minFaceRatio?, minFrameSide?, edgeMargin?,
    ///     maxPose?, blurMin?, exposureMin?, exposureMax?, yawSign?, debug?
    ///   }
    @objc func verifyFace(_ call: CAPPluginCall) {
        guard antiSpoof27 != nil, antiSpoof40 != nil, interpreter != nil else {
            call.reject("Model liveness atau face recognition tidak ditemukan di bundle")
            return
        }

        guard let image = decodeOrientedImage(call.getString("imageBase64") ?? ""),
              let cgImage = image.cgImage
        else {
            call.reject("Gagal decode Base64 menjadi gambar")
            return
        }

        let config = GateConfig.from(call.getObject("config"), debugAllowed: debugAllowed)
        let referenceEmbedding = call.getArray("referenceEmbedding", Double.self)

        let request = VNDetectFaceRectanglesRequest { [weak self] (req, error) in
            guard let self = self else { return }

            if let err = error {
                call.reject("Vision gagal mendeteksi: \(err.localizedDescription)")
                return
            }

            let results = req.results as? [VNFaceObservation] ?? []

            // ---- Quality gate (Fase 3.1) ----
            let quality = self.runQualityGate(
                cgImage: cgImage, faces: results,
                imgWidth: image.size.width, imgHeight: image.size.height, config: config)
            if !quality.passed {
                call.resolve([
                    "isLive": false,
                    "thresholdVersion": self.thresholdVersion,
                    "liveness": NSNull(),
                    "quality": quality.toJson(),
                ])
                return
            }

            guard let face = self.pickLargestFace(results) else {
                call.reject("Tidak ada wajah yang terdeteksi")
                return
            }
            let baseFaceRect = self.pixelRect(from: face, imgWidth: image.size.width, imgHeight: image.size.height)

            // ---- Anti-spoof (Fase 2) ----
            guard let liveness = self.runAntiSpoofPipeline(
                cgImage: cgImage,
                baseFaceRect: baseFaceRect,
                imgWidth: image.size.width,
                imgHeight: image.size.height,
                config: config
            ) else {
                call.reject("Gagal menjalankan anti-spoof inference")
                return
            }

            var ret: [String: Any] = [
                "isLive": liveness.isLive,
                "thresholdVersion": self.thresholdVersion,
                "liveness": liveness.toJson(),
                "quality": quality.toJson(),
            ]

            if config.debug {
                ret["debug"] = liveness.debugToJson()
            }

            // ---- Embedding HANYA jika live, dari frame & bbox yang sama ----
            if liveness.isLive {
                guard let embeddings = self.extractEmbedding(cgImage: cgImage, faceRect: baseFaceRect) else {
                    call.reject("Gagal mengekstrak embedding dari frame yang sama")
                    return
                }
                ret["embedding"] = embeddings

                if let ref = referenceEmbedding {
                    guard let score = self.cosineSimilarity(ref, embeddings.map { Double($0) }) else {
                        call.reject("referenceEmbedding tidak valid (norma nol)")
                        return
                    }
                    ret["match"] = [
                        "isMatch": score > self.matchThreshold,
                        "score": score,
                        "threshold": self.matchThreshold,
                        "thresholdVersion": self.matchThresholdVersion,
                    ]
                }
            }

            call.resolve(ret)
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            call.reject("Gagal menjalankan Vision Request untuk verifyFace: \(error.localizedDescription)")
        }
    }

    // ============================
    // Active challenge (Fase 3.3)
    // ============================
    /// Mengeluarkan challenge acak untuk sesi liveness aktif.
    /// Aplikasi menampilkan instruksi ke user, lalu memanggil verifyFaceSequence.
    @objc func getLivenessChallenge(_ call: CAPPluginCall) {
        let challenge = Int.random(in: 0...1) == 0 ? "LOOK_LEFT" : "LOOK_RIGHT"
        call.resolve(["challenge": challenge])
    }

    /// Verifikasi sesi multi-frame dengan active challenge (Fase 3.3).
    ///
    /// options:
    ///   images: string[]            → ≥ 5 frame dalam ~1,5 detik
    ///   challenge: 'LOOK_LEFT' | 'LOOK_RIGHT'
    ///   referenceEmbedding?: number[]
    ///   config?: GateConfig (sama dengan verifyFace)
    ///
    /// Aturan lulus:
    ///   - SEMUA frame lolos quality gate
    ///   - median liveScore ≥ liveThreshold DAN mayoritas frame isLive (voting,
    ///     BUKAN "semua frame harus live" — lihat catatan BPCER di plan)
    ///   - identitas konsisten antar frame (cosine ≥ 0.8 terhadap frame pertama)
    ///   - challenge terpenuhi: yaw menyimpang ≥ 15° ke arah yang diminta, lalu
    ///     minimal 2 frame kembali mendekati frontal (|yaw| < 15°)
    ///
    /// CATATAN KONVENSI TANDA YAW: default saat ini LOOK_LEFT = yaw negatif.
    /// Jika di perangkat ternyata terbalik, set config.yawSign = -1.
    /// Verifikasi tanda ini bagian dari kalibrasi Fase 5.
    @objc func verifyFaceSequence(_ call: CAPPluginCall) {
        guard antiSpoof27 != nil, antiSpoof40 != nil, interpreter != nil else {
            call.reject("Model liveness atau face recognition tidak ditemukan di bundle")
            return
        }
        guard checkRateLimit() else {
            call.reject("Terlalu banyak percobaan sesi liveness. Coba lagi dalam beberapa menit.", "RATE_LIMITED")
            return
        }

        guard let images = call.getArray("images", String.self), images.count >= 2 else {
            call.reject("images minimal berisi 2 frame (disarankan ≥ 5)")
            return
        }
        let challenge = call.getString("challenge") ?? ""
        guard challenge == "LOOK_LEFT" || challenge == "LOOK_RIGHT" else {
            call.reject("challenge harus LOOK_LEFT atau LOOK_RIGHT (dari getLivenessChallenge)")
            return
        }

        let config = GateConfig.from(call.getObject("config"), debugAllowed: debugAllowed)
        let referenceEmbedding = call.getArray("referenceEmbedding", Double.self)

        guard let frames = processFrames(base64Frames: images, config: config) else {
            call.reject("Gagal memproses frame (decode / Vision)")
            return
        }

        let result = finishSequence(frames: frames, config: config,
                                    challenge: challenge, referenceEmbedding: referenceEmbedding)
        call.resolve(result)
    }

    /// Proses frame sekuensial (VNImageRequestHandler bersifat sinkron).
    /// Mengembalikan nil jika ada frame gagal di-decode atau Vision error.
    private func processFrames(base64Frames: [String], config: GateConfig) -> [FrameResult]? {
        var frames: [FrameResult] = []

        for (index, base64) in base64Frames.enumerated() {
            guard let image = decodeOrientedImage(base64), let cgImage = image.cgImage else {
                print("[verifyFaceSequence] Frame \(index) gagal di-decode")
                return nil
            }

            let request = VNDetectFaceRectanglesRequest { [weak self] (req, error) in
                guard let self = self else { return }

                var fr = FrameResult()
                fr.index = index

                if let err = error {
                    fr.quality = QualityResult.fail(reason: "NO_FACE", faceRatio: 0, blur: 0)
                    print("[verifyFaceSequence] Frame \(index) gagal: \(err.localizedDescription)")
                } else {
                    let results = req.results as? [VNFaceObservation] ?? []
                    if results.isEmpty {
                        fr.quality = QualityResult.fail(reason: "NO_FACE", faceRatio: 0, blur: 0)
                    } else {
                        fr.quality = self.runQualityGate(
                            cgImage: cgImage, faces: results,
                            imgWidth: image.size.width, imgHeight: image.size.height, config: config)
                        if fr.quality.passed, let face = self.pickLargestFace(results) {
                            let baseFaceRect = self.pixelRect(from: face, imgWidth: image.size.width, imgHeight: image.size.height)
                            let angles = self.headAnglesDeg(face)
                            fr.yaw = angles.yaw
                            fr.pitch = angles.pitch
                            fr.roll = angles.roll
                            fr.liveness = self.runAntiSpoofPipeline(
                                cgImage: cgImage,
                                baseFaceRect: baseFaceRect,
                                imgWidth: image.size.width,
                                imgHeight: image.size.height,
                                config: config)
                            fr.embedding = self.extractEmbedding(cgImage: cgImage, faceRect: baseFaceRect)
                        }
                    }
                }
                frames.append(fr)
            }

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("[verifyFaceSequence] Vision error frame \(index): \(error)")
                return nil
            }
        }

        return frames
    }

    /// Agregasi sesi (median + voting, konsistensi identitas, challenge).
    /// Mengembalikan dictionary hasil — TIDAK menyelesaikan call, agar bisa
    /// dipakai ulang oleh captureLivenessSession (Fase 4).
    private func finishSequence(frames: [FrameResult], config: GateConfig,
                                challenge: String, referenceEmbedding: [Double]?) -> [String: Any] {
        var firstEmbedding: [Float]? = nil
        var identityConsistent = true
        var failureReason: String? = nil

        for fr in frames {
            // SEMUA frame harus lolos quality gate
            if !fr.quality.passed {
                failureReason = "QUALITY_FAILED"
                break
            }

            // Konsistensi identitas terhadap frame pertama
            if let embedding = fr.embedding {
                if firstEmbedding == nil {
                    firstEmbedding = embedding
                } else if let cos = cosineSimilarity(firstEmbedding!.map { Double($0) },
                                                     embedding.map { Double($0) }) {
                    if cos < identityConsistency {
                        identityConsistent = false
                    }
                }
            }
        }

        // ---- Agregasi (median + voting, BUKAN all-live) ----
        let liveScores: [Float] = frames.map { $0.liveness?.liveScore ?? 0 }
        let liveVotes = frames.filter { $0.liveness?.isLive == true }.count
        let yawValues = frames.filter { $0.quality.passed }.map { $0.yaw }
        let maxYaw = yawValues.max() ?? 0
        let minYaw = yawValues.min() ?? 0
        let frontalFrames = yawValues.filter { abs($0) < challengeFrontMax }.count

        let medianLiveScore = median(liveScores)
        let requiredVotes = Int(ceil(Double(frames.count) * 0.6))

        // Challenge: deviasi yaw ≥ challengeYawMin ke arah yang diminta
        let yawSign = Float(config.yawSign)
        var challengeSatisfied: Bool
        if challenge == "LOOK_LEFT" {
            challengeSatisfied = (minYaw * yawSign) <= -challengeYawMin
        } else {
            challengeSatisfied = (maxYaw * yawSign) >= challengeYawMin
        }
        challengeSatisfied = challengeSatisfied && frontalFrames >= challengeMinFrontalFrames

        let passed = failureReason == nil
            && medianLiveScore >= config.liveThreshold
            && liveVotes >= requiredVotes
            && identityConsistent
            && challengeSatisfied

        var ret = buildSequenceResult(
            passed: passed, reason: nil, challenge: challenge,
            medianLiveScore: medianLiveScore, liveVotes: liveVotes, frameCount: frames.count,
            maxYaw: maxYaw, minYaw: minYaw, frontalFrames: frontalFrames,
            identityConsistent: identityConsistent)

        if passed, let ref = referenceEmbedding, let first = firstEmbedding {
            if let score = cosineSimilarity(ref, first.map { Double($0) }) {
                ret["match"] = [
                    "isMatch": score > matchThreshold,
                    "score": score,
                    "threshold": matchThreshold,
                    "thresholdVersion": matchThresholdVersion,
                ]
            }
        }

        if !passed {
            var reason = failureReason
            if reason == nil {
                if !identityConsistent {
                    reason = "IDENTITY_INCONSISTENT"
                } else if !challengeSatisfied {
                    reason = "CHALLENGE_FAILED"
                } else if liveVotes < requiredVotes || medianLiveScore < config.liveThreshold {
                    reason = "NOT_LIVE_ENOUGH"
                }
            }
            ret["reason"] = reason ?? "UNKNOWN"
        }

        return ret
    }

    private func buildSequenceResult(passed: Bool, reason: String?, challenge: String,
                                     medianLiveScore: Float, liveVotes: Int, frameCount: Int,
                                     maxYaw: Float, minYaw: Float, frontalFrames: Int,
                                     identityConsistent: Bool) -> [String: Any] {
        var ret: [String: Any] = [
            "passed": passed,
            "challenge": challenge,
            "medianLiveScore": medianLiveScore,
            "liveVotes": liveVotes,
            "frameCount": frameCount,
            "maxYaw": maxYaw,
            "minYaw": minYaw,
            "frontalFrames": frontalFrames,
            "identityConsistent": identityConsistent,
            "thresholdVersion": thresholdVersion,
        ]
        if let reason = reason {
            ret["reason"] = reason
        }
        return ret
    }

    // ============================
    // Native capture (Fase 4) — captureLivenessSession
    // ============================
    /// Sesi liveness native (anti injeksi frame): kamera dibuka oleh view controller
    /// plugin, frame diambil dan disimpan ke file privat TANPA melewati JS, lalu
    /// pipeline verifikasi (quality gate + liveness + challenge) dijalankan terhadap
    /// frame tersebut. Yang kembali ke JS hanya HASIL — bukan frame mentah.
    ///
    /// options:
    ///   challenge?: 'LOOK_LEFT' | 'LOOK_RIGHT'  → default acak
    ///   referenceEmbedding?: number[]
    ///   config?: LivenessConfig
    ///
    /// Response = VerifyFaceSequenceResult + { native, deviceIntegrity, auditId }
    @objc func captureLivenessSession(_ call: CAPPluginCall) {
        guard antiSpoof27 != nil, antiSpoof40 != nil, interpreter != nil else {
            call.reject("Model liveness atau face recognition tidak ditemukan di bundle")
            return
        }
        guard checkRateLimit() else {
            call.reject("Terlalu banyak percobaan sesi liveness. Coba lagi dalam beberapa menit.", "RATE_LIMITED")
            return
        }

        var challenge = call.getString("challenge") ?? ""
        if challenge != "LOOK_LEFT" && challenge != "LOOK_RIGHT" {
            challenge = Int.random(in: 0...1) == 0 ? "LOOK_LEFT" : "LOOK_RIGHT"
        }

        let config = GateConfig.from(call.getObject("config"), debugAllowed: debugAllowed)
        let referenceEmbedding = call.getArray("referenceEmbedding", Double.self)

        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                guard let self = self else { return }
                guard granted else {
                    call.reject("Izin kamera ditolak")
                    return
                }
                guard let presenter = self.bridge?.viewController else {
                    call.reject("ViewController tidak tersedia untuk membuka kamera")
                    return
                }

                let captureVC = LivenessCaptureViewController(
                    challenge: challenge, frameCount: 5, interval: 0.3, warmup: 1.2
                ) { [weak self] result in
                    guard let self = self else { return }
                    switch result {
                    case .failure(let error):
                        call.reject("Capture gagal: \(error.localizedDescription)")
                    case .success(let urls):
                        self.processCapturedSession(
                            call: call, urls: urls, challenge: challenge,
                            config: config, referenceEmbedding: referenceEmbedding)
                    }
                }
                captureVC.modalPresentationStyle = .fullScreen
                presenter.present(captureVC, animated: true)
            }
        }
    }

    /// Verifikasi frame hasil native capture + jejak audit terenkripsi.
    private func processCapturedSession(call: CAPPluginCall, urls: [URL], challenge: String,
                                        config: GateConfig, referenceEmbedding: [Double]?) {
        var base64Frames: [String] = []
        var frameBytes: [Data] = []
        for url in urls {
            guard let data = try? Data(contentsOf: url) else {
                call.reject("Gagal membaca frame capture")
                return
            }
            frameBytes.append(data)
            base64Frames.append(data.base64EncodedString())
            // Frame sementara dihapus — salinan terenkripsi disimpan oleh AuditStore
            try? FileManager.default.removeItem(at: url)
        }

        guard let frames = processFrames(base64Frames: base64Frames, config: config) else {
            call.reject("Gagal memproses frame capture (decode / Vision)")
            return
        }

        var ret = finishSequence(frames: frames, config: config,
                                 challenge: challenge, referenceEmbedding: referenceEmbedding)
        ret["native"] = true
        ret["deviceIntegrity"] = DeviceIntegrity.snapshot()

        if let json = try? JSONSerialization.data(withJSONObject: ret, options: [.sortedKeys]) {
            do {
                ret["auditId"] = try AuditStore.storeSession(json: json, frames: frameBytes)
            } catch {
                print("[FaceRecognition] Audit store gagal: \(error)")
                ret["auditId"] = NSNull()
            }
        }

        call.resolve(ret)
    }

    // ============================
    // Device integrity (Fase 4) — sinyal risiko, bukan bukti
    // ============================
    @objc func getDeviceIntegrity(_ call: CAPPluginCall) {
        call.resolve(["deviceIntegrity": DeviceIntegrity.snapshot()])
    }

    // ============================
    // Quality Gate (Fase 3.1)
    // ============================
    private func runQualityGate(cgImage: CGImage, faces: [VNFaceObservation],
                                imgWidth: CGFloat, imgHeight: CGFloat,
                                config: GateConfig) -> QualityResult {
        guard let largest = pickLargestFace(faces) else {
            return QualityResult.fail(reason: "NO_FACE", faceRatio: 0, blur: 0)
        }

        // MULTIPLE_FACES: tolak jika ada > 1 wajah yang cukup besar
        let largestArea = max(0.0001, largest.boundingBox.width * largest.boundingBox.height)
        var significantFaces = 0
        for f in faces {
            let area = f.boundingBox.width * f.boundingBox.height
            if area >= largestArea * 0.3 { significantFaces += 1 }
        }
        if significantFaces > 1 {
            return QualityResult.fail(reason: "MULTIPLE_FACES", faceRatio: 0, blur: 0)
        }

        // LOW_RESOLUTION
        if min(imgWidth, imgHeight) < CGFloat(config.minFrameSide) {
            return QualityResult.fail(reason: "LOW_RESOLUTION", faceRatio: 0, blur: 0)
        }

        // FACE_TOO_SMALL
        let faceRect = pixelRect(from: largest, imgWidth: imgWidth, imgHeight: imgHeight)
        let faceRatio = Float(faceRect.width / imgWidth)
        if faceRatio < config.minFaceRatio {
            return QualityResult.fail(reason: "FACE_TOO_SMALL", faceRatio: faceRatio, blur: 0)
        }

        // FACE_AT_EDGE: bbox menyentuh tepi frame (margin 5%)
        let marginX = imgWidth * CGFloat(config.edgeMargin)
        let marginY = imgHeight * CGFloat(config.edgeMargin)
        if faceRect.minX <= marginX || faceRect.minY <= marginY
                || faceRect.maxX >= imgWidth - marginX || faceRect.maxY >= imgHeight - marginY {
            return QualityResult.fail(reason: "FACE_AT_EDGE", faceRatio: faceRatio, blur: 0)
        }

        // BAD_POSE
        let angles = headAnglesDeg(largest)
        if abs(angles.yaw) > config.maxPose || abs(angles.pitch) > config.maxPose
                || abs(angles.roll) > config.maxPose {
            return QualityResult.fail(reason: "BAD_POSE", faceRatio: faceRatio, blur: 0)
        }

        // BLURRY + BAD_LIGHTING dari crop wajah
        guard let faceCrop = cgImage.cropping(to: faceRect) else {
            return QualityResult.fail(reason: "BLURRY", faceRatio: faceRatio, blur: 0)
        }
        let blur = varianceOfLaplacian(faceCrop)
        let exposure = meanLuminance(faceCrop)
        if blur < config.blurMin {
            return QualityResult.fail(reason: "BLURRY", faceRatio: faceRatio, blur: blur)
        }
        if exposure < config.exposureMin || exposure > config.exposureMax {
            return QualityResult.fail(reason: "BAD_LIGHTING", faceRatio: faceRatio, blur: blur)
        }

        return QualityResult.pass(faceRatio: faceRatio, blur: blur)
    }

    // ============================
    // Anti-Spoof Pipeline (Fase 2) — port referensi Silent-Face
    // ============================
    private struct AntiSpoofResult {
        var label: String
        var liveScore: Float
        var printScore: Float
        var replayScore: Float
        var spoofScore: Float
        var isLive: Bool
        var probs27: [Float]
        var probs40: [Float]

        func toJson() -> [String: Any] {
            return [
                "label": label,
                "liveScore": liveScore,
                "printScore": printScore,
                "replayScore": replayScore,
            ]
        }

        func debugToJson() -> [String: Any] {
            return [
                "probs27": probs27.map { Double($0) },
                "probs40": probs40.map { Double($0) },
            ]
        }
    }

    /// Pipeline referensi:
    ///   crop 2.7x (shift) → model 2.7 ─┐
    ///                                    ├→ sum softmax → argmax + veto → keputusan
    ///   crop 4.0x (shift) → model 4.0 ─┘
    private func runAntiSpoofPipeline(
        cgImage: CGImage,
        baseFaceRect: CGRect,
        imgWidth: CGFloat,
        imgHeight: CGFloat,
        config: GateConfig? = nil
    ) -> AntiSpoofResult? {
        guard let interp27 = antiSpoof27, let interp40 = antiSpoof40 else { return nil }
        let cfg = config ?? GateConfig.from(nil)

        guard let logits27 = runAntiSpoofModel(
                interpreter: interp27, lock: antiSpoof27Lock, cgImage: cgImage,
                baseFaceRect: baseFaceRect, cropScale: cropScale27,
                imgWidth: imgWidth, imgHeight: imgHeight),
              let logits40 = runAntiSpoofModel(
                interpreter: interp40, lock: antiSpoof40Lock, cgImage: cgImage,
                baseFaceRect: baseFaceRect, cropScale: cropScale40,
                imgWidth: imgWidth, imgHeight: imgHeight)
        else {
            return nil
        }

        let probs27 = softmax(logits27)
        let probs40 = softmax(logits40)

        // p = softmax(out27) + softmax(out40) → rentang 0..2
        var p = [Float](repeating: 0, count: antiSpoofOutputSize)
        for i in 0..<antiSpoofOutputSize {
            p[i] = probs27[i] + probs40[i]
        }

        // argmax 3 kelas: [print, live, replay]
        var argmax = 0
        for i in 1..<p.count where p[i] > p[argmax] {
            argmax = i
        }

        var result = AntiSpoofResult(
            label: argmax == 1 ? "LIVE" : (argmax == 0 ? "PRINT" : "REPLAY"),
            liveScore: p[1] / 2,
            printScore: p[0] / 2,
            replayScore: p[2] / 2,
            spoofScore: max(p[0], p[2]) / 2,
            isLive: false,
            probs27: probs27,
            probs40: probs40
        )
        result.isLive = argmax == 1
            && result.liveScore >= cfg.liveThreshold
            && result.spoofScore < cfg.spoofVeto
        return result
    }

    /// Menjalankan satu inferensi anti-spoofing untuk satu model dengan crop scale-nya.
    /// Interpreter TFLite TIDAK thread-safe → akses disinkronkan per model.
    ///
    /// - Returns: raw logits (3 kelas), atau nil jika gagal
    private func runAntiSpoofModel(
        interpreter: Interpreter,
        lock: NSLock,
        cgImage: CGImage,
        baseFaceRect: CGRect,
        cropScale: CGFloat,
        imgWidth: CGFloat,
        imgHeight: CGFloat
    ) -> [Float]? {
        // Crop sesuai referensi (_get_new_box): scale diturunkan jika tidak muat,
        // w & h terpisah, SHIFT bukan clip, tanpa square-ifikasi
        let faceRect = getNewBox(baseFaceRect, cropScale: cropScale, imgWidth: imgWidth, imgHeight: imgHeight)

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
            lock.lock()
            defer { lock.unlock() }
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data

            var scores = [Float](repeating: 0, count: antiSpoofOutputSize)
            _ = scores.withUnsafeMutableBytes { ptr in
                outputData.copyBytes(to: ptr)
            }

            // Model mengeluarkan raw logits — softmax dilakukan di pemanggil
            return scores
        } catch {
            print("[AntiSpoof] Inference error (scale \(cropScale)): \(error)")
            return nil
        }
    }

    /// Port logika crop referensi (_get_new_box):
    ///   scale diturunkan jika crop tidak muat di gambar,
    ///   lebar dan tinggi diskalakan TERPISAH (tanpa square),
    ///   jika keluar batas maka DIGESER (shift), bukan dipotong (clip).
    private func getNewBox(_ bounds: CGRect, cropScale: CGFloat,
                           imgWidth: CGFloat, imgHeight: CGFloat) -> CGRect {
        let boxW = bounds.width
        let boxH = bounds.height
        let cx = bounds.midX
        let cy = bounds.midY

        // Turunkan scale jika crop tidak muat
        let maxScaleW = (imgWidth - 1) / boxW
        let maxScaleH = (imgHeight - 1) / boxH
        let s = min(cropScale, min(maxScaleW, maxScaleH))

        let newW = boxW * s
        let newH = boxH * s

        var left = cx - newW / 2
        var top = cy - newH / 2
        var right = left + newW
        var bottom = top + newH

        // SHIFT, bukan clip
        if left < 0 { right -= left; left = 0 }
        if top < 0 { bottom -= top; top = 0 }
        if right > imgWidth - 1 { left -= (right - (imgWidth - 1)); right = imgWidth - 1 }
        if bottom > imgHeight - 1 { top -= (bottom - (imgHeight - 1)); bottom = imgHeight - 1 }

        // Penjagaan akhir untuk kasus ekstrem
        left = max(0, left)
        top = max(0, top)

        let x = (left).rounded()
        let y = (top).rounded()
        let w = max(1, min(imgWidth - x, (right - left).rounded()))
        let h = max(1, min(imgHeight - y, (bottom - top).rounded()))
        return CGRect(x: x, y: y, width: w, height: h)
    }

    // ============================
    // Face Recognition helpers
    // ============================

    /// Ekstrak embedding MobileFaceNet dari frame dan bbox yang diberikan.
    /// Crop memakai logika yang SAMA dengan pipeline enrollment (bbox langsung,
    /// di-clip ke batas gambar) agar embedding enroll dan verify konsisten.
    private func extractEmbedding(cgImage: CGImage, faceRect: CGRect) -> [Float]? {
        guard let interpreter = interpreter else { return nil }

        let clamped = faceRect.intersection(CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        guard !clamped.isNull, clamped.width > 0, clamped.height > 0,
              let croppedCgImage = cgImage.cropping(to: clamped) else { return nil }

        let targetSize = CGSize(width: inputSize, height: inputSize)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let scaledImage = renderer.image { _ in
            UIImage(cgImage: croppedCgImage).draw(in: CGRect(origin: .zero, size: targetSize))
        }

        guard let inputData = imageToInputData(scaledImage, size: inputSize) else { return nil }

        do {
            embeddingLock.lock()
            defer { embeddingLock.unlock() }
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data

            var embeddings = [Float](repeating: 0, count: outputData.count / MemoryLayout<Float>.size)
            _ = embeddings.withUnsafeMutableBytes { ptr in
                outputData.copyBytes(to: ptr)
            }
            return embeddings
        } catch {
            print("[FaceRecognition] extractEmbedding error: \(error)")
            return nil
        }
    }

    private func pickLargestFace(_ faces: [VNFaceObservation]) -> VNFaceObservation? {
        return faces.max { a, b in
            (a.boundingBox.width * a.boundingBox.height) < (b.boundingBox.width * b.boundingBox.height)
        }
    }

    /// Konversi bounding box Vision (kiri-bawah, normalized) → piksel (kiri-atas)
    private func pixelRect(from face: VNFaceObservation,
                           imgWidth: CGFloat, imgHeight: CGFloat) -> CGRect {
        let bb = face.boundingBox
        return CGRect(
            x: bb.origin.x * imgWidth,
            y: (1 - bb.origin.y - bb.size.height) * imgHeight,
            width: bb.size.width * imgWidth,
            height: bb.size.height * imgHeight
        )
    }

    /// Head angles dari VNFaceObservation (radian) → derajat.
    /// Konvensi sama dengan ML Kit Android: X = pitch, Y = yaw, Z = roll.
    private func headAnglesDeg(_ face: VNFaceObservation) -> (pitch: Float, yaw: Float, roll: Float) {
        let rad2deg = Float(180.0 / .pi)
        let pitch = face.pitch.map { Float(truncating: $0) } ?? 0
        let yaw = face.yaw.map { Float(truncating: $0) } ?? 0
        let roll = face.roll.map { Float(truncating: $0) } ?? 0
        return (pitch: pitch * rad2deg, yaw: yaw * rad2deg, roll: roll * rad2deg)
    }

    private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double? {
        guard a.count == b.count, !a.isEmpty else { return nil }
        var dot = 0.0, norm1 = 0.0, norm2 = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            norm1 += a[i] * a[i]
            norm2 += b[i] * b[i]
        }
        guard norm1 > 0, norm2 > 0 else { return nil }
        return dot / (sqrt(norm1) * sqrt(norm2))
    }

    // ============================
    // Quality gate helpers — blur, exposure
    // ============================

    /// Variance of Laplacian pada grayscale — ukuran blur (semakin tinggi semakin tajam).
    private func varianceOfLaplacian(_ cgImage: CGImage) -> Float {
        let size = 160 // ukuran kerja tetap agar skor konsisten antar frame
        guard let gray = grayscalePixels(cgImage, size: size) else { return 0 }

        // Kernel Laplacian 3x3: [0 1 0; 1 -4 1; 0 1 0]
        var sum: Double = 0, sumSq: Double = 0
        var count = 0
        for y in 1..<(size - 1) {
            for x in 1..<(size - 1) {
                let idx = y * size + x
                let lap = Double(gray[idx - size]) + Double(gray[idx + size])
                        + Double(gray[idx - 1]) + Double(gray[idx + 1])
                        - 4 * Double(gray[idx])
                sum += lap
                sumSq += lap * lap
                count += 1
            }
        }
        guard count > 0 else { return 0 }
        let mean = sum / Double(count)
        return Float(sumSq / Double(count) - mean * mean)
    }

    /// Mean luminance crop wajah (0..255) — untuk gate exposure.
    private func meanLuminance(_ cgImage: CGImage) -> Float {
        let size = 64
        guard let gray = grayscalePixels(cgImage, size: size) else { return 0 }
        let sum = gray.reduce(0.0) { $0 + Double($1) }
        return Float(sum / Double(gray.count))
    }

    /// Render CGImage ke grayscale (8-bit) berukuran size x size.
    private func grayscalePixels(_ cgImage: CGImage, size: Int) -> [UInt8]? {
        var pixels = [UInt8](repeating: 0, count: size * size)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.linearGray),
              let context = CGContext(
                  data: &pixels,
                  width: size,
                  height: size,
                  bitsPerComponent: 8,
                  bytesPerRow: size,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.none.rawValue
              )
        else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        return pixels
    }

    // ============================
    // Image / preprocessing helpers
    // ============================

    /// Decode Base64 → UIImage dengan orientasi sudah dinormalisasi (EXIF/UIImage.orientation).
    private func decodeOrientedImage(_ imageBase64: String) -> UIImage? {
        guard let data = Data(base64Encoded: imageBase64, options: .ignoreUnknownCharacters),
              let image = UIImage(data: data)
        else {
            return nil
        }
        guard image.imageOrientation != .up else { return image }

        // Render ulang agar piksel benar-benar tersetel orientasinya
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
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
    // Helper: UIImage → ByteData TFLite — Anti-Spoofing (Silent-Face)
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

        // Model Silent-Face mengekspektasikan nilai piksel raw [0, 255] dalam urutan BGR
        // (normalisasi ada di dalam graf model). Jangan paksa ke [-1, 1].
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

    private func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[mid] }
        return (sorted[mid - 1] + sorted[mid]) / 2
    }

    // ============================
    // Konfigurasi quality gate yang bisa di-override per panggilan
    // ============================
    private struct GateConfig {
        var liveThreshold: Float = 0.80
        var spoofVeto: Float = 0.20
        var minFrameSide: Int = 480
        var minFaceRatio: Float = 0.25
        var edgeMargin: Float = 0.05
        var maxPose: Float = 20
        var blurMin: Float = 50
        var exposureMin: Float = 60
        var exposureMax: Float = 200
        var yawSign: Int = 1
        var debug: Bool = false

        static func from(_ config: JSObject?) -> GateConfig {
            return from(config, debugAllowed: false)
        }

        /// - Parameter debugAllowed: hanya true pada build DEBUG — output debug
        ///   tidak pernah dikembalikan dari build rilis (Fase 4)
        static func from(_ config: JSObject?, debugAllowed: Bool) -> GateConfig {
            var c = GateConfig()
            c.debug = false
            guard let config = config else { return c }
            if let v = config["liveThreshold"] as? Double { c.liveThreshold = Float(v) }
            if let v = config["spoofVeto"] as? Double { c.spoofVeto = Float(v) }
            if let v = config["minFrameSide"] as? Int { c.minFrameSide = v }
            if let v = config["minFaceRatio"] as? Double { c.minFaceRatio = Float(v) }
            if let v = config["edgeMargin"] as? Double { c.edgeMargin = Float(v) }
            if let v = config["maxPose"] as? Double { c.maxPose = Float(v) }
            if let v = config["blurMin"] as? Double { c.blurMin = Float(v) }
            if let v = config["exposureMin"] as? Double { c.exposureMin = Float(v) }
            if let v = config["exposureMax"] as? Double { c.exposureMax = Float(v) }
            if let v = config["yawSign"] as? Int { c.yawSign = v }
            if let v = config["debug"] as? Bool { c.debug = v && debugAllowed }
            return c
        }
    }

    private struct QualityResult {
        var passed: Bool
        var reason: String?
        var faceRatio: Float
        var blur: Float

        static func pass(faceRatio: Float, blur: Float) -> QualityResult {
            return QualityResult(passed: true, reason: nil, faceRatio: faceRatio, blur: blur)
        }

        static func fail(reason: String, faceRatio: Float, blur: Float) -> QualityResult {
            return QualityResult(passed: false, reason: reason, faceRatio: faceRatio, blur: blur)
        }

        func toJson() -> [String: Any] {
            var o: [String: Any] = [
                "passed": passed,
                "faceRatio": faceRatio,
                "blur": blur,
            ]
            if let reason = reason {
                o["reason"] = reason
            }
            return o
        }
    }

    private struct FrameResult {
        var index: Int = 0
        var yaw: Float = 0
        var pitch: Float = 0
        var roll: Float = 0
        var quality: QualityResult = QualityResult.fail(reason: "NO_FACE", faceRatio: 0, blur: 0)
        var liveness: AntiSpoofResult?
        var embedding: [Float]?
    }
}
