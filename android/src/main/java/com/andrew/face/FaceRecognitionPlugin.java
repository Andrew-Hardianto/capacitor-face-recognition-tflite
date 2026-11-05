package com.andrew.face;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.media.ExifInterface;
import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

@CapacitorPlugin(
    name = "FaceRecognition",
    permissions = @Permission(strings = { android.Manifest.permission.CAMERA }, alias = "camera")
)
public class FaceRecognitionPlugin extends Plugin {

    // ============================
    // Konfigurasi — Face Recognition (MobileFaceNet)
    // ============================
    // Nama file model di folder android/src/main/assets/
    private static final String MODEL_FILE = "mobile_face_net.tflite";
    // Ukuran input model (MobileFaceNet = 112x112)
    private static final int INPUT_SIZE = 112;
    // Jumlah channel warna (RGB = 3)
    private static final int PIXEL_CHANNELS = 3;
    // Ukuran output embedding model (model ini output 192 dimensi, bukan 128)
    // Terverifikasi dari tensor output model: shape (1, 192)
    private static final int EMBEDDING_SIZE = 192;
    // Ukuran byte per float
    private static final int BYTES_PER_FLOAT = 4;
    // Threshold cosine similarity untuk face match
    // CATATAN: default ini BELUM dikalibrasi dengan data sendiri (Fase 5)
    private static final double MATCH_THRESHOLD = 0.75;

    // ============================
    // Konfigurasi — Anti-Spoofing (Silent-Face / minivision-ai)
    // ============================
    // Dua model dengan crop scale masing-masing (sesuai referensi):
    //   anti_spoof_2_7.tflite = spoof_model_scale_2_7.tflite (crop 2.7x)
    //   anti_spoof_4_0.tflite = spoof_model_scale_4_0.tflite (crop 4.0x)
    // Source: https://github.com/shubham0204/OnDevice-Face-Recognition-Android
    private static final String ANTI_SPOOF_MODEL_FILE_27 = "anti_spoof_2_7.tflite";
    private static final String ANTI_SPOOF_MODEL_FILE_40 = "anti_spoof_4_0.tflite";
    // Nama file legacy — fallback untuk instalasi lama yang masih memakai anti_spoof.tflite
    private static final String ANTI_SPOOF_MODEL_FILE_LEGACY = "anti_spoof.tflite";
    // MiniFASNet menggunakan input 80x80
    private static final int ANTI_SPOOF_INPUT_SIZE = 80;
    // Output model: 3 kelas → [print_spoof, live, replay_spoof]
    // Index 1 = LIVE (wajah asli). Urutan ini hasil verifikasi empiris terhadap
    // model referensi, BUKAN dari komentar kode lama.
    private static final int ANTI_SPOOF_OUTPUT_SIZE = 3;
    // Crop scale per model (sesuai referensi Silent-Face; TIDAK ada scale 1.0)
    private static final float CROP_SCALE_27 = 2.7f;
    private static final float CROP_SCALE_40 = 4.0f;

    // ============================
    // Threshold — SATU sumber kebenaran (Fase 1)
    // ============================
    // Keputusan live (Fase 2.4):
    //   p = softmax(out27) + softmax(out40)        (rentang 0..2)
    //   label = argmax(p); liveScore = p[1]/2; spoofScore = max(p[0],p[2])/2
    //   isLive = label==1 && liveScore >= LIVE_THRESHOLD && spoofScore < SPOOF_VETO
    private static final float LIVE_THRESHOLD = 0.80f;
    private static final float SPOOF_VETO = 0.20f;
    // Versi threshold — disertakan di setiap hasil agar perubahan semantik terlacak
    private static final String THRESHOLD_VERSION = "liveness-v2";
    private static final String MATCH_THRESHOLD_VERSION = "match-v1";

    // ============================
    // Quality gate (Fase 3.1) — semua bisa di-override via config
    // ============================
    private static final int MIN_FRAME_SIDE = 480;          // sisi terpendek frame (px)
    private static final float MIN_FACE_RATIO = 0.25f;      // faceWidth / frameWidth
    private static final float EDGE_MARGIN = 0.05f;         // bbox tidak menyentuh tepi frame
    private static final float MAX_POSE_DEG = 20f;          // |yaw|, |pitch|, |roll|
    private static final float BLUR_MIN = 50f;              // variance of Laplacian (perlu kalibrasi Fase 5)
    private static final float EXPOSURE_MIN = 60f;          // mean luminance wajah
    private static final float EXPOSURE_MAX = 200f;
    // Konsistensi identitas antar frame (verifyFaceSequence)
    private static final float IDENTITY_CONSISTENCY = 0.8f; // cosine terhadap frame pertama
    // Active challenge
    private static final float CHALLENGE_YAW_MIN = 15f;     // deviasi minimum yang diwajibkan
    private static final float CHALLENGE_FRONT_MAX = 15f;   // kembali mendekati frontal
    private static final int CHALLENGE_MIN_FRONTAL_FRAMES = 2;

    // ============================
    // Pertahanan berlapis (Fase 4)
    // ============================
    // Rate limiting percobaan sesi liveness (anti brute-force threshold/serangan beruntun)
    private static final int MAX_SESSION_ATTEMPTS = 10;
    private static final long SESSION_ATTEMPT_WINDOW_MS = 5 * 60 * 1000L;
    private static final List<Long> sessionAttempts = new ArrayList<>();
    // Native capture — captureLivenessSession
    private static final int REQUEST_LIVENESS_SESSION = 4321;
    private static final int NATIVE_FRAME_COUNT = 5;
    private static final long NATIVE_FRAME_INTERVAL_MS = 300L;
    private static final long NATIVE_WARMUP_MS = 1200L;

    // ============================
    // ML Kit Face Detectors
    // ============================
    // Untuk liveness/quality gate — mode akurat (Fase 2/3)
    private final FaceDetectorOptions livenessOptions = new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .build();
    private final FaceDetector livenessDetector = FaceDetection.getClient(livenessOptions);

    // Untuk detectFaces (dengan klasifikasi: euler angle + eye open probability)
    private final FaceDetectorOptions metaOptions = new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .build();
    private final FaceDetector metaDetector = FaceDetection.getClient(metaOptions);

    // TFLite Interpreter — Face Recognition
    private Interpreter tfliteInterpreter;

    // TFLite Interpreter — Anti-Spoofing (nullable, mungkin tidak ada modelnya)
    // TIDAK thread-safe → semua akses wajib melalui synchronized (lihat runAntiSpoofModel)
    private Interpreter antiSpoof27;
    private Interpreter antiSpoof40;
    private final Object antiSpoof27Lock = new Object();
    private final Object antiSpoof40Lock = new Object();

    // ============================
    // Inisialisasi Plugin
    // ============================
    @Override
    public void load() {
        super.load();

        // Muat model face recognition
        try {
            tfliteInterpreter = new Interpreter(loadModelFile(MODEL_FILE));
        } catch (IOException e) {
            // Model gagal dimuat - pastikan file MODEL_FILE ada di android/src/main/assets/
            e.printStackTrace();
        }

        // Muat model anti-spoofing (opsional — tidak crash jika tidak ada)
        antiSpoof27 = loadAntiSpoofModel(ANTI_SPOOF_MODEL_FILE_27, ANTI_SPOOF_MODEL_FILE_LEGACY);
        antiSpoof40 = loadAntiSpoofModel(ANTI_SPOOF_MODEL_FILE_40, null);
    }

    private Interpreter loadAntiSpoofModel(String fileName, String fallbackName) {
        try {
            return new Interpreter(loadModelFile(fileName));
        } catch (IOException e) {
            if (fallbackName != null) {
                try {
                    return new Interpreter(loadModelFile(fallbackName));
                } catch (IOException ignored) {
                    // jatuh ke bawah
                }
            }
            // Model anti-spoofing tidak ditemukan — checkLiveness akan return error informatif
            return null;
        }
    }

    /**
     * Memuat file model TFLite dari folder assets secara memory-mapped
     * untuk performa yang lebih baik.
     */
    private MappedByteBuffer loadModelFile(String fileName) throws IOException {
        AssetFileDescriptor fileDescriptor = getContext().getAssets().openFd(fileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // ============================
    // Plugin Methods
    // ============================

    @PluginMethod
    public void extractFaceFeature(PluginCall call) {
        // Periksa apakah TFLite berhasil dimuat
        if (tfliteInterpreter == null) {
            call.reject("Model TFLite gagal dimuat. Pastikan file '" + MODEL_FILE + "' ada di folder android/src/main/assets/");
            return;
        }

        String imageBase64 = call.getString("imageBase64");
        if (imageBase64 == null) {
            call.reject("Base64 gambar tidak boleh kosong");
            return;
        }

        try {
            // 1. Decode Base64 ke Bitmap (sudah dinormalisasi orientasi EXIF)
            Bitmap bitmap = decodeOrientedBitmap(imageBase64);
            if (bitmap == null) {
                call.reject("Gagal decode Base64 menjadi gambar");
                return;
            }
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            // 2. Deteksi wajah menggunakan ML Kit
            livenessDetector.process(image)
                .addOnSuccessListener(faces -> {
                    if (faces.isEmpty()) {
                        call.reject("Tidak ada wajah yang terdeteksi");
                        return;
                    }

                    try {
                        // 3. Pilih wajah dengan bbox terbesar (konsisten dengan pipeline liveness)
                        Face face = pickLargestFace(faces);
                        Rect bounds = face.getBoundingBox();

                        // 4. Ekstrak embedding dari frame dan bbox yang sama
                        float[] embeddings = extractEmbedding(bitmap, bounds);
                        if (embeddings == null) {
                            call.reject("Gagal menjalankan inference face recognition");
                            return;
                        }

                        // 5. Kirim hasil embedding ke JavaScript
                        JSONArray jsonArray = new JSONArray();
                        for (float val : embeddings) {
                            jsonArray.put((double) val);
                        }
                        JSObject ret = new JSObject();
                        ret.put("embedding", jsonArray);
                        call.resolve(ret);
                    } catch (Exception e) {
                        call.reject("Gagal mem-parsing embedding", e);
                    }
                })
                .addOnFailureListener(e -> call.reject("ML Kit gagal mendeteksi wajah", e));

        } catch (Exception e) {
            call.reject("Error memproses gambar", e);
        }
    }

    @PluginMethod
    public void compareFaces(PluginCall call) {
        try {
            org.json.JSONArray vec1Json = call.getArray("vector1");
            org.json.JSONArray vec2Json = call.getArray("vector2");

            if (vec1Json == null || vec2Json == null || vec1Json.length() != vec2Json.length()) {
                call.reject("Vector tidak valid atau panjangnya tidak sama");
                return;
            }

            double cosineSimilarity = cosineSimilarity(vec1Json, vec2Json);
            if (Double.isNaN(cosineSimilarity)) {
                call.reject("Data vektor ada yang bernilai nol semua, cek apakah embedding berhasil");
                return;
            }

            boolean isMatch = cosineSimilarity > MATCH_THRESHOLD;

            // Persentase kemiripan yang lebih akurat
            double similarityPercentage = Math.max(0, cosineSimilarity * 100.0);

            JSObject ret = new JSObject();
            ret.put("isMatch", isMatch);
            ret.put("similarityPercentage", similarityPercentage);
            ret.put("score", cosineSimilarity);
            ret.put("threshold", MATCH_THRESHOLD);
            ret.put("thresholdVersion", MATCH_THRESHOLD_VERSION);

            call.resolve(ret);
        } catch (Exception e) {
            call.reject("Gagal membandingkan wajah", e);
        }
    }

    /**
     * Mendeteksi semua wajah dalam gambar menggunakan ML Kit dengan mode klasifikasi penuh.
     * Mengembalikan jumlah wajah, bounding box, euler angles (X/Y/Z), dan eye open probability.
     *
     * Berguna untuk validasi kualitas frame kamera sebelum extractFaceFeature:
     *   - Pastikan tepat 1 wajah (faces.count === 1)
     *   - Cek kepala lurus: Math.abs(headEulerAngleY) < 15 && Math.abs(headEulerAngleZ) < 15
     *   - Cek mata terbuka: leftEyeOpenProbability > 0.4 && rightEyeOpenProbability > 0.4
     *
     * @param call PluginCall dengan parameter "imageBase64" (string Base64)
     */
    @PluginMethod
    public void detectFaces(PluginCall call) {
        String imageBase64 = call.getString("imageBase64");
        if (imageBase64 == null) {
            call.reject("Base64 gambar tidak boleh kosong");
            return;
        }

        try {
            // 1. Decode Base64 ke Bitmap (sudah dinormalisasi orientasi EXIF)
            Bitmap bitmap = decodeOrientedBitmap(imageBase64);
            if (bitmap == null) {
                call.reject("Gagal decode Base64 menjadi gambar");
                return;
            }
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            // 2. Jalankan metaDetector (CLASSIFICATION_MODE_ALL aktif)
            metaDetector.process(image)
                .addOnSuccessListener(faces -> {
                    try {
                        // 3. Susun array bounding box + euler angle + eye probability
                        JSONArray facesArray = new JSONArray();
                        for (Face face : faces) {
                            Rect bounds = face.getBoundingBox();

                            // Bounding box — klem agar tidak keluar batas bitmap
                            int x      = Math.max(0, bounds.left);
                            int y      = Math.max(0, bounds.top);
                            int width  = Math.min(bounds.width(), bitmap.getWidth() - x);
                            int height = Math.min(bounds.height(), bitmap.getHeight() - y);

                            // Eye open probability — hanya tersedia dengan CLASSIFICATION_MODE_ALL
                            // Mengembalikan null jika tidak bisa dihitung (wajah terlalu kecil, dll)
                            Float leftEyeProb  = face.getLeftEyeOpenProbability();
                            Float rightEyeProb = face.getRightEyeOpenProbability();

                            JSObject faceObj = new JSObject();
                            faceObj.put("x", x);
                            faceObj.put("y", y);
                            faceObj.put("width", width);
                            faceObj.put("height", height);

                            // Head euler angles (derajat, sesuai dokumentasi ML Kit)
                            faceObj.put("headEulerAngleX", face.getHeadEulerAngleX()); // pitch (atas-bawah)
                            faceObj.put("headEulerAngleY", face.getHeadEulerAngleY()); // yaw (kiri-kanan)
                            faceObj.put("headEulerAngleZ", face.getHeadEulerAngleZ()); // roll (miring)

                            // Eye open probability (0.0 = tutup, 1.0 = buka penuh)
                            if (leftEyeProb  != null) faceObj.put("leftEyeOpenProbability",  leftEyeProb.doubleValue());
                            if (rightEyeProb != null) faceObj.put("rightEyeOpenProbability", rightEyeProb.doubleValue());

                            facesArray.put(faceObj);
                        }

                        JSObject ret = new JSObject();
                        ret.put("count", faces.size());
                        ret.put("faces", facesArray);
                        call.resolve(ret);
                    } catch (Exception e) {
                        call.reject("Gagal mem-parsing hasil deteksi wajah", e);
                    }
                })
                .addOnFailureListener(e -> call.reject("ML Kit gagal mendeteksi wajah", e));

        } catch (Exception e) {
            call.reject("Error memproses gambar", e);
        }
    }

    // ============================
    // checkLiveness — wrapper backward-compatible (Fase 1.3)
    // ============================
    /**
     * Memeriksa apakah wajah dalam gambar adalah wajah asli (live) atau spoofing.
     *
     * CATATAN PERUBAHAN SEMANTIK (liveness-v2):
     *   Versi lama menghitung rata-rata skor live antar scale (1.0 & 2.7) tanpa
     *   argmax kelas. Versi ini memakai pipeline referensi Silent-Face:
     *   dua model (crop 2.7 & 4.0), softmax dijumlahkan, argmax 3 kelas, plus
     *   syarat liveScore dan veto spoofScore. Field "score" kini dipetakan ke
     *   liveScore pasca-argmax — NILAINYA TIDAK SEBANDING dengan versi lama.
     *
     * Response lama {isLive, score, confidence} tetap ada, ditambah field baru:
     *   label, liveScore, printScore, replayScore, thresholdVersion.
     */
    @PluginMethod
    public void checkLiveness(PluginCall call) {
        // Periksa apakah model anti-spoofing berhasil dimuat
        if (antiSpoof27 == null || antiSpoof40 == null) {
            call.reject(
                "Model anti-spoofing tidak ditemukan. " +
                "Letakkan file '" + ANTI_SPOOF_MODEL_FILE_27 + "' dan '" + ANTI_SPOOF_MODEL_FILE_40 +
                "' di folder android/src/main/assets/. " +
                "Download model dari: https://github.com/shubham0204/OnDevice-Face-Recognition-Android"
            );
            return;
        }

        String imageBase64 = call.getString("imageBase64");
        if (imageBase64 == null) {
            call.reject("Base64 gambar tidak boleh kosong");
            return;
        }

        try {
            // 1. Decode Base64 ke Bitmap
            Bitmap bitmap = decodeOrientedBitmap(imageBase64);
            if (bitmap == null) {
                call.reject("Gagal decode Base64 menjadi gambar");
                return;
            }
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            // 2. Deteksi wajah menggunakan ML Kit
            livenessDetector.process(image)
                .addOnSuccessListener(faces -> {
                    if (faces.isEmpty()) {
                        call.reject("Tidak ada wajah yang terdeteksi untuk pemeriksaan liveness");
                        return;
                    }

                    try {
                        // 3. Pilih wajah dengan bbox terbesar
                        Face face = pickLargestFace(faces);
                        Rect bounds = face.getBoundingBox();

                        // 4. Pipeline anti-spoof (dua model, crop shift, argmax + veto)
                        AntiSpoofResult result = runAntiSpoofPipeline(bitmap, bounds);

                        // 5. Tentukan tingkat kepercayaan (kompatibel dengan response lama)
                        String confidence;
                        if (result.liveScore > 0.85f || result.liveScore < 0.15f) {
                            confidence = "HIGH";
                        } else if (result.liveScore > 0.6f || result.liveScore < 0.4f) {
                            confidence = "MEDIUM";
                        } else {
                            confidence = "LOW";
                        }

                        // 6. Kirim hasil ke JavaScript
                        JSObject ret = new JSObject();
                        ret.put("isLive", result.isLive);
                        ret.put("score", result.liveScore);       // dipetakan ke liveScore
                        ret.put("confidence", confidence);
                        ret.put("label", result.label);
                        ret.put("liveScore", result.liveScore);
                        ret.put("printScore", result.printScore);
                        ret.put("replayScore", result.replayScore);
                        ret.put("thresholdVersion", THRESHOLD_VERSION);
                        call.resolve(ret);

                    } catch (Exception e) {
                        call.reject("Gagal menjalankan inference anti-spoofing", e);
                    }
                })
                .addOnFailureListener(e -> call.reject("ML Kit gagal mendeteksi wajah", e));

        } catch (Exception e) {
            call.reject("Error memproses gambar untuk liveness check", e);
        }
    }

    // ============================
    // verifyFace — API terpadu (Fase 3.2, menutup TOCTOU)
    // ============================
    /**
     * Liveness + quality gate + embedding dari FRAME DAN BBOX YANG SAMA.
     * Embedding hanya dikembalikan jika isLive = true.
     *
     * options:
     *   imageBase64: string
     *   referenceEmbedding?: number[]   → jika ada, sekaligus compare (match)
     *   config?: {
     *     liveThreshold?, spoofVeto?, minFaceRatio?, minFrameSide?, edgeMargin?,
     *     maxPose?, blurMin?, exposureMin?, exposureMax?, debug?
     *   }
     */
    @PluginMethod
    public void verifyFace(PluginCall call) {
        if (antiSpoof27 == null || antiSpoof40 == null || tfliteInterpreter == null) {
            call.reject("Model liveness atau face recognition tidak ditemukan di assets");
            return;
        }

        String imageBase64 = call.getString("imageBase64");
        if (imageBase64 == null) {
            call.reject("Base64 gambar tidak boleh kosong");
            return;
        }

        GateConfig config = GateConfig.from(call.getObject("config"), DeviceIntegrity.isDebuggable(getContext()));
        JSONArray referenceEmbedding = call.getArray("referenceEmbedding");

        try {
            Bitmap bitmap = decodeOrientedBitmap(imageBase64);
            if (bitmap == null) {
                call.reject("Gagal decode Base64 menjadi gambar");
                return;
            }
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            livenessDetector.process(image)
                .addOnSuccessListener(faces -> {
                    try {
                        // ---- Quality gate (Fase 3.1) ----
                        QualityResult quality = runQualityGate(bitmap, faces, config);
                        if (!quality.passed) {
                            JSObject ret = new JSObject();
                            ret.put("isLive", false);
                            ret.put("thresholdVersion", THRESHOLD_VERSION);
                            ret.put("liveness", JSONObject.NULL);
                            ret.put("quality", quality.toJson());
                            call.resolve(ret);
                            return;
                        }

                        Face face = pickLargestFace(faces);
                        Rect bounds = face.getBoundingBox();

                        // ---- Anti-spoof (Fase 2) ----
                        AntiSpoofResult liveness = runAntiSpoofPipeline(bitmap, bounds, config);

                        JSObject ret = new JSObject();
                        ret.put("isLive", liveness.isLive);
                        ret.put("thresholdVersion", THRESHOLD_VERSION);
                        ret.put("liveness", liveness.toJson());
                        ret.put("quality", quality.toJson());

                        if (config.debug) {
                            ret.put("debug", liveness.debugToJson());
                        }

                        // ---- Embedding HANYA jika live, dari frame & bbox yang sama ----
                        if (liveness.isLive) {
                            float[] embeddings = extractEmbedding(bitmap, bounds);
                            if (embeddings == null) {
                                call.reject("Gagal mengekstrak embedding dari frame yang sama");
                                return;
                            }
                            JSONArray jsonArray = new JSONArray();
                            for (float val : embeddings) {
                                jsonArray.put((double) val);
                            }
                            ret.put("embedding", jsonArray);

                            if (referenceEmbedding != null) {
                                double score = cosineSimilarity(referenceEmbedding, jsonArray);
                                if (Double.isNaN(score)) {
                                    call.reject("referenceEmbedding tidak valid (norma nol)");
                                    return;
                                }
                                JSObject match = new JSObject();
                                match.put("isMatch", score > MATCH_THRESHOLD);
                                match.put("score", score);
                                match.put("threshold", MATCH_THRESHOLD);
                                match.put("thresholdVersion", MATCH_THRESHOLD_VERSION);
                                ret.put("match", match);
                            }
                        }

                        call.resolve(ret);
                    } catch (Exception e) {
                        call.reject("Gagal menjalankan verifyFace", e);
                    }
                })
                .addOnFailureListener(e -> call.reject("ML Kit gagal mendeteksi wajah", e));

        } catch (Exception e) {
            call.reject("Error memproses gambar untuk verifyFace", e);
        }
    }

    // ============================
    // Active challenge (Fase 3.3)
    // ============================
    /**
     * Mengeluarkan challenge acak untuk sesi liveness aktif.
     * Aplikasi menampilkan instruksi ke user, lalu memanggil verifyFaceSequence.
     */
    @PluginMethod
    public void getLivenessChallenge(PluginCall call) {
        String challenge = Math.random() < 0.5 ? "LOOK_LEFT" : "LOOK_RIGHT";
        JSObject ret = new JSObject();
        ret.put("challenge", challenge);
        call.resolve(ret);
    }

    /**
     * Verifikasi sesi multi-frame dengan active challenge (Fase 3.3).
     *
     * options:
     *   images: string[]            → ≥ 5 frame dalam ~1,5 detik
     *   challenge: 'LOOK_LEFT' | 'LOOK_RIGHT'
     *   referenceEmbedding?: number[]
     *   config?: GateConfig (sama dengan verifyFace)
     *
     * Aturan lulus:
     *   - SEMUA frame lolos quality gate
     *   - median liveScore ≥ liveThreshold DAN mayoritas frame isLive (voting,
     *     BUKAN "semua frame harus live" — lihat catatan BPCER di plan)
     *   - identitas konsisten antar frame (cosine ≥ 0.8 terhadap frame pertama)
     *   - challenge terpenuhi: yaw menyimpang ≥ 15° ke arah yang diminta, lalu
     *     minimal 2 frame kembali mendekati frontal (|yaw| < 15°)
     *
     * CATATAN KONVENSI TANDA YAW: default saat ini LOOK_LEFT = yaw negatif.
     * Jika di perangkat ternyata terbalik, set config.yawSign = -1.
     * Verifikasi tanda ini bagian dari kalibrasi Fase 5.
     */
    @PluginMethod
    public void verifyFaceSequence(PluginCall call) {
        if (antiSpoof27 == null || antiSpoof40 == null || tfliteInterpreter == null) {
            call.reject("Model liveness atau face recognition tidak ditemukan di assets");
            return;
        }

        if (!checkRateLimit()) {
            call.reject("Terlalu banyak percobaan sesi liveness. Coba lagi dalam beberapa menit.", "RATE_LIMITED");
            return;
        }

        JSONArray imagesJson = call.getArray("images");
        String challenge = call.getString("challenge");
        if (imagesJson == null || imagesJson.length() < 2) {
            call.reject("images minimal berisi 2 frame (disarankan ≥ 5)");
            return;
        }
        if (!"LOOK_LEFT".equals(challenge) && !"LOOK_RIGHT".equals(challenge)) {
            call.reject("challenge harus LOOK_LEFT atau LOOK_RIGHT (dari getLivenessChallenge)");
            return;
        }

        GateConfig config =
                GateConfig.from(call.getObject("config"), DeviceIntegrity.isDebuggable(getContext()));
        JSONArray referenceEmbedding = call.getArray("referenceEmbedding");

        try {
            List<String> images = new ArrayList<>();
            for (int i = 0; i < imagesJson.length(); i++) {
                images.add(imagesJson.getString(i));
            }

            // Proses frame SEKUENSIAL via rantai callback (aman dari deadlock,
            // listener ML Kit berjalan di main thread).
            SequenceSink sink = pluginSink(call);
            processFramesSequentially(sink, images, 0, config, challenge, referenceEmbedding,
                    new ArrayList<FrameResult>());
        } catch (Exception e) {
            call.reject("Gagal menjalankan verifyFaceSequence", e);
        }
    }

    /** Hasil sesi dikirim ke pemanggil (PluginCall atau alur native capture). */
    private interface SequenceSink {
        void resolve(JSObject result);
        void reject(String message);
    }

    private SequenceSink pluginSink(PluginCall call) {
        return new SequenceSink() {
            @Override
            public void resolve(JSObject result) {
                call.resolve(result);
            }

            @Override
            public void reject(String message) {
                call.reject(message);
            }
        };
    }

    /** Callback per frame: menerima FrameResult lalu lanjut ke frame berikutnya. */
    private void processFramesSequentially(SequenceSink sink, List<String> images, int index,
                                           GateConfig config, String challenge,
                                           JSONArray referenceEmbedding, List<FrameResult> frames) {
        if (index >= images.size()) {
            finishSequence(sink, frames, config, challenge, referenceEmbedding);
            return;
        }

        Bitmap bitmap;
        try {
            bitmap = decodeOrientedBitmap(images.get(index));
        } catch (Exception e) {
            bitmap = null;
        }
        if (bitmap == null) {
            sink.reject("Frame ke-" + index + " gagal di-decode");
            return;
        }
        final Bitmap frameBitmap = bitmap; // effectively final untuk lambda

        InputImage image = InputImage.fromBitmap(frameBitmap, 0);
        livenessDetector.process(image)
            .addOnSuccessListener(faces -> {
                FrameResult fr = new FrameResult();
                fr.index = index;
                if (faces.isEmpty()) {
                    fr.quality = QualityResult.fail("NO_FACE", 0, 0);
                } else {
                    fr.quality = runQualityGate(frameBitmap, faces, config);
                    if (fr.quality.passed) {
                        Face face = pickLargestFace(faces);
                        fr.bounds = face.getBoundingBox();
                        fr.yaw = face.getHeadEulerAngleY();
                        fr.pitch = face.getHeadEulerAngleX();
                        fr.roll = face.getHeadEulerAngleZ();
                        fr.liveness = runAntiSpoofPipeline(frameBitmap, fr.bounds, config);
                        fr.embedding = extractEmbedding(frameBitmap, fr.bounds);
                    }
                }
                frames.add(fr);
                processFramesSequentially(sink, images, index + 1, config, challenge,
                        referenceEmbedding, frames);
            })
            .addOnFailureListener(e -> {
                FrameResult fr = new FrameResult();
                fr.index = index;
                fr.quality = QualityResult.fail("NO_FACE", 0, 0);
                frames.add(fr);
                processFramesSequentially(sink, images, index + 1, config, challenge,
                        referenceEmbedding, frames);
            });
    }

    private void finishSequence(SequenceSink sink, List<FrameResult> frames, GateConfig config,
                                String challenge, JSONArray referenceEmbedding) {
        try {
        float[] firstEmbedding = null;
        boolean identityConsistent = true;
        String failureReason = null;

        for (FrameResult fr : frames) {
            // SEMUA frame harus lolos quality gate
            if (!fr.quality.passed) {
                failureReason = "QUALITY_FAILED";
                break;
            }

            // Konsistensi identitas terhadap frame pertama
            if (fr.embedding != null) {
                if (firstEmbedding == null) {
                    firstEmbedding = fr.embedding;
                } else {
                    double cos = cosineSimilarity(firstEmbedding, fr.embedding);
                    if (cos < IDENTITY_CONSISTENCY) {
                        identityConsistent = false;
                    }
                }
            }
        }

            // ---- Agregasi (median + voting, BUKAN all-live) ----
            float[] liveScores = new float[frames.size()];
            int liveVotes = 0;
            float maxYaw = Float.NEGATIVE_INFINITY;
            float minYaw = Float.POSITIVE_INFINITY;
            int frontalFrames = 0;

            for (int i = 0; i < frames.size(); i++) {
                FrameResult fr = frames.get(i);
                liveScores[i] = fr.liveness != null ? fr.liveness.liveScore : 0f;
                if (fr.liveness != null && fr.liveness.isLive) liveVotes++;
                if (fr.quality.passed) {
                    maxYaw = Math.max(maxYaw, fr.yaw);
                    minYaw = Math.min(minYaw, fr.yaw);
                    if (Math.abs(fr.yaw) < CHALLENGE_FRONT_MAX) frontalFrames++;
                }
            }

            float medianLiveScore = median(liveScores);
            int requiredVotes = (int) Math.ceil(frames.size() * 0.6);

            // Challenge: deviasi yaw ≥ CHALLENGE_YAW_MIN ke arah yang diminta
            int yawSign = config.yawSign;
            boolean challengeSatisfied;
            if ("LOOK_LEFT".equals(challenge)) {
                challengeSatisfied = (minYaw * yawSign) <= -CHALLENGE_YAW_MIN;
            } else {
                challengeSatisfied = (maxYaw * yawSign) >= CHALLENGE_YAW_MIN;
            }
            challengeSatisfied = challengeSatisfied && frontalFrames >= CHALLENGE_MIN_FRONTAL_FRAMES;

            boolean passed = failureReason == null
                    && medianLiveScore >= config.liveThreshold
                    && liveVotes >= requiredVotes
                    && identityConsistent
                    && challengeSatisfied;

            if (passed) {
                // verifikasi opsional terhadap referenceEmbedding (frame pertama)
                if (referenceEmbedding != null && firstEmbedding != null) {
                    JSONArray firstJson = new JSONArray();
                    for (float v : firstEmbedding) firstJson.put(Double.valueOf(v));
                    double score = cosineSimilarity(referenceEmbedding, firstJson);
                    if (Double.isNaN(score)) {
                        sink.reject("referenceEmbedding tidak valid (norma nol)");
                        return;
                    }
                    // match dilaporkan; tidak menggagalkan sesi (keputusan ada di aplikasi)
                    JSObject match = new JSObject();
                    match.put("isMatch", score > MATCH_THRESHOLD);
                    match.put("score", score);
                    match.put("threshold", MATCH_THRESHOLD);
                    match.put("thresholdVersion", MATCH_THRESHOLD_VERSION);

                    JSObject ret = buildSequenceResult(true, null, challenge, medianLiveScore,
                            liveVotes, frames.size(), maxYaw, minYaw, frontalFrames, identityConsistent);
                    ret.put("match", match);
                    sink.resolve(ret);
                    return;
                }
            }

            String reason = failureReason;
            if (reason == null) {
                if (!identityConsistent) reason = "IDENTITY_INCONSISTENT";
                else if (!challengeSatisfied) reason = "CHALLENGE_FAILED";
                else if (liveVotes < requiredVotes || medianLiveScore < config.liveThreshold) reason = "NOT_LIVE_ENOUGH";
            }

            sink.resolve(buildSequenceResult(passed, reason, challenge, medianLiveScore,
                    liveVotes, frames.size(), maxYaw, minYaw, frontalFrames, identityConsistent));

        } catch (Exception e) {
            sink.reject("Gagal menjalankan verifyFaceSequence: " + e.getMessage());
        }
    }

    private JSObject buildSequenceResult(boolean passed, String reason, String challenge,
                                         float medianLiveScore, int liveVotes, int frameCount,
                                         float maxYaw, float minYaw, int frontalFrames,
                                         boolean identityConsistent) {
        JSObject ret = new JSObject();
        ret.put("passed", passed);
        if (reason != null) ret.put("reason", reason);
        ret.put("challenge", challenge);
        ret.put("medianLiveScore", medianLiveScore);
        ret.put("liveVotes", liveVotes);
        ret.put("frameCount", frameCount);
        ret.put("maxYaw", maxYaw);
        ret.put("minYaw", minYaw);
        ret.put("frontalFrames", frontalFrames);
        ret.put("identityConsistent", identityConsistent);
        ret.put("thresholdVersion", THRESHOLD_VERSION);
        return ret;
    }

    // ============================
    // Native capture (Fase 4) — captureLivenessSession
    // ============================
    /**
     * Sesi liveness native (anti injeksi frame): kamera dibuka oleh activity plugin,
     * frame diambil dan disimpan ke file privat TANPA melewati JS, lalu pipeline
     * verifikasi (quality gate + liveness + challenge) dijalankan terhadap frame
     * tersebut. Yang kembali ke JS hanya HASIL — bukan frame mentah.
     *
     * options:
     *   challenge?: 'LOOK_LEFT' | 'LOOK_RIGHT'  → default acak
     *   referenceEmbedding?: number[]
     *   config?: LivenessConfig
     *
     * Response = VerifyFaceSequenceResult + { native, deviceIntegrity, auditId }
     */
    @PluginMethod
    public void captureLivenessSession(PluginCall call) {
        if (antiSpoof27 == null || antiSpoof40 == null || tfliteInterpreter == null) {
            call.reject("Model liveness atau face recognition tidak ditemukan di assets");
            return;
        }

        if (!checkRateLimit()) {
            call.reject("Terlalu banyak percobaan sesi liveness. Coba lagi dalam beberapa menit.", "RATE_LIMITED");
            return;
        }

        if (getPermissionState("camera") != PermissionState.GRANTED) {
            requestAllPermissions(call, "cameraPermissionCallback");
            return;
        }

        startLivenessSession(call);
    }

    @PermissionCallback
    private void cameraPermissionCallback(PluginCall call) {
        if (getPermissionState("camera") == PermissionState.GRANTED) {
            startLivenessSession(call);
        } else {
            call.reject("Izin kamera ditolak");
        }
    }

    private void startLivenessSession(PluginCall call) {
        String challenge = call.getString("challenge");
        if (!"LOOK_LEFT".equals(challenge) && !"LOOK_RIGHT".equals(challenge)) {
            challenge = Math.random() < 0.5 ? "LOOK_LEFT" : "LOOK_RIGHT";
        }

        saveCall(call);
        Intent intent = LivenessCaptureActivity.createIntent(
                getContext(), challenge, NATIVE_FRAME_COUNT,
                NATIVE_FRAME_INTERVAL_MS, NATIVE_WARMUP_MS);
        startActivityForResult(call, intent, REQUEST_LIVENESS_SESSION);
    }

    @Override
    protected void handleOnActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != REQUEST_LIVENESS_SESSION) return;

        PluginCall call = getSavedCall();
        if (call == null) return;

        if (resultCode != Activity.RESULT_OK || data == null) {
            String error = data != null ? data.getStringExtra(LivenessCaptureActivity.EXTRA_ERROR) : null;
            freeSavedCall();
            call.reject(error != null ? error : "Capture dibatalkan");
            return;
        }

        String challenge = data.getStringExtra(LivenessCaptureActivity.EXTRA_CHALLENGE);
        ArrayList<String> framePaths =
                data.getStringArrayListExtra(LivenessCaptureActivity.EXTRA_FRAME_PATHS);
        freeSavedCall();

        if (framePaths == null || framePaths.isEmpty()) {
            call.reject("Tidak ada frame yang tertangkap");
            return;
        }

        try {
            List<String> frames = new ArrayList<>();
            List<byte[]> frameBytes = new ArrayList<>();
            for (String path : framePaths) {
                byte[] bytes = readFileBytes(new File(path));
                frameBytes.add(bytes);
                frames.add(android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP));
            }

            GateConfig config =
                    GateConfig.from(call.getObject("config"), DeviceIntegrity.isDebuggable(getContext()));
            JSONArray referenceEmbedding = call.getArray("referenceEmbedding");

            processFramesSequentially(nativeSessionSink(call, frameBytes),
                    frames, 0, config, challenge, referenceEmbedding, new ArrayList<FrameResult>());
        } catch (Exception e) {
            deleteFrameFiles();
            call.reject("Gagal memproses frame capture: " + e.getMessage());
        }
    }

    /** Sink khusus sesi native: lengkapi hasil dengan integrity + jejak audit terenkripsi. */
    private SequenceSink nativeSessionSink(PluginCall call, List<byte[]> frameBytes) {
        return new SequenceSink() {
            @Override
            public void resolve(JSObject result) {
                result.put("native", true);
                result.put("deviceIntegrity", DeviceIntegrity.snapshot(getContext()));
                try {
                    String auditId = AuditStore.storeSession(
                            getContext(), result.toString(), frameBytes);
                    result.put("auditId", auditId);
                } catch (Exception e) {
                    android.util.Log.e("FaceRecognition", "Audit store gagal: " + e.getMessage());
                    result.put("auditId", JSONObject.NULL);
                }
                deleteFrameFiles();
                call.resolve(result);
            }

            @Override
            public void reject(String message) {
                deleteFrameFiles();
                call.reject(message);
            }
        };
    }

    private void deleteFrameFiles() {
        File dir = new File(getContext().getCacheDir(), "liveness_frames");
        File[] files = dir.listFiles();
        if (files != null) {
            for (File f : files) {
                //noinspection ResultOfMethodCallIgnored
                f.delete();
            }
        }
    }

    private byte[] readFileBytes(File file) throws IOException {
        try (FileInputStream in = new FileInputStream(file)) {
            byte[] bytes = new byte[(int) file.length()];
            int offset = 0;
            while (offset < bytes.length) {
                int read = in.read(bytes, offset, bytes.length - offset);
                if (read < 0) break;
                offset += read;
            }
            return bytes;
        }
    }

    // ============================
    // Device integrity (Fase 4) — sinyal risiko, bukan bukti
    // ============================
    @PluginMethod
    public void getDeviceIntegrity(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("deviceIntegrity", DeviceIntegrity.snapshot(getContext()));
        call.resolve(ret);
    }

    /** Rate limiting percobaan sesi (anti brute-force). */
    private boolean checkRateLimit() {
        long now = System.currentTimeMillis();
        synchronized (sessionAttempts) {
            while (!sessionAttempts.isEmpty()
                    && now - sessionAttempts.get(0) > SESSION_ATTEMPT_WINDOW_MS) {
                sessionAttempts.remove(0);
            }
            if (sessionAttempts.size() >= MAX_SESSION_ATTEMPTS) return false;
            sessionAttempts.add(now);
            return true;
        }
    }

    // ============================
    // Quality Gate (Fase 3.1)
    // ============================
    private QualityResult runQualityGate(Bitmap bitmap, List<Face> faces, GateConfig config) {
        int frameW = bitmap.getWidth();
        int frameH = bitmap.getHeight();

        // MULTIPLE_FACES: tolak jika ada > 1 wajah yang cukup besar
        Face largest = pickLargestFace(faces);
        int largestArea = Math.max(1, largest.getBoundingBox().width() * largest.getBoundingBox().height());
        int significantFaces = 0;
        for (Face f : faces) {
            int area = f.getBoundingBox().width() * f.getBoundingBox().height();
            if (area >= largestArea * 0.3f) significantFaces++;
        }
        if (significantFaces > 1) {
            return QualityResult.fail("MULTIPLE_FACES", 0, 0);
        }

        // LOW_RESOLUTION
        if (Math.min(frameW, frameH) < config.minFrameSide) {
            return QualityResult.fail("LOW_RESOLUTION", 0, 0);
        }

        // FACE_TOO_SMALL
        Rect bounds = largest.getBoundingBox();
        float faceRatio = bounds.width() / (float) frameW;
        if (faceRatio < config.minFaceRatio) {
            return QualityResult.fail("FACE_TOO_SMALL", faceRatio, 0);
        }

        // FACE_AT_EDGE: bbox menyentuh tepi frame (margin 5%)
        float marginX = frameW * config.edgeMargin;
        float marginY = frameH * config.edgeMargin;
        if (bounds.left <= marginX || bounds.top <= marginY
                || bounds.right >= frameW - marginX || bounds.bottom >= frameH - marginY) {
            return QualityResult.fail("FACE_AT_EDGE", faceRatio, 0);
        }

        // BAD_POSE
        float yaw = largest.getHeadEulerAngleY();
        float pitch = largest.getHeadEulerAngleX();
        float roll = largest.getHeadEulerAngleZ();
        if (Math.abs(yaw) > config.maxPose || Math.abs(pitch) > config.maxPose || Math.abs(roll) > config.maxPose) {
            return QualityResult.fail("BAD_POSE", faceRatio, 0);
        }

        // BLURRY + BAD_LIGHTING dari crop wajah
        Bitmap faceCrop = cropFace(bitmap, bounds);
        float blur = varianceOfLaplacian(faceCrop);
        float exposure = meanLuminance(faceCrop);
        if (blur < config.blurMin) {
            return QualityResult.fail("BLURRY", faceRatio, blur);
        }
        if (exposure < config.exposureMin || exposure > config.exposureMax) {
            return QualityResult.fail("BAD_LIGHTING", faceRatio, blur);
        }

        return QualityResult.pass(faceRatio, blur);
    }

    // ============================
    // Anti-Spoof Pipeline (Fase 2) — port referensi Silent-Face
    // ============================
    private static class AntiSpoofResult {
        String label;
        float liveScore, printScore, replayScore, spoofScore;
        boolean isLive;
        float[] probs27, probs40;

        JSObject toJson() {
            JSObject o = new JSObject();
            o.put("label", label);
            o.put("liveScore", liveScore);
            o.put("printScore", printScore);
            o.put("replayScore", replayScore);
            return o;
        }

        JSObject debugToJson() {
            JSObject o = new JSObject();
            JSONArray p27 = new JSONArray();
            if (probs27 != null) for (float v : probs27) p27.put(Double.valueOf(v));
            JSONArray p40 = new JSONArray();
            if (probs40 != null) for (float v : probs40) p40.put(Double.valueOf(v));
            o.put("probs27", p27);
            o.put("probs40", p40);
            return o;
        }
    }

    private AntiSpoofResult runAntiSpoofPipeline(Bitmap bitmap, Rect bounds) {
        return runAntiSpoofPipeline(bitmap, bounds, GateConfig.from(null));
    }

    /**
     * Pipeline referensi:
     *   crop 2.7x (shift) → model 2.7 ─┐
     *                                    ├→ sum softmax → argmax + veto → keputusan
     *   crop 4.0x (shift) → model 4.0 ─┘
     */
    private AntiSpoofResult runAntiSpoofPipeline(Bitmap bitmap, Rect bounds, GateConfig config) {
        float[] logits27 = runAntiSpoofModel(antiSpoof27, antiSpoof27Lock, bitmap, bounds, CROP_SCALE_27);
        float[] logits40 = runAntiSpoofModel(antiSpoof40, antiSpoof40Lock, bitmap, bounds, CROP_SCALE_40);
        if (logits27 == null || logits40 == null) {
            throw new IllegalStateException("Inference anti-spoof gagal pada salah satu model");
        }

        float[] probs27 = softmax(logits27);
        float[] probs40 = softmax(logits40);

        // p = softmax(out27) + softmax(out40) → rentang 0..2
        float[] p = new float[ANTI_SPOOF_OUTPUT_SIZE];
        for (int i = 0; i < ANTI_SPOOF_OUTPUT_SIZE; i++) {
            p[i] = probs27[i] + probs40[i];
        }

        // argmax 3 kelas: [print, live, replay]
        int argmax = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[argmax]) argmax = i;
        }

        AntiSpoofResult result = new AntiSpoofResult();
        result.printScore = p[0] / 2f;
        result.liveScore = p[1] / 2f;
        result.replayScore = p[2] / 2f;
        result.spoofScore = Math.max(p[0], p[2]) / 2f;
        result.label = argmax == 1 ? "LIVE" : (argmax == 0 ? "PRINT" : "REPLAY");
        result.isLive = argmax == 1
                && result.liveScore >= config.liveThreshold
                && result.spoofScore < config.spoofVeto;
        result.probs27 = probs27;
        result.probs40 = probs40;
        return result;
    }

    /**
     * Menjalankan satu inferensi anti-spoofing untuk satu model dengan crop scale-nya.
     * Interpreter TFLite TIDAK thread-safe → akses disinkronkan per model.
     *
     * @return raw logits (3 kelas), atau null jika gagal
     */
    private float[] runAntiSpoofModel(Interpreter interpreter, Object lock, Bitmap bitmap,
                                      Rect bounds, float cropScale) {
        if (interpreter == null) return null;
        try {
            // Crop sesuai referensi (_get_new_box): scale diturunkan jika tidak muat,
            // w & h terpisah, SHIFT bukan clip, tanpa square-ifikasi
            int[] box = getNewBox(bounds, cropScale, bitmap.getWidth(), bitmap.getHeight());
            Bitmap croppedFace = Bitmap.createBitmap(bitmap, box[0], box[1], box[2], box[3]);

            // Resize ke 80x80 (input MiniFASNet)
            Bitmap scaledFace = Bitmap.createScaledBitmap(
                    croppedFace, ANTI_SPOOF_INPUT_SIZE, ANTI_SPOOF_INPUT_SIZE, true);

            // Konversi ke ByteBuffer: BGR mentah 0..255 (TANPA normalisasi)
            ByteBuffer inputBuffer = bitmapToByteBufferAntiSpoof(scaledFace, ANTI_SPOOF_INPUT_SIZE);

            // Output model: [1, 3] → [print_spoof, live, replay_spoof] (raw logits)
            float[][] outputArray = new float[1][ANTI_SPOOF_OUTPUT_SIZE];
            synchronized (lock) {
                interpreter.run(inputBuffer, outputArray);
            }
            return outputArray[0];
        } catch (Exception e) {
            android.util.Log.e("AntiSpoof", "Inference error (scale " + cropScale + "): " + e.getMessage());
            return null;
        }
    }

    /**
     * Port logika crop referensi (_get_new_box):
     *   scale diturunkan jika crop tidak muat di gambar,
     *   lebar dan tinggi diskalakan TERPISAH (tanpa square),
     *   jika keluar batas maka DIGESER (shift), bukan dipotong (clip).
     */
    private int[] getNewBox(Rect bounds, float scale, int imgW, int imgH) {
        float boxW = bounds.width();
        float boxH = bounds.height();
        float cx = bounds.exactCenterX();
        float cy = bounds.exactCenterY();

        // Turunkan scale jika crop tidak muat
        float maxScaleW = (imgW - 1) / boxW;
        float maxScaleH = (imgH - 1) / boxH;
        float s = Math.min(scale, Math.min(maxScaleW, maxScaleH));

        float newW = boxW * s;
        float newH = boxH * s;

        float left = cx - newW / 2f;
        float top = cy - newH / 2f;
        float right = left + newW;
        float bottom = top + newH;

        // SHIFT, bukan clip
        if (left < 0) { right -= left; left = 0; }
        if (top < 0) { bottom -= top; top = 0; }
        if (right > imgW - 1) { left -= (right - (imgW - 1)); right = imgW - 1; }
        if (bottom > imgH - 1) { top -= (bottom - (imgH - 1)); bottom = imgH - 1; }

        // Penjagaan akhir untuk kasus ekstrem
        left = Math.max(0, left);
        top = Math.max(0, top);

        int x = Math.round(left);
        int y = Math.round(top);
        int w = Math.min(imgW - x, Math.round(right - left));
        int h = Math.min(imgH - y, Math.round(bottom - top));
        w = Math.max(1, w);
        h = Math.max(1, h);
        return new int[]{x, y, w, h};
    }

    // ============================
    // Face Recognition helpers
    // ============================

    /**
     * Ekstrak embedding MobileFaceNet dari frame dan bbox yang diberikan.
     * Crop memakai logika yang SAMA dengan pipeline enrollment (bbox langsung,
     * di-clip ke batas gambar) agar embedding enroll dan verify konsisten.
     */
    private float[] extractEmbedding(Bitmap bitmap, Rect bounds) {
        if (tfliteInterpreter == null) return null;
        try {
            int left = Math.max(0, bounds.left);
            int top = Math.max(0, bounds.top);
            int width = Math.min(bounds.width(), bitmap.getWidth() - left);
            int height = Math.min(bounds.height(), bitmap.getHeight() - top);
            if (width <= 0 || height <= 0) return null;

            Bitmap croppedFace = Bitmap.createBitmap(bitmap, left, top, width, height);
            Bitmap scaledFace = Bitmap.createScaledBitmap(croppedFace, INPUT_SIZE, INPUT_SIZE, true);
            ByteBuffer inputBuffer = bitmapToByteBuffer(scaledFace, INPUT_SIZE);

            float[][] outputArray = new float[1][EMBEDDING_SIZE];
            synchronized (tfliteInterpreter) {
                tfliteInterpreter.run(inputBuffer, outputArray);
            }
            return outputArray[0];
        } catch (Exception e) {
            android.util.Log.e("FaceRecognition", "extractEmbedding error: " + e.getMessage());
            return null;
        }
    }

    private Face pickLargestFace(List<Face> faces) {
        Face best = faces.get(0);
        int bestArea = best.getBoundingBox().width() * best.getBoundingBox().height();
        for (Face f : faces) {
            int area = f.getBoundingBox().width() * f.getBoundingBox().height();
            if (area > bestArea) {
                best = f;
                bestArea = area;
            }
        }
        return best;
    }

    private double cosineSimilarity(JSONArray a, JSONArray b) throws Exception {
        if (a.length() != b.length()) return Double.NaN;
        double dot = 0, norm1 = 0, norm2 = 0;
        for (int i = 0; i < a.length(); i++) {
            double v1 = a.getDouble(i);
            double v2 = b.getDouble(i);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        if (norm1 == 0 || norm2 == 0) return Double.NaN;
        return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    private double cosineSimilarity(float[] a, float[] b) {
        double dot = 0, norm1 = 0, norm2 = 0;
        for (int i = 0; i < a.length && i < b.length; i++) {
            dot += a[i] * b[i];
            norm1 += a[i] * a[i];
            norm2 += b[i] * b[i];
        }
        if (norm1 == 0 || norm2 == 0) return Double.NaN;
        return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    // ============================
    // Quality gate helpers — blur, exposure
    // ============================

    private Bitmap cropFace(Bitmap bitmap, Rect bounds) {
        int left = Math.max(0, bounds.left);
        int top = Math.max(0, bounds.top);
        int width = Math.min(bounds.width(), bitmap.getWidth() - left);
        int height = Math.min(bounds.height(), bitmap.getHeight() - top);
        if (width <= 0 || height <= 0) return bitmap;
        return Bitmap.createBitmap(bitmap, left, top, width, height);
    }

    /** Variance of Laplacian pada grayscale — ukuran blur (semakin tinggi semakin tajam). */
    private float varianceOfLaplacian(Bitmap faceCrop) {
        final int size = 160; // ukuran kerja tetap agar skor konsisten antar frame
        Bitmap work = Bitmap.createScaledBitmap(faceCrop, size, size, true);
        int[] pixels = new int[size * size];
        work.getPixels(pixels, 0, size, 0, 0, size, size);

        float[] gray = new float[size * size];
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            float r = (p >> 16) & 0xFF;
            float g = (p >> 8) & 0xFF;
            float b = p & 0xFF;
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
        }

        // Kernel Laplacian 3x3: [0 1 0; 1 -4 1; 0 1 0]
        double sum = 0, sumSq = 0;
        int count = 0;
        for (int y = 1; y < size - 1; y++) {
            for (int x = 1; x < size - 1; x++) {
                int idx = y * size + x;
                float lap = gray[idx - size] + gray[idx + size] + gray[idx - 1] + gray[idx + 1]
                        - 4f * gray[idx];
                sum += lap;
                sumSq += lap * lap;
                count++;
            }
        }
        if (count == 0) return 0;
        double mean = sum / count;
        return (float) (sumSq / count - mean * mean);
    }

    /** Mean luminance crop wajah (0..255) — untuk gate exposure. */
    private float meanLuminance(Bitmap faceCrop) {
        final int size = 64;
        Bitmap work = Bitmap.createScaledBitmap(faceCrop, size, size, true);
        int[] pixels = new int[size * size];
        work.getPixels(pixels, 0, size, 0, 0, size, size);
        double sum = 0;
        for (int p : pixels) {
            float r = (p >> 16) & 0xFF;
            float g = (p >> 8) & 0xFF;
            float b = p & 0xFF;
            sum += 0.299 * r + 0.587 * g + 0.114 * b;
        }
        return (float) (sum / pixels.length);
    }

    // ============================
    // Image / preprocessing helpers
    // ============================

    /** Decode Base64 → Bitmap dengan rotasi/flip EXIF sudah dinormalisasi. */
    private Bitmap decodeOrientedBitmap(String imageBase64) {
        byte[] decodedString = android.util.Base64.decode(imageBase64, android.util.Base64.DEFAULT);
        Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
        if (bitmap == null) return null;

        try {
            ExifInterface exif = new ExifInterface(new ByteArrayInputStream(decodedString));
            int orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            Matrix matrix = new Matrix();
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:  matrix.postRotate(90); break;
                case ExifInterface.ORIENTATION_ROTATE_180: matrix.postRotate(180); break;
                case ExifInterface.ORIENTATION_ROTATE_270: matrix.postRotate(270); break;
                case ExifInterface.ORIENTATION_FLIP_HORIZONTAL: matrix.postScale(-1, 1); break;
                case ExifInterface.ORIENTATION_FLIP_VERTICAL:   matrix.postScale(1, -1); break;
                case ExifInterface.ORIENTATION_TRANSPOSE:
                    matrix.postRotate(90); matrix.postScale(-1, 1); break;
                case ExifInterface.ORIENTATION_TRANSVERSE:
                    matrix.postRotate(270); matrix.postScale(-1, 1); break;
                default: break;
            }
            if (!matrix.isIdentity()) {
                Bitmap oriented = Bitmap.createBitmap(
                        bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                if (oriented != bitmap) bitmap.recycle();
                bitmap = oriented;
            }
        } catch (IOException ignored) {
            // EXIF tidak tersedia — pakai bitmap apa adanya
        }
        return bitmap;
    }

    /**
     * Mengkonversi Bitmap menjadi ByteBuffer untuk input TFLite (face recognition).
     *
     * Normalisasi ke rentang [-1, 1]:
     *   normalized = (pixel_value / 128.0f) - 1.0f
     *
     * Format buffer: [1, size, size, 3] dengan FLOAT32
     */
    private ByteBuffer bitmapToByteBuffer(Bitmap bitmap, int size) {
        // Alokasi buffer: 1 batch * tinggi * lebar * channel * bytes_per_float
        int bufferSize = 1 * size * size * PIXEL_CHANNELS * BYTES_PER_FLOAT;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder());
        byteBuffer.rewind();

        // Ambil semua pixel sekaligus untuk performa lebih baik
        int[] pixels = new int[size * size];
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size);

        for (int pixelValue : pixels) {
            // Ekstrak komponen RGB
            int r = (pixelValue >> 16) & 0xFF;
            int g = (pixelValue >> 8) & 0xFF;
            int b = pixelValue & 0xFF;

            // Normalisasi ke [-1, 1] sesuai format input MobileFaceNet
            byteBuffer.putFloat((r / 128.0f) - 1.0f);
            byteBuffer.putFloat((g / 128.0f) - 1.0f);
            byteBuffer.putFloat((b / 128.0f) - 1.0f);
        }

        return byteBuffer;
    }

    /**
     * Mengkonversi Bitmap menjadi ByteBuffer untuk input TFLite (anti-spoofing).
     *
     * Model Silent-Face mengekspektasikan nilai piksel RAW [0, 255] dalam urutan BGR
     * (normalisasi ada di dalam graf model). Jangan paksa ke [-1, 1].
     */
    private ByteBuffer bitmapToByteBufferAntiSpoof(Bitmap bitmap, int size) {
        int bufferSize = 1 * size * size * PIXEL_CHANNELS * BYTES_PER_FLOAT;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder());
        byteBuffer.rewind();

        int[] pixels = new int[size * size];
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size);

        for (int pixelValue : pixels) {
            int r = (pixelValue >> 16) & 0xFF;
            int g = (pixelValue >> 8) & 0xFF;
            int b = pixelValue & 0xFF;

            // BGR, unnormalized [0..255]
            byteBuffer.putFloat((float) b);
            byteBuffer.putFloat((float) g);
            byteBuffer.putFloat((float) r);
        }

        return byteBuffer;
    }

    // ============================
    // Softmax — konversi raw logits ke probability
    // ============================
    private float[] softmax(float[] logits) {
        float maxVal = logits[0];
        for (float v : logits) if (v > maxVal) maxVal = v;

        float sum = 0f;
        float[] exps = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = (float) Math.exp(logits[i] - maxVal); // numerically stable
            sum += exps[i];
        }
        for (int i = 0; i < exps.length; i++) {
            exps[i] /= sum;
        }
        return exps;
    }

    private float median(float[] values) {
        if (values.length == 0) return 0f;
        float[] sorted = values.clone();
        java.util.Arrays.sort(sorted);
        int mid = sorted.length / 2;
        if (sorted.length % 2 == 1) return sorted[mid];
        return (sorted[mid - 1] + sorted[mid]) / 2f;
    }

    // ============================
    // Konfigurasi quality gate yang bisa di-override per panggilan
    // ============================
    private static class GateConfig {
        float liveThreshold = LIVE_THRESHOLD;
        float spoofVeto = SPOOF_VETO;
        int minFrameSide = MIN_FRAME_SIDE;
        float minFaceRatio = MIN_FACE_RATIO;
        float edgeMargin = EDGE_MARGIN;
        float maxPose = MAX_POSE_DEG;
        float blurMin = BLUR_MIN;
        float exposureMin = EXPOSURE_MIN;
        float exposureMax = EXPOSURE_MAX;
        int yawSign = 1;
        boolean debug = false;

        static GateConfig from(JSObject config) {
            return from(config, false);
        }

        /**
         * @param debugAllowed hanya true pada build debuggable — output debug
         *                     tidak pernah dikembalikan dari build rilis (Fase 4)
         */
        static GateConfig from(JSObject config, boolean debugAllowed) {
            GateConfig c = new GateConfig();
            c.debug = false;
            if (config == null) return c;
            // opt* tidak melempar JSONException ter-cek
            c.liveThreshold  = (float) config.optDouble("liveThreshold",  c.liveThreshold);
            c.spoofVeto      = (float) config.optDouble("spoofVeto",      c.spoofVeto);
            c.minFrameSide   = config.optInt("minFrameSide", MIN_FRAME_SIDE);
            c.minFaceRatio   = (float) config.optDouble("minFaceRatio",   c.minFaceRatio);
            c.edgeMargin     = (float) config.optDouble("edgeMargin",     c.edgeMargin);
            c.maxPose        = (float) config.optDouble("maxPose",        c.maxPose);
            c.blurMin        = (float) config.optDouble("blurMin",        c.blurMin);
            c.exposureMin    = (float) config.optDouble("exposureMin",    c.exposureMin);
            c.exposureMax    = (float) config.optDouble("exposureMax",    c.exposureMax);
            c.yawSign        = config.optInt("yawSign", 1);
            c.debug          = config.optBoolean("debug", false) && debugAllowed;
            return c;
        }
    }

    private static class QualityResult {
        boolean passed;
        String reason;
        float faceRatio, blur;

        static QualityResult pass(float faceRatio, float blur) {
            QualityResult q = new QualityResult();
            q.passed = true;
            q.faceRatio = faceRatio;
            q.blur = blur;
            return q;
        }

        static QualityResult fail(String reason, float faceRatio, float blur) {
            QualityResult q = new QualityResult();
            q.passed = false;
            q.reason = reason;
            q.faceRatio = faceRatio;
            q.blur = blur;
            return q;
        }

        JSObject toJson() {
            JSObject o = new JSObject();
            o.put("passed", passed);
            if (reason != null) o.put("reason", reason);
            o.put("faceRatio", faceRatio);
            o.put("blur", blur);
            return o;
        }
    }

    private static class FrameResult {
        int index;
        Rect bounds;
        float yaw, pitch, roll;
        QualityResult quality;
        AntiSpoofResult liveness;
        float[] embedding;
    }
}
