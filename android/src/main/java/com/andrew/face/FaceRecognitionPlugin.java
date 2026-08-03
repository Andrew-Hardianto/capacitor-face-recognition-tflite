package com.andrew.face;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import org.json.JSONArray;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

@CapacitorPlugin(name = "FaceRecognition")
public class FaceRecognitionPlugin extends Plugin {

    // ============================
    // Konfigurasi — Face Recognition
    // ============================
    // Nama file model di folder android/src/main/assets/
    private static final String MODEL_FILE = "mobile_face_net.tflite";
    // Ukuran input model (MobileFaceNet = 112x112)
    private static final int INPUT_SIZE = 112;
    // Jumlah channel warna (RGB = 3)
    private static final int PIXEL_CHANNELS = 3;
    // Ukuran output embedding model (model ini output 192 dimensi, bukan 128)
    private static final int EMBEDDING_SIZE = 192;
    // Ukuran byte per float
    private static final int BYTES_PER_FLOAT = 4;

    // ============================
    // Konfigurasi — Anti-Spoofing (Liveness Detection)
    // ============================
    // Nama file model anti-spoofing di folder android/src/main/assets/
    // Model: MiniFASNetV1 dari shubham0204/OnDevice-Face-Recognition-Android (spoof_model_scale_2_7.tflite)
    // Source: https://github.com/shubham0204/OnDevice-Face-Recognition-Android
    private static final String ANTI_SPOOF_MODEL_FILE = "anti_spoof.tflite";
    // MiniFASNet menggunakan input 80x80 (bukan 128x128)
    private static final int ANTI_SPOOF_INPUT_SIZE = 80;
    // Output model: 3 kelas softmax → [live_score, print_spoof_score, replay_spoof_score]
    // Index 0 = live (wajah asli)
    private static final int ANTI_SPOOF_OUTPUT_SIZE = 3;
    // Threshold liveness: skor live index-0 di atas ini dianggap Live (wajah asli)
    // Nilai default: 0.5 — bisa dinaikkan ke 0.6 untuk lebih ketat
    private static final float LIVENESS_THRESHOLD = 0.5f;

    // ML Kit Face Detector
    private final FaceDetectorOptions options = new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build();
    private final FaceDetector detector = FaceDetection.getClient(options);

    // TFLite Interpreter — Face Recognition
    private Interpreter tfliteInterpreter;

    // TFLite Interpreter — Anti-Spoofing (nullable, mungkin tidak ada modelnya)
    private Interpreter antiSpoofInterpreter;

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
        try {
            antiSpoofInterpreter = new Interpreter(loadModelFile(ANTI_SPOOF_MODEL_FILE));
        } catch (IOException e) {
            // Model anti-spoofing tidak ditemukan — checkLiveness akan return error informatif
            antiSpoofInterpreter = null;
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
            // 1. Decode Base64 ke Bitmap
            byte[] decodedString = android.util.Base64.decode(imageBase64, android.util.Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            // 2. Deteksi wajah menggunakan ML Kit
            detector.process(image)
                .addOnSuccessListener(faces -> {
                    if (faces.isEmpty()) {
                        call.reject("Tidak ada wajah yang terdeteksi");
                        return;
                    }

                    // 3. Ambil bounding box wajah pertama
                    Face face = faces.get(0);
                    Rect bounds = face.getBoundingBox();

                    // Pastikan bounding box tidak keluar dari batas bitmap
                    int left = Math.max(0, bounds.left);
                    int top = Math.max(0, bounds.top);
                    int width = Math.min(bounds.width(), bitmap.getWidth() - left);
                    int height = Math.min(bounds.height(), bitmap.getHeight() - top);

                    // 4. Crop wajah dari gambar asli
                    Bitmap croppedFace = Bitmap.createBitmap(bitmap, left, top, width, height);

                    // 5. Resize ke 80x80 (input untuk model face recognition)
                    Bitmap scaledFace = Bitmap.createScaledBitmap(croppedFace, INPUT_SIZE, INPUT_SIZE, true);

                    // 6. Konversi Bitmap ke ByteBuffer (input TFLite)
                    ByteBuffer inputBuffer = bitmapToByteBuffer(scaledFace, INPUT_SIZE);

                    // 7. Siapkan output buffer untuk embedding - shape: [1, EMBEDDING_SIZE]
                    float[][] outputArray = new float[1][EMBEDDING_SIZE];

                    // 8. Jalankan TFLite inference
                    tfliteInterpreter.run(inputBuffer, outputArray);

                    // 9. Kirim hasil embedding ke JavaScript
                    try {
                        float[] embeddings = outputArray[0];
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

            double dotProduct = 0.0;
            double norm1 = 0.0;
            double norm2 = 0.0;

            // Menghitung Cosine Similarity
            for (int i = 0; i < vec1Json.length(); i++) {
                double v1 = vec1Json.getDouble(i);
                double v2 = vec2Json.getDouble(i);

                dotProduct += v1 * v2;
                norm1 += v1 * v1;
                norm2 += v2 * v2;
            }

            // Mencegah pembagian dengan nol jika data cacat
            if (norm1 == 0 || norm2 == 0) {
                call.reject("Data vektor ada yang bernilai nol semua, cek apakah embedding berhasil");
                return;
            }

            double cosineSimilarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

            // THRESHOLD / AMBANG BATAS
            // 0.75 adalah standar yang baik.
            // Jika masih tembus wajah orang lain, naikkan ke 0.80 atau 0.85
            boolean isMatch = cosineSimilarity > 0.75;

            // Persentase kemiripan yang lebih akurat
            double similarityPercentage = Math.max(0, cosineSimilarity * 100.0);

            JSObject ret = new JSObject();
            ret.put("isMatch", isMatch);
            ret.put("similarityPercentage", similarityPercentage);
            ret.put("score", cosineSimilarity);

            call.resolve(ret);
        } catch (Exception e) {
            call.reject("Gagal membandingkan wajah", e);
        }
    }

    /**
     * Mendeteksi semua wajah dalam gambar menggunakan ML Kit.
     * Mengembalikan jumlah wajah dan bounding box tiap wajah.
     *
     * Berguna untuk validasi awal sebelum extractFaceFeature:
     *   - Pastikan tepat 1 wajah terdeteksi
     *   - Dapatkan koordinat wajah untuk UI feedback
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
            // 1. Decode Base64 ke Bitmap
            byte[] decodedString = android.util.Base64.decode(imageBase64, android.util.Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            // 2. Jalankan ML Kit Face Detector
            detector.process(image)
                .addOnSuccessListener(faces -> {
                    try {
                        // 3. Susun array bounding box untuk setiap wajah
                        JSONArray facesArray = new JSONArray();
                        for (Face face : faces) {
                            Rect bounds = face.getBoundingBox();

                            // Klem koordinat agar tidak keluar batas bitmap
                            int x      = Math.max(0, bounds.left);
                            int y      = Math.max(0, bounds.top);
                            int width  = Math.min(bounds.width(), bitmap.getWidth() - x);
                            int height = Math.min(bounds.height(), bitmap.getHeight() - y);

                            JSObject faceObj = new JSObject();
                            faceObj.put("x", x);
                            faceObj.put("y", y);
                            faceObj.put("width", width);
                            faceObj.put("height", height);
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

    /**
     * Memeriksa apakah wajah dalam gambar adalah wajah asli (live) atau spoofing.
     *
     * Alur kerja:
     *   1. Decode Base64 → Bitmap
     *   2. Deteksi wajah (ML Kit) → crop region wajah
     *   3. Resize ke 128×128 (input MiniFASNet)
     *   4. Jalankan TFLite inference model anti-spoofing
     *   5. Output model: [spoof_score, liveness_score] → normalisasi dengan softmax
     *   6. Kembalikan isLive, score, dan confidence ke JavaScript
     *
     * Memerlukan file 'anti_spoof.tflite' di folder android/src/main/assets/.
     * Download: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
     */
    @PluginMethod
    public void checkLiveness(PluginCall call) {
        // Periksa apakah model anti-spoofing berhasil dimuat
        if (antiSpoofInterpreter == null) {
            call.reject(
                "Model anti-spoofing tidak ditemukan. " +
                "Letakkan file '" + ANTI_SPOOF_MODEL_FILE + "' di folder android/src/main/assets/. " +
                "Download model dari: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
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
            byte[] decodedString = android.util.Base64.decode(imageBase64, android.util.Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            // 2. Deteksi wajah menggunakan ML Kit
            detector.process(image)
                .addOnSuccessListener(faces -> {
                    if (faces.isEmpty()) {
                        call.reject("Tidak ada wajah yang terdeteksi untuk pemeriksaan liveness");
                        return;
                    }

                    try {
                        // 3. Ambil bounding box wajah pertama
                        Face face = faces.get(0);
                        Rect bounds = face.getBoundingBox();

                        // Margin tambahan agar konteks kulit sekitar wajah ikut tertangkap
                        // (membantu model anti-spoofing mendeteksi tepi foto/layar)
                        int marginW = (int) (bounds.width() * 0.2f);
                        int marginH = (int) (bounds.height() * 0.2f);

                        int left   = Math.max(0, bounds.left - marginW);
                        int top    = Math.max(0, bounds.top - marginH);
                        int right  = Math.min(bitmap.getWidth(), bounds.right + marginW);
                        int bottom = Math.min(bitmap.getHeight(), bounds.bottom + marginH);
                        int width  = right - left;
                        int height = bottom - top;

                        // 4. Crop wajah (dengan margin)
                        Bitmap croppedFace = Bitmap.createBitmap(bitmap, left, top, width, height);

                        // 5. Resize ke ANTI_SPOOF_INPUT_SIZE x ANTI_SPOOF_INPUT_SIZE (80x80)
                        Bitmap scaledFace = Bitmap.createScaledBitmap(
                                croppedFace, ANTI_SPOOF_INPUT_SIZE, ANTI_SPOOF_INPUT_SIZE, true);

                        // 6. Konversi ke ByteBuffer — normalisasi ke [0, 1] (MiniFASNet standard)
                        ByteBuffer inputBuffer = bitmapToByteBufferAntiSpoof(scaledFace, ANTI_SPOOF_INPUT_SIZE);

                        // 7. Output model MiniFASNet: [1, 3] → [live_score, print_spoof, replay_spoof]
                        float[][] outputArray = new float[1][ANTI_SPOOF_OUTPUT_SIZE];

                        // 8. Jalankan inference
                        antiSpoofInterpreter.run(inputBuffer, outputArray);

                        // 9. Terapkan softmax manual karena model mungkin output raw logits
                        // Original repo (minivision-ai): Label 1 adalah Real Face
                        // Index 0 = spoof, Index 1 = live, Index 2 = spoof
                        float[] probs = softmax(outputArray[0]);
                        float spoofScore1     = probs[0];
                        float liveScore       = probs[1];
                        float spoofScore2     = probs[2];

                        // 10. Skor liveness dari output softmax (sudah dinormalisasi 0-1)
                        float livenessScore = liveScore;

                        // 11. Tentukan apakah live berdasarkan threshold
                        boolean isLive = livenessScore > LIVENESS_THRESHOLD;

                        // 12. Tentukan tingkat kepercayaan
                        String confidence;
                        if (livenessScore > 0.85f || livenessScore < 0.15f) {
                            confidence = "HIGH";
                        } else if (livenessScore > 0.6f || livenessScore < 0.4f) {
                            confidence = "MEDIUM";
                        } else {
                            confidence = "LOW";
                        }

                        // 13. Kirim hasil ke JavaScript
                        JSObject ret = new JSObject();
                        ret.put("isLive", isLive);
                        ret.put("score", livenessScore);
                        ret.put("confidence", confidence);
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
    // Helper Methods
    // ============================

    /**
     * Mengkonversi Bitmap menjadi ByteBuffer yang dinormalisasi untuk input TFLite.
     *
     * Normalisasi ke rentang [-1, 1]:
     *   normalized = (pixel_value / 128.0f) - 1.0f
     *
     * Format buffer: [1, size, size, 3] dengan FLOAT32
     *
     * @param bitmap  Bitmap yang sudah di-resize ke ukuran yang sesuai
     * @param size    Ukuran input model (112 untuk face recognition, 128 untuk anti-spoofing)
     */
    // ============================
    // Preprocessing — Face Recognition (MobileFaceNet)
    // Normalisasi ke [-1, 1]
    // ============================
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

    // ============================
    // Preprocessing — Anti-Spoofing (MiniFASNet)
    // Normalisasi ke [0, 1] + channel order BGR (MiniFASNet dilatih dengan OpenCV = BGR)
    // ============================
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

            // MiniFASNet dilatih dengan OpenCV (BGR), normalisasi HILANGKAN (model asli pakai [0, 255])
            // Urutan channel: B, G, R (BUKAN R, G, B)
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
        float max = logits[0];
        for (float v : logits) if (v > max) max = v;

        float sum = 0f;
        float[] exp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max); // numerically stable
            sum += exp[i];
        }
        for (int i = 0; i < exp.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }
}
