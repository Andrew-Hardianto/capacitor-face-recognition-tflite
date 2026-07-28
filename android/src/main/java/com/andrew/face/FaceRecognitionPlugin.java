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
    // Konfigurasi
    // ============================
    // Nama file model di folder android/src/main/assets/
    private static final String MODEL_FILE = "mobile_face_net.tflite";
    // Ukuran input model (MobileFaceNet = 112x112)
    private static final int INPUT_SIZE = 112;
    // Jumlah channel warna (RGB = 3)
    private static final int PIXEL_CHANNELS = 3;
    // Ukuran output embedding model (MobileFaceNet = 128)
    private static final int EMBEDDING_SIZE = 128;
    // Ukuran byte per float
    private static final int BYTES_PER_FLOAT = 4;

    // ML Kit Face Detector
    private final FaceDetectorOptions options = new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build();
    private final FaceDetector detector = FaceDetection.getClient(options);

    // TFLite Interpreter
    private Interpreter tfliteInterpreter;

    // ============================
    // Inisialisasi Plugin
    // ============================
    @Override
    public void load() {
        super.load();
        try {
            tfliteInterpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            // Model gagal dimuat - pastikan file MODEL_FILE ada di android/src/main/assets/
            e.printStackTrace();
        }
    }

    /**
     * Memuat file model TFLite dari folder assets secara memory-mapped
     * untuk performa yang lebih baik.
     */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getContext().getAssets().openFd(MODEL_FILE);
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

                    // 5. Resize ke INPUT_SIZE x INPUT_SIZE (112x112 untuk MobileFaceNet)
                    Bitmap scaledFace = Bitmap.createScaledBitmap(croppedFace, INPUT_SIZE, INPUT_SIZE, true);

                    // 6. Konversi Bitmap ke ByteBuffer (input TFLite)
                    ByteBuffer inputBuffer = bitmapToByteBuffer(scaledFace);

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

    /**
     * Mengkonversi Bitmap menjadi ByteBuffer yang dinormalisasi untuk input TFLite.
     *
     * Model MobileFaceNet mengharapkan nilai pixel dalam rentang [-1, 1]:
     *   normalized = (pixel_value / 128.0f) - 1.0f
     *
     * Format buffer: [1, 112, 112, 3] dengan FLOAT32
     */
    private ByteBuffer bitmapToByteBuffer(Bitmap bitmap) {
        // Alokasi buffer: 1 batch * tinggi * lebar * channel * bytes_per_float
        int bufferSize = 1 * INPUT_SIZE * INPUT_SIZE * PIXEL_CHANNELS * BYTES_PER_FLOAT;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder());
        byteBuffer.rewind();

        // Ambil semua pixel sekaligus untuk performa lebih baik
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

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
}
