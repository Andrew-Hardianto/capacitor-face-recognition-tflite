package com.andrew.face;

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
import java.io.File;

@CapacitorPlugin(name = "FaceRecognition")
public class FaceRecognitionPlugin extends Plugin {

    // Konfigurasi ML Kit agar cepat (tanpa landmark)
    private FaceDetectorOptions options = new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build();
    private FaceDetector detector = FaceDetection.getClient(options);

    @PluginMethod
    public void extractFaceFeature(PluginCall call) {
        String imageBase64 = call.getString("imageBase64");
        if (imageBase64 == null) {
            call.reject("Base64 gambar tidak boleh kosong");
            return;
        }

        try {
            // 1. Ubah path gambar menjadi Bitmap
            byte[] decodedString = android.util.Base64.decode(imageBase64, android.util.Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            InputImage image = InputImage.fromBitmap(bitmap, 0);

            // 2. Kirim ke ML Kit untuk deteksi wajah
            detector.process(image)
                .addOnSuccessListener(faces -> {
                    if (faces.isEmpty()) {
                        call.reject("Tidak ada wajah yang terdeteksi");
                        return;
                    }

                    // Ambil wajah pertama
                    Face face = faces.get(0);
                    Rect bounds = face.getBoundingBox();

                    // 3. Potong (Crop) Bitmap sesuai bounding box wajah
                    Bitmap croppedFace = Bitmap.createBitmap(bitmap, bounds.left, bounds.top, bounds.width(), bounds.height());
                    
                    // 4. Resize ke 112x112 pixel (Syarat input MobileFaceNet)
                    Bitmap scaledFace = Bitmap.createScaledBitmap(croppedFace, 112, 112, false);

                    // 5. TODO: Masukkan 'scaledFace' ke TensorFlow Lite (Interpreter)
                    // Di sini Anda menjalankan TFLite inference.
                    // Sebagai contoh struktur, kita asumsikan TFLite mengembalikan float[] berukuran 128
                    float[] embeddings = new float[128]; // Hasil dari TFLite

                    // 6. Kembalikan array angka ke Angular
                    try {
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
                .addOnFailureListener(e -> call.reject("ML Kit gagal mendeteksi", e));

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

            double sum = 0;
            for (int i = 0; i < vec1Json.length(); i++) {
                double diff = vec1Json.getDouble(i) - vec2Json.getDouble(i);
                sum += diff * diff;
            }
            
            double distance = Math.sqrt(sum);
            boolean isMatch = distance < 1.0;
            double similarityPercentage = 100.0 - (distance * 30.0);
            if (similarityPercentage < 0) similarityPercentage = 0;

            JSObject ret = new JSObject();
            ret.put("distance", distance);
            ret.put("similarityPercentage", similarityPercentage);
            call.resolve(ret);
        } catch (Exception e) {
            call.reject("Gagal membandingkan wajah", e);
        }
    }
}
