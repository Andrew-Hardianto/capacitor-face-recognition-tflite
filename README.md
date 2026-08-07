# capacitor-face-recognition-tflite

Capacitor plugin untuk **Face Recognition** + **Anti-Spoofing (Passive Liveness Detection)** menggunakan ML Kit & TensorFlow Lite, berjalan sepenuhnya on-device tanpa koneksi internet.

---

## Platform Support

| Platform | Face Recognition | Liveness Detection |
|----------|:-:|:-:|
| Android | ✅ | ✅ |
| iOS | ✅ | ✅ |
| Web | ⚠️ Stub only | ⚠️ Stub only |

---

## Requirements

### Capacitor

| Dependency | Minimum Version |
|------------|----------------|
| `@capacitor/core` | `>= 8.0.0` |

### Android

| Requirement | Minimum |
|-------------|---------|
| **Android API Level** | `24` (Android 7.0 Nougat) |
| **compileSdk** | `36` |
| **targetSdk** | `36` |
| **Java** | `21` |
| **Gradle** | `8.13.0` |
| `com.google.mlkit:face-detection` | `16.1.5` |
| `org.tensorflow:tensorflow-lite` | `2.14.0` |
| `org.tensorflow:tensorflow-lite-support` | `0.4.4` |

> ⚠️ ML Kit Face Detection memerlukan **Google Play Services** yang sudah terinstal. Perangkat tanpa Google Play Services (AOSP murni) tidak didukung.

### iOS

| Requirement | Minimum |
|-------------|---------|
| **iOS** | `15.0` |
| **Swift** | `5.1` |
| **Xcode** | `15.0+` |
| `TensorFlowLiteSwift` (via CocoaPods) | sesuai versi plugin |
| `TensorFlowLite` (via SPM) | `>= 2.14.0` |

> ℹ️ Face detection menggunakan framework **Apple Vision** bawaan iOS — tidak memerlukan library eksternal tambahan.

---

## Install

```bash
npm install capacitor-face-recognition-tflite
npx cap sync
```

---

## Model Files

Plugin menyertakan kedua model TFLite langsung di dalam package — **tidak perlu download manual**.

| Model | Platform | Ukuran | Fungsi |
|-------|----------|--------|--------|
| `mobile_face_net.tflite` | Android & iOS | ~5.0 MB | Face recognition (embedding 192-dim) |
| `anti_spoof.tflite` | Android & iOS | ~5.7 MB | Anti-spoofing / Liveness detection |

### Cara kerja bundling otomatis

**Android** — Model disalin ke `android/src/main/assets/` dan dibaca via `AssetManager`.

**iOS (Swift Package Manager)** — Model didaftarkan di `Package.swift` sebagai `.process("Resources/...")` dan diakses via `Bundle.module`.

**iOS (CocoaPods)** — Model didaftarkan di `.podspec` sebagai `s.resources` dan diakses via `Bundle.main`.

> ✅ Setelah `npx cap sync`, model siap digunakan tanpa konfigurasi tambahan.

---

## API

<docgen-index>

* [`extractFaceFeature(...)`](#extractfacefeature)
* [`compareFaces(...)`](#comparefaces)
* [`checkLiveness(...)`](#checkliveness)
* [`detectFaces(...)`](#detectfaces)

</docgen-index>

<docgen-api>
<!--Update the source file JSDoc comments and rerun docgen to update the docs below-->

### extractFaceFeature(...)

```typescript
extractFaceFeature(options: { imageBase64: string; }) => Promise<{ embedding: number[]; }>
```

Mengirim base64 gambar ke Native, mengembalikan array of numbers (embeddings)

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ embedding: number[]; }&gt;</code>

--------------------


### compareFaces(...)

```typescript
compareFaces(options: { vector1: number[]; vector2: number[]; }) => Promise<{ isMatch: boolean; score: number; similarityPercentage: number; }>
```

Membandingkan dua embedding wajah menggunakan Cosine Similarity.
Mengembalikan isMatch (threshold 0.75), score cosine, dan persentase kemiripan.

| Param         | Type                                                   |
| ------------- | ------------------------------------------------------ |
| **`options`** | <code>{ vector1: number[]; vector2: number[]; }</code> |

**Returns:** <code>Promise&lt;{ isMatch: boolean; score: number; similarityPercentage: number; }&gt;</code>

--------------------


### checkLiveness(...)

```typescript
checkLiveness(options: { imageBase64: string; }) => Promise<{ isLive: boolean; score: number; confidence: 'HIGH' | 'MEDIUM' | 'LOW'; }>
```

Memeriksa apakah wajah dalam gambar adalah wajah asli (live) atau spoofing
(foto cetak, layar HP, video, topeng).

Implementasi: Multi-Scale MiniFASNet — inference dua kali (scale 1.0x dan 2.7x),
hasilnya di-average. Threshold 0.75 untuk HRIS production.

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ isLive: boolean; score: number; confidence: 'HIGH' | 'MEDIUM' | 'LOW'; }&gt;</code>

--------------------


### detectFaces(...)

```typescript
detectFaces(options: { imageBase64: string; }) => Promise<{ count: number; faces: { x: number; y: number; width: number; height: number; headEulerAngleY: number; headEulerAngleZ: number; leftEyeOpenProbability: number | null; rightEyeOpenProbability: number | null; }[]; }>
```

Mendeteksi semua wajah dalam gambar tanpa melakukan recognition.
Berguna untuk validasi awal dan blink challenge di layer JS (HRIS production).

iOS: menggunakan VNDetectFaceLandmarksRequest untuk eye landmark (EAR-based).
Android: menggunakan ML Kit dengan CLASSIFICATION_MODE_ALL.

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ count: number; faces: { x: number; y: number; width: number; height: number; headEulerAngleY: number; headEulerAngleZ: number; leftEyeOpenProbability: number | null; rightEyeOpenProbability: number | null; }[]; }&gt;</code>

--------------------

</docgen-api>

---

## Contoh Penggunaan

### Verifikasi wajah dengan liveness check (direkomendasikan)

```typescript
import { FaceRecognition } from 'capacitor-face-recognition-tflite';

async function verifyFaceWithLiveness(imageBase64: string, storedEmbedding: number[]) {
  // Langkah 1: Cek liveness terlebih dahulu untuk mencegah spoofing
  const liveness = await FaceRecognition.checkLiveness({ imageBase64 });

  if (!liveness.isLive) {
    throw new Error(
      `Spoofing terdeteksi! Score: ${liveness.score.toFixed(2)}, ` +
      `Confidence: ${liveness.confidence}`
    );
  }

  // Langkah 2: Ekstrak embedding wajah
  const { embedding } = await FaceRecognition.extractFaceFeature({ imageBase64 });

  // Langkah 3: Bandingkan dengan embedding tersimpan
  const result = await FaceRecognition.compareFaces({
    vector1: embedding,
    vector2: storedEmbedding,
  });

  return {
    isAuthenticated: result.isMatch,
    similarity: result.similarityPercentage,
    livenessScore: liveness.score,
  };
}
```

### Simpan wajah baru (enrollment)

```typescript
async function enrollFace(imageBase64: string): Promise<number[]> {
  // Cek liveness sebelum menyimpan
  const liveness = await FaceRecognition.checkLiveness({ imageBase64 });
  if (!liveness.isLive) throw new Error('Wajah tidak terdeteksi sebagai live');

  // Ekstrak dan simpan embedding
  const { embedding } = await FaceRecognition.extractFaceFeature({ imageBase64 });
  return embedding; // Simpan ke database / AsyncStorage
}
```

### Hanya cek liveness

```typescript
const result = await FaceRecognition.checkLiveness({ imageBase64 });

if (result.isLive) {
  console.log(`✅ Live — Score: ${result.score.toFixed(2)}, Confidence: ${result.confidence}`);
} else {
  console.log(`❌ Spoof — Score: ${result.score.toFixed(2)}, Confidence: ${result.confidence}`);
}
```

---

## Threshold & Konfigurasi

### Face Recognition — Cosine Similarity Threshold

Default threshold adalah `0.75`. Sesuaikan sesuai kebutuhan keamanan:

| Nilai | Penggunaan |
|-------|-----------|
| `0.70` | Toleran — cocok untuk aplikasi non-kritis |
| `0.75` | **Default** — keseimbangan akurasi & kenyamanan |
| `0.80` | Ketat — cocok untuk sistem absensi |
| `0.85` | Sangat ketat — cocok untuk aplikasi keuangan |

**Android** (`FaceRecognitionPlugin.java` baris ~195):
```java
boolean isMatch = cosineSimilarity > 0.75; // ubah nilai ini
```

**iOS** (`FaceRecognitionPlugin.swift` baris ~160):
```swift
let isMatch = cosineSimilarity > 0.75 // ubah nilai ini
```

### Liveness Detection — Score Threshold

Default threshold adalah `0.5`. Naikkan untuk lebih ketat:

**Android** (`FaceRecognitionPlugin.java`):
```java
private static final float LIVENESS_THRESHOLD = 0.6f; // lebih ketat
```

**iOS** (`FaceRecognitionPlugin.swift`):
```swift
private let livenessThreshold: Float = 0.6 // lebih ketat
```

---

## Cara Kerja Internal

### Face Recognition Flow
```
imageBase64 → Bitmap → ML Kit Face Detection → Crop wajah
→ Resize 112×112 → Normalize [-1,1] → MobileFaceNet TFLite
→ Embedding 192-dim → Cosine Similarity
```

### Liveness Detection Flow
```
imageBase64 → Bitmap → ML Kit / Apple Vision → Crop wajah (+20% margin)
→ Resize 80×80 → Normalize [-1,1] → MiniFASNetV1 TFLite
→ Softmax [live, print_spoof, replay_spoof] → isLive (score[0] > 0.5)
```

### Model Specs

| | Face Recognition | Anti-Spoofing |
|-|-----------------|---------------|
| **Model** | MobileFaceNet | MiniFASNetV1 |
| **Input** | `[1, 112, 112, 3]` | `[1, 80, 80, 3]` |
| **Output** | `[1, 192]` embedding | `[1, 3]` softmax |
| **Normalisasi** | `(pixel/128.0) - 1.0` | `(pixel/128.0) - 1.0` |
| **Format** | FLOAT32 | FLOAT32 |

---

## Troubleshooting

### Android — Model tidak ditemukan
```
Error: Model TFLite gagal dimuat. Pastikan file 'mobile_face_net.tflite' ada di folder android/src/main/assets/
```
**Solusi:** Jalankan `npx cap sync` ulang.

### Android — Tidak ada wajah terdeteksi
- Pastikan gambar cukup terang dan wajah terlihat jelas
- Pastikan wajah tidak terlalu kecil (minimal ~20% dari ukuran gambar)
- ML Kit memerlukan Google Play Services — pastikan perangkat mendukung

### iOS — Model tidak ditemukan
```
[FaceRecognitionPlugin] ERROR: File 'mobile_face_net.tflite' tidak ditemukan di bundle.
```
**Solusi:** Jalankan `npx cap sync`. Pastikan menggunakan **SPM** (Package.swift) atau **CocoaPods** (podspec) — tidak keduanya bersamaan.

### Liveness score selalu rendah / tinggi
- Pastikan gambar wajah cukup resolusi (minimal 200×200 px)
- Gunakan gambar frontal (wajah menghadap kamera langsung)
- Hindari kondisi pencahayaan yang sangat buruk

---

## Dependensi Native

### Android
```gradle
implementation 'com.google.mlkit:face-detection:16.1.5'
implementation 'org.tensorflow:tensorflow-lite:2.14.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
```

### iOS (CocoaPods)
```ruby
pod 'TensorFlowLiteSwift'
pod 'Capacitor'
```

### iOS (Swift Package Manager)
Sudah terdaftar di `Package.swift` — tidak perlu konfigurasi tambahan.

---

## License

MIT
