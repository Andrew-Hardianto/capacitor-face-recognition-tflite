# capacitor-face-recognition-tflite

Capacitor plugin untuk Face Recognition + **Anti-Spoofing (Liveness Detection)** menggunakan ML Kit & TFLite.

## Fitur

| Fitur | Android | iOS |
|-------|---------|-----|
| Ekstraksi embedding wajah | ✅ | ✅ |
| Perbandingan wajah (Cosine Similarity) | ✅ | ✅ |
| **Anti-Spoofing / Liveness Detection** | ✅ | ✅ |

---

## Install

```bash
npm install capacitor-face-recognition-tflite
npx cap sync
```

---

## Setup Model

Plugin ini memerlukan **dua model TFLite**:

### 1. Model Face Recognition (`mobile_face_net.tflite`)
Model MobileFaceNet untuk ekstraksi embedding wajah.

### 2. Model Anti-Spoofing (`anti_spoof.tflite`) ← **Baru!**
Model MiniFASNet untuk deteksi liveness (wajah asli vs spoofing).

**Download model anti-spoofing:**
```
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
```
Gunakan model versi TFLite (export ke `.tflite`) atau cari versi siap pakai di community releases.

---

### Android — Letakkan model di Assets

```
android/app/src/main/assets/
├── mobile_face_net.tflite   ← face recognition
└── anti_spoof.tflite        ← anti-spoofing (BARU)
```

### iOS — Tambahkan model ke Xcode Bundle

1. Buka Xcode → pilih target app
2. Drag & drop `anti_spoof.tflite` ke dalam project
3. Pastikan **"Add to targets"** dicentang
4. Verifikasi di **Build Phases → Copy Bundle Resources** ada `anti_spoof.tflite`

---

## API

<docgen-index>

* [`extractFaceFeature(...)`](#extractfacefeature)
* [`compareFaces(...)`](#comparefaces)
* [`checkLiveness(...)`](#checkliveness)

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

Memeriksa apakah wajah dalam gambar adalah wajah asli (live) atau spoofing (foto cetak, layar HP, video, topeng).

Memerlukan model `anti_spoof.tflite` di folder assets (Android) atau bundle (iOS).

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:**
| Field | Type | Keterangan |
|-------|------|-----------|
| `isLive` | `boolean` | `true` = wajah asli, `false` = terdeteksi spoofing |
| `score` | `number` | Skor liveness antara `0.0` (spoof) hingga `1.0` (live) |
| `confidence` | `'HIGH' \| 'MEDIUM' \| 'LOW'` | Tingkat kepercayaan hasil deteksi |

**Confidence levels:**
- `HIGH` — score > 0.85 atau < 0.15 (hasil sangat yakin)
- `MEDIUM` — score > 0.6 atau < 0.4
- `LOW` — score antara 0.4–0.6 (ambang batas, perlu review manual)

--------------------

</docgen-api>

---

## Contoh Penggunaan

### Face Recognition + Liveness Check (dikombinasikan)

```typescript
import { FaceRecognition } from 'capacitor-face-recognition-tflite';

async function verifyFaceWithLiveness(imageBase64: string, storedEmbedding: number[]) {
  // 1. Cek liveness terlebih dahulu
  const liveness = await FaceRecognition.checkLiveness({ imageBase64 });

  if (!liveness.isLive) {
    throw new Error(`Spoofing terdeteksi! Score: ${liveness.score.toFixed(2)}, Confidence: ${liveness.confidence}`);
  }

  console.log(`✅ Liveness OK — Score: ${liveness.score.toFixed(2)}, Confidence: ${liveness.confidence}`);

  // 2. Baru ekstrak embedding dan bandingkan wajah
  const { embedding } = await FaceRecognition.extractFaceFeature({ imageBase64 });

  const result = await FaceRecognition.compareFaces({
    vector1: embedding,
    vector2: storedEmbedding,
  });

  return {
    isAuthenticated: result.isMatch,
    similarity: result.similarityPercentage,
    isLive: liveness.isLive,
    livenessScore: liveness.score,
  };
}
```

### checkLiveness saja

```typescript
const result = await FaceRecognition.checkLiveness({ imageBase64: myBase64Image });

console.log(result.isLive);       // true / false
console.log(result.score);        // 0.0 – 1.0
console.log(result.confidence);   // "HIGH" | "MEDIUM" | "LOW"
```

---

## Threshold

### Anti-Spoofing
Default threshold adalah `0.5`. Untuk aplikasi dengan keamanan tinggi, naikkan threshold di kode native:

**Android** (`FaceRecognitionPlugin.java`):
```java
private static final float LIVENESS_THRESHOLD = 0.6f; // lebih ketat
```

**iOS** (`FaceRecognitionPlugin.swift`):
```swift
private let livenessThreshold: Float = 0.6 // lebih ketat
```

### Face Recognition
Default threshold adalah `0.75` (Cosine Similarity).
