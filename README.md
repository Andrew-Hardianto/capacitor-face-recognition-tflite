# capacitor-face-recognition-tflite

Capacitor plugin for **Face Recognition** + **Anti-Spoofing (Passive Liveness Detection)** using ML Kit & TensorFlow Lite.

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
| `@capacitor/core` | `>= 7.0.0` |

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

> ⚠️ ML Kit Face Detection requires **Google Play Services** to be installed. Devices without Google Play Services (pure AOSP) are not supported.

### iOS

| Requirement | Minimum |
|-------------|---------|
| **iOS** | `15.0` |
| **Swift** | `5.1` |
| **Xcode** | `15.0+` |
| `TensorFlowLiteSwift` (via CocoaPods) | depends on plugin version |
| `TensorFlowLite` (via SPM) | `>= 2.14.0` |

> ℹ️ Face detection uses the built-in **Apple Vision** framework in iOS — no additional external libraries are required.

---

## Install

```bash
npm install capacitor-face-recognition-tflite
npx cap sync
```

---

## Model Files

The plugin includes both TFLite models directly in the package — **no manual download is required**.

| Model | Platform | Size | Function |
|-------|----------|--------|--------|
| `mobile_face_net.tflite` | Android & iOS | ~5.0 MB | Face recognition (192-dim embedding) |
| `anti_spoof.tflite` | Android & iOS | ~5.7 MB | Anti-spoofing / Liveness detection |

### How auto-bundling works

**Android** — Models are copied to `android/src/main/assets/` and read via `AssetManager`.

**iOS (Swift Package Manager)** — Models are registered in `Package.swift` as `.process("Resources/...")` and accessed via `Bundle.module`.

**iOS (CocoaPods)** — Models are registered in `.podspec` as `s.resources` and accessed via `Bundle.main`.

> ✅ After running `npx cap sync`, the models are ready to use without additional configuration.

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

Sends base64 image to Native, returning an array of numbers (embeddings)

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ embedding: number[]; }&gt;</code>

--------------------


### compareFaces(...)

```typescript
compareFaces(options: { vector1: number[]; vector2: number[]; }) => Promise<{ isMatch: boolean; score: number; similarityPercentage: number; }>
```

Compares two face embeddings using Cosine Similarity.
Returns isMatch (0.75 threshold), cosine score, and similarity percentage.

| Param         | Type                                                   |
| ------------- | ------------------------------------------------------ |
| **`options`** | <code>{ vector1: number[]; vector2: number[]; }</code> |

**Returns:** <code>Promise&lt;{ isMatch: boolean; score: number; similarityPercentage: number; }&gt;</code>

--------------------


### checkLiveness(...)

```typescript
checkLiveness(options: { imageBase64: string; }) => Promise<{ isLive: boolean; score: number; confidence: 'HIGH' | 'MEDIUM' | 'LOW'; }>
```

Checks if the face in the image is a real (live) face or a spoof
(printed photo, phone screen, video, mask).

Implementation: Multi-Scale MiniFASNet — inference twice (1.0x and 2.7x scales),
results are averaged. Threshold 0.75.

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ isLive: boolean; score: number; confidence: 'HIGH' | 'MEDIUM' | 'LOW'; }&gt;</code>

--------------------


### detectFaces(...)

```typescript
detectFaces(options: { imageBase64: string; }) => Promise<{ count: number; faces: { x: number; y: number; width: number; height: number; headEulerAngleY: number; headEulerAngleZ: number; leftEyeOpenProbability: number | null; rightEyeOpenProbability: number | null; }[]; }>
```

Detects all faces in the image without performing recognition.
Useful for early validation and blink challenge in JS layer.

iOS: uses VNDetectFaceLandmarksRequest for eye landmarks (EAR-based).
Android: uses ML Kit with CLASSIFICATION_MODE_ALL.

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ count: number; faces: { x: number; y: number; width: number; height: number; headEulerAngleY: number; headEulerAngleZ: number; leftEyeOpenProbability: number | null; rightEyeOpenProbability: number | null; }[]; }&gt;</code>

--------------------

</docgen-api>

---

## Usage Examples

### Face verification with liveness check (recommended)

```typescript
import { FaceRecognition } from 'capacitor-face-recognition-tflite';

async function verifyFaceWithLiveness(imageBase64: string, storedEmbedding: number[]) {
  // Step 1: Check liveness first to prevent spoofing
  const liveness = await FaceRecognition.checkLiveness({ imageBase64 });

  if (!liveness.isLive) {
    throw new Error(
      `Spoofing detected! Score: ${liveness.score.toFixed(2)}, ` +
      `Confidence: ${liveness.confidence}`
    );
  }

  // Step 2: Extract face embedding
  const { embedding } = await FaceRecognition.extractFaceFeature({ imageBase64 });

  // Step 3: Compare with stored embedding
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

### Save new face (enrollment)

```typescript
async function enrollFace(imageBase64: string): Promise<number[]> {
  const liveness = await FaceRecognition.checkLiveness({ imageBase64 });
  if (!liveness.isLive) throw new Error('Face is not detected as live');

  const { embedding } = await FaceRecognition.extractFaceFeature({ imageBase64 });
  return embedding; 
}
```

### Liveness check only

```typescript
const result = await FaceRecognition.checkLiveness({ imageBase64 });

if (result.isLive) {
  console.log(`✅ Live — Score: ${result.score.toFixed(2)}, Confidence: ${result.confidence}`);
} else {
  console.log(`❌ Spoof — Score: ${result.score.toFixed(2)}, Confidence: ${result.confidence}`);
}
```

---

## Thresholds & Configuration

### Face Recognition — Cosine Similarity Threshold

Default threshold is `0.75`. Adjust according to your security needs:

| Value | Usage |
|-------|-----------|
| `0.70` | Tolerant — suitable for non-critical apps |
| `0.75` | **Default** — balance of accuracy & convenience |
| `0.80` | Strict — suitable for attendance systems |
| `0.85` | Very strict — suitable for financial apps |

**Android** (`FaceRecognitionPlugin.java` line ~195):
```java
boolean isMatch = cosineSimilarity > 0.75; // change this value
```

**iOS** (`FaceRecognitionPlugin.swift` line ~160):
```swift
let isMatch = cosineSimilarity > 0.75 // change this value
```

### Liveness Detection — Score Threshold

Default threshold is `0.5`. Increase for stricter validation:

**Android** (`FaceRecognitionPlugin.java`):
```java
private static final float LIVENESS_THRESHOLD = 0.6f; 
```

**iOS** (`FaceRecognitionPlugin.swift`):
```swift
private let livenessThreshold: Float = 0.6 
```

---

## Internal Workflow

### Face Recognition Flow
```
imageBase64 → Bitmap → ML Kit Face Detection → Crop face
→ Resize 112×112 → Normalize [-1,1] → MobileFaceNet TFLite
→ 192-dim Embedding → Cosine Similarity
```

### Liveness Detection Flow
```
imageBase64 → Bitmap → ML Kit / Apple Vision → Crop face (+20% margin)
→ Resize 80×80 → Normalize [-1,1] → MiniFASNetV1 TFLite
→ Softmax [live, print_spoof, replay_spoof] → isLive (score[0] > 0.5)
```

### Model Specs

| | Face Recognition | Anti-Spoofing |
|-|-----------------|---------------|
| **Model** | MobileFaceNet | MiniFASNetV1 |
| **Input** | `[1, 112, 112, 3]` | `[1, 80, 80, 3]` |
| **Output** | `[1, 192]` embedding | `[1, 3]` softmax |
| **Normalization** | `(pixel/128.0) - 1.0` | `(pixel/128.0) - 1.0` |
| **Format** | FLOAT32 | FLOAT32 |

---

## Troubleshooting

### Android — Model not found
```
Error: Failed to load TFLite model. Make sure 'mobile_face_net.tflite' exists in android/src/main/assets/ folder
```
**Solution:** Run `npx cap sync` again.

### Android — No face detected
- Ensure the image is well-lit and the face is clearly visible
- Ensure the face is not too small (minimum ~20% of image size)
- ML Kit requires Google Play Services — ensure the device supports it

### iOS — Model not found
```
[FaceRecognitionPlugin] ERROR: File 'mobile_face_net.tflite' not found in bundle.
```
**Solution:** Run `npx cap sync`. Ensure you are using either **SPM** (Package.swift) or **CocoaPods** (podspec) — not both simultaneously.

### Liveness score is always low / high
- Ensure the face image has sufficient resolution (minimum 200×200 px)
- Use frontal images (face looking directly at the camera)
- Avoid extremely poor lighting conditions

---

## Native Dependencies

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
Already registered in `Package.swift` — no additional configuration required.

---

## Acknowledgements

- **Anti-Spoofing Model**: The liveness detection utilizes the MiniFASNet model originating from [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), which is licensed under the Apache License 2.0.

---

## License

MIT
