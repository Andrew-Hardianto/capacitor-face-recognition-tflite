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
| `com.google.ai.edge.litert:litert` | `2.3.2` (LiteRT — pengganti TFLite, mendukung 16 KB page size) |

> ℹ️ Uses bundled ML Kit (`com.google.mlkit:face-detection`) & local TensorFlow Lite runtime. Fully functional on offline and non-Google Play Services devices (e.g. Huawei HMS, pure AOSP).

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
| `anti_spoof_2_7.tflite` | Android & iOS | ~5.7 MB | Anti-spoofing / Liveness (crop 2.7x) |
| `anti_spoof_4_0.tflite` | Android & iOS | ~6.1 MB | Anti-spoofing / Liveness (crop 4.0x) |

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
* [`verifyFace(...)`](#verifyface)
* [`createChallengeSession()`](#createchallengesession)
* [`verifyFaceSequence(...)`](#verifyfacesequence)
* [`captureLivenessSession(...)`](#capturelivenesssession)
* [`getDeviceIntegrity()`](#getdeviceintegrity)
* [Interfaces](#interfaces)
* [Type Aliases](#type-aliases)

</docgen-index>

<docgen-api>
<!--Update the source file JSDoc comments and rerun docgen to update the docs below-->

### extractFaceFeature(...)

```typescript
extractFaceFeature(options: { imageBase64: string; }) => Promise<{ embedding: number[]; }>
```

Sends base64 image to Native, returning an array of numbers (embeddings).
Embedding size: 192 (MobileFaceNet).

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ embedding: number[]; }&gt;</code>

--------------------


### compareFaces(...)

```typescript
compareFaces(options: { vector1: number[]; vector2: number[]; }) => Promise<{ isMatch: boolean; score: number; similarityPercentage: number; threshold: number; thresholdVersion: string; }>
```

Compares two face embeddings using Cosine Similarity.
Returns isMatch (0.75 default threshold, NOT yet calibrated), cosine score,
similarity percentage, and threshold metadata.

| Param         | Type                                                   |
| ------------- | ------------------------------------------------------ |
| **`options`** | <code>{ vector1: number[]; vector2: number[]; }</code> |

**Returns:** <code>Promise&lt;{ isMatch: boolean; score: number; similarityPercentage: number; threshold: number; thresholdVersion: string; }&gt;</code>

--------------------


### checkLiveness(...)

```typescript
checkLiveness(options: { imageBase64: string; }) => Promise<{ isLive: boolean; score: number; confidence: 'HIGH' | 'MEDIUM' | 'LOW'; label: LivenessLabel; liveScore: number; printScore: number; replayScore: number; thresholdVersion: string; }>
```

Checks if the face in the image is a real (live) face or a spoof
(printed photo, phone screen, video).

Implementation (liveness-v2, Silent-Face reference pipeline):
two models (crop 2.7x and 4.0x with shift, BGR raw input), summed softmax,
3-class argmax ([PRINT, LIVE, REPLAY]) plus liveScore threshold and spoof veto.

NOTE: the "score" field is now mapped to post-argmax liveScore — its values
are NOT comparable with the previous (liveness-v1) implementation.

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ isLive: boolean; score: number; confidence: 'HIGH' | 'MEDIUM' | 'LOW'; label: <a href="#livenesslabel">LivenessLabel</a>; liveScore: number; printScore: number; replayScore: number; thresholdVersion: string; }&gt;</code>

--------------------


### detectFaces(...)

```typescript
detectFaces(options: { imageBase64: string; }) => Promise<{ count: number; faces: { x: number; y: number; width: number; height: number; headEulerAngleX: number; headEulerAngleY: number; headEulerAngleZ: number; leftEyeOpenProbability: number | null; rightEyeOpenProbability: number | null; }[]; }>
```

Detects all faces in the image without performing recognition.
Useful for early validation and active challenge in JS layer.

iOS: uses VNDetectFaceLandmarksRequest for eye landmarks (EAR-based).
Android: uses ML Kit with CLASSIFICATION_MODE_ALL.
Angles are in DEGREES on both platforms (iOS converts from radians).
headEulerAngleY (yaw) is NORMALIZED: positive = user's anatomical right,
negative = user's anatomical left (same convention as the challenge).

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ count: number; faces: { x: number; y: number; width: number; height: number; headEulerAngleX: number; headEulerAngleY: number; headEulerAngleZ: number; leftEyeOpenProbability: number | null; rightEyeOpenProbability: number | null; }[]; }&gt;</code>

--------------------


### verifyFace(...)

```typescript
verifyFace(options: { imageBase64: string; referenceEmbedding?: number[]; config?: LivenessConfig; }) => Promise<VerifyFaceResult>
```

Unified verification (TOCTOU-safe): quality gate + liveness + embedding,
all computed from the SAME frame and bounding box. The embedding is only
returned when isLive is true.

| Param         | Type                                                                                                                        |
| ------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **`options`** | <code>{ imageBase64: string; referenceEmbedding?: number[]; config?: <a href="#livenessconfig">LivenessConfig</a>; }</code> |

**Returns:** <code>Promise&lt;<a href="#verifyfaceresult">VerifyFaceResult</a>&gt;</code>

--------------------


### createChallengeSession()

```typescript
createChallengeSession() => Promise<{ challengeId: string; challenge: ChallengeType; expiresAt: number; }>
```

Creates a native challenge session (random direction, single-use, 60s expiry).
Show the instruction to the user, capture frames, then call verifyFaceSequence
with the returned challengeId. The challenge direction CANNOT be chosen
from JS — it is always generated natively.

**Returns:** <code>Promise&lt;{ challengeId: string; challenge: <a href="#challengetype">ChallengeType</a>; expiresAt: number; }&gt;</code>

--------------------


### verifyFaceSequence(...)

```typescript
verifyFaceSequence(options: { images: string[]; challengeId: string; referenceEmbedding?: number[]; config?: LivenessConfig; }) => Promise<VerifyFaceSequenceResult>
```

Multi-frame verification with active challenge (helper in plugin).

DEV/QA-ONLY-BY-DEFAULT: this path accepts base64 frames from JS (a
frame-injection surface) and is therefore only available on DEBUG builds,
OR when the host app explicitly opts in via the Capacitor config flag
`plugins.FaceRecognition.enableInsecureFramePath: true` (legacy webview
integrations that accept the risk). Production should prefer
captureLivenessSession. The result always carries `native: false`.

Pass &gt;= 5 frames captured over ~1.5 seconds. Passing rules:
- every frame must pass the quality gate
- anti-replay (P2): NO frame may be a hard spoof (label PRINT/REPLAY or
  spoofScore &gt;= 0.5 → SPOOF_FRAME_DETECTED), the session duration must look
  human (TIMING_SUSPECT), and screenSuspectVeto, when configured, must not
  be reached (SCREEN_SUSPECTED)
- live voting: `minLiveVoteRatio` of frames must be isLive (default 1.0 =
  ALL frames) AND median liveScore &gt;= liveThreshold
- identity consistent among pose-comparable frames (|yaw| &lt;= 12 deg):
  cosine &gt;= identityThreshold (default 0.7) vs the MOST FRONTAL frame.
  Frames with |yaw| &gt; 12 deg serve the challenge only and are excluded —
  their embeddings are not pose-invariant.
- challenge satisfied — verified as a TEMPORAL TRAJECTORY (not min/max):
  R1 start near frontal (|yaw| &lt; 8 deg) → R2 deviation &gt;= 15 deg toward the
  requested direction (yaw is normalized: positive = user's anatomical
  right; LOOK_LEFT needs a NEGATIVE deviation) → R3 return near frontal
  (|yaw| &lt; 12 deg), with R4 no sustained movement toward the wrong
  direction (&lt; 12 deg — rejects head-shakes) and R5 sustained smooth
  motion (&gt;= 3 same-direction deltas, median-3 filtered).
  Failure reasons: CHALLENGE_WRONG_DIRECTION (R4), CHALLENGE_NO_RETURN (R3),
  CHALLENGE_FAILED (R1/R2/R5).

The challenge direction is read from the native challenge session
(single-use) — never from JS input.

| Param         | Type                                                                                                                                          |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **`options`** | <code>{ images: string[]; challengeId: string; referenceEmbedding?: number[]; config?: <a href="#livenessconfig">LivenessConfig</a>; }</code> |

**Returns:** <code>Promise&lt;<a href="#verifyfacesequenceresult">VerifyFaceSequenceResult</a>&gt;</code>

--------------------


### captureLivenessSession(...)

```typescript
captureLivenessSession(options: { referenceEmbedding?: number[]; config?: LivenessConfig; }) => Promise<NativeSessionResult>
```

Native liveness session (anti frame-injection, Fase 4).

Opens the camera natively (CameraX / AVFoundation), captures &gt;= 5 frames
over ~1.5s WITHOUT passing frames through JS, then runs the full
verification pipeline (quality gate + liveness + active challenge).
Only the RESULT is returned — never the raw frames.

The session is rate-limited (max 10 attempts per 5 minutes) and logged to
an encrypted local audit store for HRIS fraud review.

The challenge direction is ALWAYS generated natively at random — it cannot
be chosen by the caller.

Requires camera permission (requested automatically).

| Param         | Type                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| **`options`** | <code>{ referenceEmbedding?: number[]; config?: <a href="#livenessconfig">LivenessConfig</a>; }</code> |

**Returns:** <code>Promise&lt;<a href="#nativesessionresult">NativeSessionResult</a>&gt;</code>

--------------------


### getDeviceIntegrity()

```typescript
getDeviceIntegrity() => Promise<{ deviceIntegrity: DeviceIntegrity; }>
```

Device integrity signals (Fase 4) — root/emulator/hook heuristics.
These are RISK SIGNALS, not proof. Use them to trigger extra checks or
audit logging, not as a single gate.

**Returns:** <code>Promise&lt;{ deviceIntegrity: <a href="#deviceintegrity">DeviceIntegrity</a>; }&gt;</code>

--------------------


### Interfaces


#### VerifyFaceResult

| Prop                   | Type                                                              | Description                                           |
| ---------------------- | ----------------------------------------------------------------- | ----------------------------------------------------- |
| **`isLive`**           | <code>boolean</code>                                              |                                                       |
| **`thresholdVersion`** | <code>string</code>                                               |                                                       |
| **`liveness`**         | <code><a href="#livenessresult">LivenessResult</a> \| null</code> | null jika quality gate gagal                          |
| **`quality`**          | <code><a href="#qualityresult">QualityResult</a></code>           |                                                       |
| **`embedding`**        | <code>number[]</code>                                             | HANYA diisi jika isLive — dari frame & bbox yang sama |
| **`match`**            | <code><a href="#matchresult">MatchResult</a></code>               | Hanya jika referenceEmbedding diberikan dan isLive    |
| **`debug`**            | <code>{ probs27: number[]; probs40: number[]; }</code>            | Hanya jika config.debug                               |


#### LivenessResult

| Prop                     | Type                                                    | Description                                                              |
| ------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------ |
| **`label`**              | <code><a href="#livenesslabel">LivenessLabel</a></code> |                                                                          |
| **`liveScore`**          | <code>number</code>                                     | Probabilitas live (0..1)                                                 |
| **`printScore`**         | <code>number</code>                                     | Probabilitas print spoof (0..1)                                          |
| **`replayScore`**        | <code>number</code>                                     | Probabilitas replay spoof (0..1)                                         |
| **`screenSuspectScore`** | <code>number</code>                                     | Sinyal artefak layar per frame (P2.2) — soft signal eksperimental (0..1) |


#### QualityResult

| Prop            | Type                                                    |
| --------------- | ------------------------------------------------------- |
| **`passed`**    | <code>boolean</code>                                    |
| **`reason`**    | <code><a href="#qualityreason">QualityReason</a></code> |
| **`faceRatio`** | <code>number</code>                                     |
| **`blur`**      | <code>number</code>                                     |


#### MatchResult

| Prop                   | Type                 |
| ---------------------- | -------------------- |
| **`isMatch`**          | <code>boolean</code> |
| **`score`**            | <code>number</code>  |
| **`threshold`**        | <code>number</code>  |
| **`thresholdVersion`** | <code>string</code>  |


#### LivenessConfig

Konfigurasi liveness & quality gate — semua opsional (default lihat README).

| Prop                    | Type                 | Description                                                                                                                                                                                                                                                                                      |
| ----------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`liveThreshold`**     | <code>number</code>  | Threshold liveScore minimum. Default 0.80                                                                                                                                                                                                                                                        |
| **`spoofVeto`**         | <code>number</code>  | Veto: maksimum skor spoof yang ditoleransi. Default 0.20                                                                                                                                                                                                                                         |
| **`minFrameSide`**      | <code>number</code>  | Sisi terpendek frame minimum (px). Default 480                                                                                                                                                                                                                                                   |
| **`minFaceRatio`**      | <code>number</code>  | faceWidth / frameWidth minimum. Default 0.25                                                                                                                                                                                                                                                     |
| **`edgeMargin`**        | <code>number</code>  | Margin tepi frame (0..1). Default 0.05                                                                                                                                                                                                                                                           |
| **`maxPose`**           | <code>number</code>  | Maksimum \|yaw\|/\|pitch\|/\|roll\| (derajat). Default 20                                                                                                                                                                                                                                        |
| **`blurMin`**           | <code>number</code>  | Variance of Laplacian minimum (perlu kalibrasi). Default 50                                                                                                                                                                                                                                      |
| **`exposureMin`**       | <code>number</code>  | Mean luminance minimum. Default 60                                                                                                                                                                                                                                                               |
| **`exposureMax`**       | <code>number</code>  | Mean luminance maksimum. Default 200                                                                                                                                                                                                                                                             |
| **`identityThreshold`** | <code>number</code>  | Threshold cosine konsistensi identitas antar frame sebanding. Default 0.7                                                                                                                                                                                                                        |
| **`yawSignOverride`**   | <code>number</code>  | Override tanda yaw per-perangkat (1 atau -1) untuk kasus buffer kamera ter-flip (mis. beberapa Samsung/OnePlus). Default 1. Konvensi arah normal sudah dinormalisasi plugin (yaw positif = menoleh ke kanan anatomis user) — jangan isi kecuali halaman "Kalibrasi Yaw" example-app menyuruhnya. |
| **`minLiveVoteRatio`**  | <code>number</code>  | Rasio minimum frame yang harus isLive per sesi (P2.1). Default 1.0 = SEMUA frame harus isLive. Turunkan (mis. 0.8) jika pengukuran BPCER menunjukkan user asli sering gagal — veto per-frame SPOOF_FRAME_DETECTED tetap aktif.                                                                   |
| **`screenSuspectVeto`** | <code>number</code>  | Veto screen-suspect per sesi (P2.2), 0..1. Default tidak di-set = soft signal saja (skor tetap dilaporkan di hasil & audit). Set eksplisit untuk mode ketat: sesi gagal (SCREEN_SUSPECTED) jika skor &gt;= veto.                                                                                 |
| **`debug`**             | <code>boolean</code> | Kembalikan data debug (probs per model). Default false                                                                                                                                                                                                                                           |


#### VerifyFaceSequenceResult

| Prop                     | Type                                                                    | Description                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`passed`**             | <code>boolean</code>                                                    |                                                                                                                                                       |
| **`reason`**             | <code><a href="#sequencefailurereason">SequenceFailureReason</a></code> |                                                                                                                                                       |
| **`challenge`**          | <code><a href="#challengetype">ChallengeType</a></code>                 |                                                                                                                                                       |
| **`challengeId`**        | <code>string</code>                                                     | Present on verifyFaceSequence results (from createChallengeSession).                                                                                  |
| **`native`**             | <code>boolean</code>                                                    | false = jalur JS base64 (dev-only), true = native capture (produksi)                                                                                  |
| **`medianLiveScore`**    | <code>number</code>                                                     |                                                                                                                                                       |
| **`liveVotes`**          | <code>number</code>                                                     |                                                                                                                                                       |
| **`frameCount`**         | <code>number</code>                                                     |                                                                                                                                                       |
| **`maxYaw`**             | <code>number</code>                                                     | Max normalized yaw across quality-passed frames (positive = user's anatomical right)                                                                  |
| **`minYaw`**             | <code>number</code>                                                     | Min normalized yaw across quality-passed frames (negative = user's anatomical left)                                                                   |
| **`frontalFrames`**      | <code>number</code>                                                     |                                                                                                                                                       |
| **`identityConsistent`** | <code>boolean</code>                                                    |                                                                                                                                                       |
| **`minIdentityCosine`**  | <code>number \| null</code>                                             | Cosine minimum antara frame sebanding terhadap frame referensi (paling frontal). null jika kurang dari 2 frame sebanding. Untuk diagnostik kalibrasi. |
| **`trajectory`**         | <code><a href="#trajectorysummary">TrajectorySummary</a></code>         | Ringkasan lintasan challenge (aturan R1–R5 + statistik yaw)                                                                                           |
| **`screenSuspect`**      | <code>number</code>                                                     | Sinyal artefak layar agregat (P2.2) — max antar frame; soft signal kecuali screenSuspectVeto di-set (mode ketat → SCREEN_SUSPECTED).                  |
| **`yawTrace`**           | <code>{ index: number; yaw: number; timestampMs?: number; }[]</code>    | Trace yaw per frame — HANYA jika config.debug (build debug)                                                                                           |
| **`thresholdVersion`**   | <code>string</code>                                                     |                                                                                                                                                       |
| **`match`**              | <code><a href="#matchresult">MatchResult</a></code>                     |                                                                                                                                                       |


#### TrajectorySummary

Ringkasan lintasan challenge (P1) — selalu ada di hasil sesi.

| Prop                | Type                                                                    | Description                                                                                     |
| ------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **`startYaw`**      | <code>number</code>                                                     |                                                                                                 |
| **`endYaw`**        | <code>number</code>                                                     |                                                                                                 |
| **`extremeYaw`**    | <code>number</code>                                                     | Deviasi ekstrem ke arah challenge                                                               |
| **`extremeFrame`**  | <code>number</code>                                                     |                                                                                                 |
| **`wrongWayMax`**   | <code>number</code>                                                     | Deviasi teramati ke arah SALAH (diagnostik)                                                     |
| **`deltaVariance`** | <code>number</code>                                                     | Varians magnitudo delta yaw — mendekati 0 menandai gerakan konstan sempurna (dugaan video-loop) |
| **`tooSmooth`**     | <code>boolean</code>                                                    | Lintasan "terlalu sempurna" (deltaVariance ≈ 0) — soft signal dugaan video-loop (P2.3)          |
| **`durationMs`**    | <code>number</code>                                                     | Durasi sesi dalam ms (0 jika timestamp tidak tersedia)                                          |
| **`rulesPassed`**   | <code><a href="#trajectoryrulespassed">TrajectoryRulesPassed</a></code> |                                                                                                 |


#### TrajectoryRulesPassed

Hasil per-aturan lintasan challenge (P1) — lihat ChallengeTrajectory.

| Prop                  | Type                 | Description                                                                            |
| --------------------- | -------------------- | -------------------------------------------------------------------------------------- |
| **`frontalStart`**    | <code>boolean</code> | R1 — frame awal mendekati frontal (\|yaw\| &lt; 8°)                                    |
| **`deviation`**       | <code>boolean</code> | R2 — ada deviasi ≥ 15° ke arah challenge, setelah fase awal                            |
| **`returnFrontal`**   | <code>boolean</code> | R3 — frame akhir kembali mendekati frontal (\|yaw\| &lt; 12°)                          |
| **`singleDirection`** | <code>boolean</code> | R4 — tanpa gerakan berkelanjutan ke arah SALAH (deviasi &lt; 12°) — menolak head-shake |
| **`smoothMotion`**    | <code>boolean</code> | R5 — gerakan halus berkelanjutan (≥ 3 delta searah pada nilai ter-filter median-3)     |


#### NativeSessionResult

Hasil captureLivenessSession — frame diambil native, tidak melewati JS.

| Prop                  | Type                                                        | Description                                                                 |
| --------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------- |
| **`native`**          | <code>true</code>                                           | Selalu true — penanda bahwa frame diambil native                            |
| **`deviceIntegrity`** | <code><a href="#deviceintegrity">DeviceIntegrity</a></code> |                                                                             |
| **`auditId`**         | <code>string \| null</code>                                 | ID jejak audit terenkripsi lokal (frame + hasil tersimpan untuk audit HRIS) |


#### DeviceIntegrity

Sinyal risiko perangkat (Fase 4) — heuristik sisi klien, BUKAN bukti.
Keputusan akhir tetap di aplikasi/server; jangan jadikan gerbang tunggal.

| Prop                | Type                 | Description                                                       |
| ------------------- | -------------------- | ----------------------------------------------------------------- |
| **`isEmulator`**    | <code>boolean</code> | Berjalan di emulator / simulator                                  |
| **`isRooted`**      | <code>boolean</code> | Perangkat rooted / jailbroken (heuristik)                         |
| **`isDebuggable`**  | <code>boolean</code> | Build debuggable (development)                                    |
| **`hooksDetected`** | <code>boolean</code> | Framework hook terdeteksi (Xposed / Frida / Substrate, heuristik) |


### Type Aliases


#### LivenessLabel

<code>'LIVE' | 'PRINT' | 'REPLAY'</code>


#### QualityReason

<code>'NO_FACE' | 'MULTIPLE_FACES' | 'LOW_RESOLUTION' | 'FACE_TOO_SMALL' | 'FACE_AT_EDGE' | 'BAD_POSE' | 'BLURRY' | 'BAD_LIGHTING'</code>


#### ChallengeType

<code>'LOOK_LEFT' | 'LOOK_RIGHT'</code>


#### SequenceFailureReason

<code>'QUALITY_FAILED' | 'IDENTITY_INCONSISTENT' | 'CHALLENGE_FAILED' | 'CHALLENGE_WRONG_DIRECTION' | 'CHALLENGE_NO_RETURN' | 'SPOOF_FRAME_DETECTED' | 'SCREEN_SUSPECTED' | 'TIMING_SUSPECT' | 'NOT_LIVE_ENOUGH'</code>

</docgen-api>

---

## Usage Examples

### Face verification with liveness check (recommended — TOCTOU-safe)

Gunakan `verifyFace` — liveness, quality gate, dan embedding dihitung dari **frame
dan bbox yang sama**, dan embedding hanya dikembalikan jika `isLive`. Jangan memanggil
`checkLiveness` lalu `extractFaceFeature` secara terpisah: base64 yang berbeda antar
panggilan membuka celah TOCTOU (frame live untuk liveness, frame spoof untuk embedding).

```typescript
import { FaceRecognition } from 'capacitor-face-recognition-tflite';

async function verifyFaceWithLiveness(imageBase64: string, storedEmbedding: number[]) {
  const result = await FaceRecognition.verifyFace({
    imageBase64,
    referenceEmbedding: storedEmbedding,
  });

  if (!result.quality.passed) {
    throw new Error(`Quality gate failed: ${result.quality.reason}`);
  }
  if (!result.isLive) {
    throw new Error(
      `Spoofing detected! label=${result.liveness?.label} ` +
      `liveScore=${result.liveness?.liveScore.toFixed(2)}`
    );
  }

  // result.embedding hanya terisi karena isLive = true
  return {
    isAuthenticated: result.match?.isMatch ?? false,
    similarity: result.match?.score ?? 0,
    embedding: result.embedding,
  };
}
```

### Save new face (enrollment)

Enrollment adalah titik paling kritis — embedding palsu yang masuk akan terus
dianggap valid. Gunakan standar yang lebih ketat dari verifikasi harian: multi-frame
dengan active challenge.

```typescript
async function enrollFace(images: string[]): Promise<number[]> {
  // images: ≥ 5 frame yang diambil dalam ~1.5 detik
  const { challengeId, challenge } = await FaceRecognition.createChallengeSession();
  // tampilkan instruksi challenge ke user (arah TIDAK bisa dipilih dari JS),
  // lalu tangkap frame-frame sesi sebelum challengeId kedaluwarsa (60 detik)

  const session = await FaceRecognition.verifyFaceSequence({ images, challengeId });
  if (!session.passed) throw new Error(`Enrollment liveness gagal: ${session.reason}`);

  // Ambil embedding dari frame pertama yang lolos (verifyFace untuk frame yang sama)
  const result = await FaceRecognition.verifyFace({ imageBase64: images[0] });
  if (!result.isLive || !result.embedding) throw new Error('Face is not detected as live');
  return result.embedding;
}
```

### Liveness check only (single frame)

```typescript
const result = await FaceRecognition.checkLiveness({ imageBase64 });

if (result.isLive) {
  console.log(`✅ Live — label=${result.label}, liveScore=${result.liveScore.toFixed(2)}`);
} else {
  console.log(`❌ Spoof — label=${result.label}, printScore=${result.printScore.toFixed(2)}, replayScore=${result.replayScore.toFixed(2)}`);
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

**Android** (`FaceRecognitionPlugin.java`):
```java
private static final double MATCH_THRESHOLD = 0.75; // change this value
```

**iOS** (`FaceRecognitionPlugin.swift`):
```swift
private let matchThreshold: Double = 0.75 // change this value
```

### Liveness Detection — Threshold (satu sumber kebenaran)

Keputusan live (liveness-v2): `p = softmax(out27) + softmax(out40)` (rentang 0..2),
`label = argmax(p)`, `liveScore = p[1]/2`, `spoofScore = max(p[0],p[2])/2`,
lalu `isLive = label === 'LIVE' && liveScore >= liveThreshold && spoofScore < spoofVeto`.

| Konstanta | Default | Arti |
|-----------|---------|------|
| `liveThreshold` | `0.80` | Minimum liveScore |
| `spoofVeto` | `0.20` | Maksimum skor spoof yang ditoleransi |

Default bisa di-override per panggilan lewat `config` pada `verifyFace` / `verifyFaceSequence`
(untuk kalibrasi). Setiap hasil menyertakan `thresholdVersion` (`liveness-v2`) —
skor **tidak sebanding** dengan implementasi lama (liveness-v1, rata-rata skor antar scale).

Konstanta di kode: `LIVE_THRESHOLD` / `SPOOF_VETO` (Android: `FaceRecognitionPlugin.java`,
iOS: `FaceRecognitionPlugin.swift`).

---

## Defense in Depth (Fase 4)

Lapisan pertahanan untuk skenario HRIS dengan risiko fraud tinggi:

### Native capture — `captureLivenessSession()`
Kamera dibuka **oleh plugin** (CameraX / AVFoundation), frame diambil dan disimpan
ke file privat **tanpa melewati layer JS**, lalu seluruh pipeline verifikasi
(quality gate + liveness + challenge) dijalankan terhadap frame tersebut.
Yang dikembalikan ke JavaScript hanya **hasil**, tidak pernah frame mentah —
menutup celah injeksi frame dari aplikasi yang dimodifikasi.

Response = `VerifyFaceSequenceResult` + `native`, `deviceIntegrity`, `auditId`.

### Device integrity — `getDeviceIntegrity()`
Heuristik root/jailbreak, emulator/simulator, build debuggable, dan framework
hook (Xposed/Frida/Substrate). **Ini sinyal risiko, bukan bukti** — gunakan untuk
memicu pemeriksaan tambahan atau pencatatan audit, bukan sebagai gerbang tunggal.
Attestation terverifikasi server (Play Integrity / App Attest) di luar scope plugin.

### Jejak audit terenkripsi
Setiap sesi native menyimpan hasil + frame yang dipakai dalam bentuk
AES-256-GCM — Android: kunci di AndroidKeyStore; iOS: kunci di Keychain
(`WhenUnlockedThisDeviceOnly`), file di Application Support dengan proteksi
file penuh. Untuk pemeriksaan fraud HRIS. Hak hapus (UU PDP): hapus folder
`liveness_audit`.

### Pembatasan lain
- **Rate limiting**: maksimal 10 percobaan sesi per 5 menit (`RATE_LIMITED`).
- **Debug gating**: output `config.debug` hanya dikembalikan pada build
  debuggable/DEBUG — tidak pernah dari build rilis.

---

## Calibration & Testing (Fase 5)

Halaman uji & kalibrasi tersedia di `example-app/` (`npm start` di folder itu).
Fasilitasnya:

1. **Kalibrasi Yaw (P0-a)** — tombol "Mulai Kalibrasi Yaw": ikuti instruksi
   (dagu ke bahu kiri, lalu kanan). Konvensi liveness-v3: **yaw negatif = menoleh
   ke kiri anatomis Anda**. Jika hasil TERBALIK (buffer kamera ter-flip, mis.
   beberapa Samsung/OnePlus), tool otomatis set `yawSignOverride: -1` di konfigurasi —
   laporkan temuan itu agar konstanta platform diperbaiki. Wajib diverifikasi per
   perangkat sebelum memakai active challenge.
2. **Kalibrasi threshold** — slider `liveThreshold`, `spoofVeto`, `blurMin`, dll.
   langsung dikirim ke `verifyFace` sebagai `LivenessConfig` (tanpa rebuild).
3. **Uji single frame** — `checkLiveness`, `verifyFace`, `detectFaces` pada frame
   kamera atau file gambar. Untuk regression test: muat screenshot yang dulu lolos,
   pastikan kini **ditolak** dengan label PRINT/REPLAY.
4. **Sesi multi-frame** — `createChallengeSession` + `verifyFaceSequence` otomatis
   (5 frame, ~1,5 detik).
5. **Dataset logger** — labeli hasil uji sebagai LIVE / REPLAY / PRINT, lalu ekspor
   JSON/CSV. Estimasi APCER/BPCER ditampilkan dari log (angka sementara — untuk
   laporan resmi: pisahkan set kalibrasi dan set uji, laporkan APCER per jenis
   serangan, minimal beberapa ratus sampel per jenis).

Target awal: APCER ≤ 5% dan BPCER ≤ 10%. Threshold dipilih dari kurva hasil uji,
bukan ditebak. Threshold match cosine (0.75) juga perlu dikalibrasi sendiri
(FAR/FRR) dengan data Anda.

---

## Internal Workflow

### Face Recognition Flow
```
imageBase64 → Bitmap (orientasi EXIF dinormalisasi) → ML Kit / Apple Vision → Crop wajah
→ Resize 112×112 → Normalize [-1,1] → MobileFaceNet TFLite
→ 192-dim Embedding → Cosine Similarity
```

### Liveness Detection Flow (liveness-v2, port referensi Silent-Face)
```
imageBase64 → Bitmap (orientasi EXIF dinormalisasi) → ML Kit / Apple Vision → bbox wajah terbesar
→ Quality gate (resolusi, ukuran wajah, tepi frame, pose, blur, exposure)
→ Crop 2.7x (shift, w/h terpisah) → Resize 80×80 → BGR raw [0..255] → Model 2.7 ─┐
→ Crop 4.0x (shift, w/h terpisah) → Resize 80×80 → BGR raw [0..255] → Model 4.0 ─┤
                                                                                  ↓
                          sum softmax → argmax [PRINT, LIVE, REPLAY] + liveScore + spoof veto
                                                                                  ↓
                              isLive + label + liveScore/printScore/replayScore
```

### Model Specs

| | Face Recognition | Anti-Spoofing |
|-|-----------------|---------------|
| **Model** | MobileFaceNet | Silent-Face MiniFASNet (2 model) |
| **File** | `mobile_face_net.tflite` | `anti_spoof_2_7.tflite`, `anti_spoof_4_0.tflite` |
| **Input** | `[1, 112, 112, 3]` | `[1, 80, 80, 3]` per model |
| **Output** | `[1, 192]` embedding | `[1, 3]` logits per model |
| **Kelas output** | — | `[print_spoof, live, replay_spoof]` (index 1 = LIVE) |
| **Preprocessing** | RGB, `(pixel/128.0) - 1.0` | BGR raw `[0..255]` (tanpa normalisasi) |
| **Crop** | bbox wajah | 2.7x / 4.0x dengan logika shift (referensi `_get_new_box`) |
| **Format** | FLOAT32 | FLOAT32 |

Model anti-spoof berasal dari [shubham0204/OnDevice-Face-Recognition-Android](https://github.com/shubham0204/OnDevice-Face-Recognition-Android)
(`spoof_model_scale_2_7.tflite` dan `spoof_model_scale_4_0.tflite`).

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
- Ensure the face orientation is upright or within acceptable tilt angles

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
implementation 'com.google.ai.edge.litert:litert:2.3.2'
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

- **Anti-Spoofing Model**: The liveness detection utilizes the MiniFASNet model originating from [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), which is licensed under the Apache License 2.0. The TFLite conversion is from [shubham0204/OnDevice-Face-Recognition-Android](https://github.com/shubham0204/OnDevice-Face-Recognition-Android).
- **Face Recognition Model**: The face embedding uses the MobileFaceNet architecture (192-dim, input 112×112) built with [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF) (Apache License 2.0) — verified via graph-op fingerprint matching against that repository's pretrained model; the bundled weights are a 192-dim checkpoint variant of the same codebase. Full provenance record and SHA-256 checksums: [`doc/MODEL_SOURCES.md`](doc/MODEL_SOURCES.md).

---

## License

MIT
