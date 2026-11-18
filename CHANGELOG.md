# Changelog

Format per versi. Install versi tertentu: `npm i capacitor-face-recognition-tflite@<versi>`

| Versi | dist-tag | Catatan |
|-------|----------|---------|
| 1.0.1 | `latest` | Pipeline baru + API V2 (di bawah) |
| 0.0.1 | `v0` | Versi awal (perilaku tag `v0.0.1`) — `npm i capacitor-face-recognition-tflite@v0` |

## 1.0.1

API lama **tetap ada dengan perilaku 0.0.1** — tidak ada breaking change.

### Ditambahkan
- `extractFaceFeatureV2`, `compareFacesV2`, `checkLivenessV2` — pipeline baru:
  wajah terbesar, normalisasi orientasi EXIF, dua model anti-spoof (2.7 & 4.0),
  response berisi `label`, `liveScore`, `printScore`, `replayScore`, `threshold`, `thresholdVersion`.
- `verifyFace`, `createChallengeSession`, `verifyFaceSequence`, `captureLivenessSession`, `getDeviceIntegrity`.
- Quality gate, challenge session (single-use + expiry), audit store, deteksi screen-suspect.
- Model `anti_spoof_2_7.tflite` (rename dari `anti_spoof.tflite`) dan `anti_spoof_4_0.tflite`.

### Tidak berubah (perilaku 0.0.1)
- `extractFaceFeature`, `compareFaces`, `checkLiveness` — satu model, multi-scale 1.0 & 2.7, ambang 0.75.

## 0.0.1

- Rilis awal: `extractFaceFeature`, `compareFaces`, `checkLiveness`, `detectFaces`.
