# Model Sources — Provenance & Integrity

> Dokumen ini mencatat **asal-usul, lisensi, spesifikasi, dan checksum** setiap
> model TFLite yang dibundel plugin ini. Komponen biometrik membutuhkan rantai
> provenance yang dapat diaudit — jangan mengganti/upgrade model tanpa
> memperbarui dokumen ini.
>
> Terakhir diverifikasi: 2026-11-06 (checksum + jejak tensor internal).

## Ringkasan

| File | Fungsi | Arsitektur | Lisensi | Status provenance |
|------|--------|-----------|---------|-------------------|
| `anti_spoof_2_7.tflite` | Anti-spoofing (crop 2.7×) | MiniFASNet (Silent-Face) | Apache 2.0 | ✅ Terverifikasi penuh |
| `anti_spoof_4_0.tflite` | Anti-spoofing (crop 4.0×) | MiniFASNet (Silent-Face) | Apache 2.0 | ✅ Terverifikasi penuh |
| `mobile_face_net.tflite` | Face embedding 192-dim | MobileFaceNet TF (sirius-ai) | Apache 2.0 (kode) | ✅ Kode sumber terkonfirmasi — bobot varian 192-dim (§3) |

Lokasi bundling (salinan identik, diverifikasi via SHA-256):

- `android/src/main/assets/`
- `ios/Sources/FaceRecognitionPlugin/Resources/`

---

## 1–2. `anti_spoof_2_7.tflite` & `anti_spoof_4_0.tflite`

### Rantai provenance

```
minivision-ai/Silent-Face-Anti-Spoofing        (bobot asli MiniFASNet, PyTorch)
  → deepface (deepface.models.spoofing.FasNet) (pembungkusan bobot)
  → shubham0204/OnDevice-Face-Recognition-Android
      (konversi TFLite via ai-edge-torch/ONNX — notebook
       resources/Liveness_PT_Model_to_TF.ipynb;
       file: spoof_model_scale_2_7.tflite / spoof_model_scale_4_0.tflite)
  → plugin ini (di-rename anti_spoof_2_7.tflite / anti_spoof_4_0.tflite)
```

### Bukti verifikasi

- **Jejak tensor internal** di dalam file:
  `...deepface.models.spoofing.FasNetBackbone.MiniFASNet_wrapped...` dan
  `ai_edge_torch.convert.to_channel_last_io...` — sesuai persis dengan proses
  konversi di notebook upstream.
- Pipeline inferensi plugin identik dengan spesifikasi upstream
  (dual-scale crop 2.7×/4.0× dengan logika shift, input 80×80, BGR raw
  `[0..255]` tanpa normalisasi, output 3 kelas `[print, live, replay]`).

### Sumber & lisensi

- MiniFASNet: [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) — **Apache License 2.0**
- Konversi TFLite: [shubham0204/OnDevice-Face-Recognition-Android](https://github.com/shubham0204/OnDevice-Face-Recognition-Android)

### Catatan akurasi (dokumentasi upstream)

| Serangan | Akurasi deteksi (klaim upstream) |
|---|---|
| Foto cetak | ~90% |
| Layar digital | ~85% |
| Replay video sederhana | ~80% |
| Topeng 3D berkualitas | ~50–70% |

Angka ini adalah **alasan arsitektur pertahanan berlapis** plugin (challenge
trajectory + veto per-frame `SPOOF_FRAME_DETECTED` + screen-suspect signal) —
model pasif saja TIDAK cukup untuk kasus replay video.

---

## 3. `mobile_face_net.tflite`

### ✅ Kode sumber terkonfirmasi: [sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF) (Apache 2.0)

Graf model ini **dibangun oleh kode `nets/MobileFaceNet.py` dari repo
sirius-ai/MobileFaceNet_TF** — dibuktikan dengan pencocokan fingerprint nama
op terhadap model pretrained resmi repo itu (`arch/pretrained_model/
MobileFaceNet_9925_9680.pb`, diunduh 2026-11-06):

| Fingerprint nama op | pb sirius-ai | Qualcomm HF | file plugin |
|---|---|---|---|
| `MobileFaceNet/MobileFaceNet/Conv2d_6_InvResBlock_1/…` | 358× ✓ | 0 | 19× ✓ |
| `MobileFaceNet/Logits/LinearConv1x1/…` | 157× ✓ | 0 | 3× ✓ |
| `MobileFaceNet/Logits/SpatialSqueeze…` | 3× ✓ | 0 | 2× ✓ |
| `MobileFaceNet/SeparableConv2d_15/depthwise_weights/…` | 5× ✓ | 0 | 1× ✓ |

Catatan fingerprint: prefix ganda `MobileFaceNet/MobileFaceNet/` adalah hasil
quirk nested `tf.variable_scope` di `mobilenet_v2_base()` repo tersebut, dan
`Conv2d_6_InvResBlock_1` berasal dari pola penamaan `end_point + '_InvResBlock'
+ suffix` di kode yang sama — keduanya mustahil kebetulan.

### Yang tersisa belum diketahui

- **Checkpoint persisnya**: model pretrained repo itu **128-dim**
  (`Logits/LinearConv1x1` shape `(1,1,512,128)`), sedangkan file plugin ini
  **192-dim**. Artinya file ini dibuat dari kode yang sama dengan parameter
  `bottleneck_layer_size=192` — sebuah **varian 192-dim yang beredar luas di
  ekosistem Tiongkok** (node name & dimensi identik dengan model pada artikel
  kalibrasi AIPU "mobilefacenet人脸特征模型"). Siapa yang melatih checkpoint
  192-dim ini dan dengan data apa, belum tercatat.
- Dampak praktis kecil: preprocessing plugin `(pixel/128) - 1.0` vs
  `(pixel − 127.5)/128` yang dipakai beberapa turunan — selisih ~0,004,
  dapat diabaikan, tapi dicatat untuk konsistensi.

### Tindakan yang disarankan

1. ~~Konfirmasikan sumber file~~ — selesai: kode sumber terkonfirmasi (tabel di atas).
2. Jika ingin provenance 100% bersih: bangun ulang TFLite dari checkpoint yang
   dilatih sendiri / berlisensi jelas memakai `nets/MobileFaceNet.py`
   (`bottleneck_layer_size=192`) — **CATATAN**: pergantian bobot = kalibrasi
   ulang `matchThreshold` (0.75) dan seluruh embedding terdaftar harus
   di-enroll ulang.
3. Status lisensi: kode **Apache 2.0** (header `nets/MobileFaceNet.py`:
   "Copyright 2018 The AI boy xsr-ai") — cukup untuk produk HRIS, dengan
   atribusi di README.

### Kandidat yang sudah diperiksa dan DISINGKIRKAN

- **[huggingface.co/qualcomm/MobileFaceNet](https://huggingface.co/qualcomm/MobileFaceNet)** (Qualcomm AI Hub, Apache 2.0) —
  **BUKAN sumber file ini**. Dibuktikan dengan mengunduh rilis
  `mobile_facenet-tflite-float` (v0.66.0) dan membandingkan langsung:

  | | Qualcomm HF | File plugin ini |
  |---|---|---|
  | SHA-256 | `2046ab77…420e5f3d7` | `be4bc7cf…b5354854` |
  | Ukuran | 3,8 MB | 5,0 MB |
  | Output | **128**-dim | **192**-dim |
  | Input layout | `[1,3,112,112]` (NCHW) | `[1,112,112,3]` (NHWC) |
  | Jejak tensor | ekspor QAIRT/ONNX (foamliu/MobileFaceNet, PyTorch) | `MobileFaceNet/Logits/LinearConv1x1` (graf TF klasik) |

  (Diverifikasi 2026-11-06. Catatan: model Qualcomm tetap kandidat pengganti
  yang sah untuk masa depan — terdokumentasi & Apache 2.0 — tetapi 128-dim
  artinya semua embedding harus enroll ulang dan `embeddingSize` di plugin
  berubah dari 192.)

---

## Checksum (SHA-256)

Salinan Android & iOS diverifikasi **identik** (2026-11-06):

```
5d573d20b795a19d7b2a0f67cc4c1130ce8ec30d5f6aa06b2c788a344a49d4dd  anti_spoof_2_7.tflite
32eecfa06cd0cf3c514276fb8521ed820fc256bacf4c616a1f05612d1c156463  anti_spoof_4_0.tflite
be4bc7cfc53f7bc336d0f28b1ab92535f618c913a422b683210750f6b5354854  mobile_face_net.tflite
```

Verifikasi ulang kapan pun:

```sh
shasum -a 256 android/src/main/assets/*.tflite ios/Sources/FaceRecognitionPlugin/Resources/*.tflite
```

---

## Prosedur upgrade model

1. Ganti file di **kedua** lokasi (Android & iOS) dengan file yang sama.
2. Hitung SHA-256 baru → perbarui tabel checksum di dokumen ini + tanggal.
3. Catat sumber (repo/URL + commit/tag) dan lisensi di §1–3.
4. Jalankan ulang kalibrasi threshold (`liveness-v3`: `liveThreshold`,
   `spoofVeto`, `matchThreshold`) via example-app — threshold TIDAK bisa dibawa
   antar model.
5. Uji regresi: fixture unit test + uji replay terukur (lihat
   `doc/REMEDIATION_PLAN.md` §P2 acceptance).
