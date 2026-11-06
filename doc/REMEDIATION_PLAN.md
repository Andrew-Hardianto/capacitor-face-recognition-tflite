# Remediation Plan — Active Challenge & Anti-Replay (liveness-v3)

> Status: **P0-a/P0-b/P1/P2 SELESAI diimplementasi** (P0-c sebagian: label fail-closed + yawTrace/trajectory sudah)
> Catatan QA manual sebelum rilis:
> 1. Kalibrasi yaw di perangkat fisik via example-app (P0-a) — WAJIB.
> 2. Ukur APCER/BPCER replay terukur (P2-d): video korban menoleh/menggeleng di depan kamera, minimal 2 kombinasi perangkat-layar — target gagal ≥ 9/10; user asli ≥ 9/10 lulus (kalau BPCER meledak, turunkan `minLiveVoteRatio` ke 0.8 — veto per-frame tetap aktif).
> Ruang lingkup: **P0 + P1 + P2** · API publik **boleh breaking** · **on-device penuh** (tanpa server)
> Baseline: analisis bug *"menoleh kiri tapi challenge kanan tetap lulus & video bisa lolos"* (Nov 2026)

---

## 0. Ringkasan temuan yang ditangani

| # | Temuan (dari analisis) | Ditangani di |
|---|---|---|
| T1 | Konvensi tanda yaw tidak dikalibrasi; default `yawSign=1` bisa terbalik per platform → arah challenge terbalik | P0 |
| T2 | Preview ter-mirror, frame capture mentah → semantik "kiri/kanan" instruksi ambigu bagi user | P0 |
| T3 | `challenge` bisa dikirim caller via JS (random bisa dilewati) | P0 |
| T4 | Validasi challenge = min/max agregat: tanpa urutan temporal, tanpa eksklusivitas arah → head-shake lolos kedua challenge | P1 |
| T5 | Sampling jarang (5 frame @ 0.3s) → lintasan gerakan tak terukur | P1 |
| T6 | Replay defense hanya model anti-spoof pasif; voting 60% + median mentolerir frame spoof | P2 |
| T7 | `verifyFaceSequence` menerima base64 frame dari JS (injeksi frame) tanpa gerbang eksplisit | P2 |
| T8 | Observability buruk: hasil tidak menyimpan lintasan yaw per frame → QA tak bisa membedakan penyebab gagal | P0/P2 |

**Keputusan desain yang sudah dikunci:**

1. **Breaking change diperbolehkan** → hapus `challenge` dari input JS, bump `thresholdVersion` ke `liveness-v3`, kontrak hasil dirapikan.
2. **On-device penuh** → challenge dihasilkan & disimpan native (single-use + expiry), bukan token server. Jalur upgrade server-side challenge dicatat sebagai fase lanjutan di luar plan ini.
3. **Satu konvensi yaw internal** → semua yaw dinormalkan saat pengukuran (`positive = menoleh ke kanan anatomis user`), sehingga aturan challenge tidak pernah lagi bergantung pada tanda platform.

---

## P0 — Kebenaran arah (semantik yaw & challenge)

**Tujuan:** menoleh ke arah yang salah pasti GAGAL, menoleh ke arah yang diminta pasti LULUS, di kedua platform, tanpa konfigurasi tambahan dari integrator.

### P0.1 Normalisasi tanda yaw di sumber pengukuran
- iOS: `FaceRecognitionPlugin.swift` — `headAnglesDeg(_:)` (line ~1256): kalibrasi tanda aktual `VNFaceObservation.yaw` untuk "menoleh ke kiri anatomis", lalu kalikan konstanta `platformYawSign` di sini.
- Android: `FaceRecognitionPlugin.java` — saat `fr.yaw = face.getHeadEulerAngleY()` (line ~723): terapkan konstanta `PLATFORM_YAW_SIGN` ML Kit.
- Setelah ini semua aturan downstream memakai **satu konvensi**: `yaw > 0 = menoleh ke kanan anatomis user`, `yaw < 0 = kiri`.
- `config.yawSign` **dihapus** dari `LivenessConfig` (breaking) dan diganti `yawSignOverride?` (opsional, hanya untuk perangkat eksotis/diagnostik; tetap diterapkan sebelum agregasi).
- **Aktivitas kalibrasi (wajib, buka blokir P0):** di example-app, uji 5× per perangkat: "dagu ke bahu KIRI Anda" → catat tanda yaw mentah iOS dan Android. Hasilnya mengisi `platformYawSign` masing-masing.

### P0.2 Instruksi eksplisit perspektif + anti-salah-label
- `LivenessCaptureViewController.swift:70-73` dan `LivenessCaptureActivity.java:100`: ganti teks menjadi template berbasis perspektif anatomis + panah:
  - `LOOK_LEFT` → "Dagu ke bahu **KIRI ANDA** ←" · `LOOK_RIGHT` → "Dagu ke bahu **KANAN ANDA** →"
- Hapus pola ternary `challenge == "LOOK_LEFT" ? kiri : kanan` yang diam-diam memetakan nilai tak dikenal ke "ke kanan"; buat tabel mapping eksplisit `switch`/`Map` yang `fail-closed` (challenge tak dikenal → error capture, bukan default).
- Keputusan mirror: preview **tetap mirror** (UX alami), tapi instruksi kini memakai kanvas anatomi sehingga mirror tidak lagi menyesatkan. Dokumentasikan di README: *"kiri/kanan = kiri/kanan tubuh user, bukan sisi layar"*.

### P0.3 Challenge dikeluarkan native, single-use, tak bisa dipilih JS
- `getLivenessChallenge()` → `createChallengeSession(): Promise<{ challengeId, challenge, expiresAt }>`; challenge + nonce + expiry (60 detik) disimpan di state native (per platform).
- `verifyFaceSequence({ images, challengeId, ... })` (JS base64 path) **wajib** `challengeId` aktif; plugin membaca challenge dari native state, bukan dari input. Sesi **dikonsumsi pada percobaan pertama** (gagal maupun lulus) — menutup brute-force retry.
- `captureLivenessSession(options)` **tidak lagi menerima `challenge`**; challenge dibuat internal saat activity/VC dibuka (instruksi ditampilkan native seperti sekarang).
- Tetap pertahankan rate-limit sesi yang ada, pindahkan pengecekan ke atas konsumsi challenge session.

### P0.4 Kontrak hasil liveness-v3 (observability arah)
- Tambah di `VerifyFaceSequenceResult` (dan turunannya):
  - `yawConvention: 'anatomical-v1'` (penanda konvensi yang dipakai),
  - `challengeId`, `challengeConsumed: true`,
  - `yawTrace` (array yaw ternormalisasi per frame + timestamp) **hanya jika `config.debug`** — untuk QA; tanpa debug cukup ringkasan (`startYaw`, `endYaw`, `extremeYaw`, `wrongWayMax`).
- Bump `thresholdVersion` → `liveness-v3` di `FaceRecognitionPlugin.swift:71` dan `FaceRecognitionPlugin.java:94`.
- Update `src/definitions.ts` + README (hapus `yawSign` lama, hapus `challenge` dari opsi capture, dokumen alur `createChallengeSession`).

**Acceptance criteria P0**
- [ ] Uji perangkat: menoleh **salah arah** 10/10 → `passed=false`, `reason ∈ {CHALLENGE_FAILED, CHALLENGE_WRONG_DIRECTION}`; arah benar 10/10 → lulus (asumsi gate lain lolos).
- [ ] Tidak ada jalur API yang menerima nilai challenge dari JS.
- [ ] `challengeId` yang dipakai ulang → reject (`CHALLENGE_EXPIRED` / `CHALLENGE_CONSUMED`).
- [ ] iOS & Android memberi hasil arah identik untuk gerakan fisik yang sama.

---

## P1 — Redesain validasi challenge: lintasan temporal

**Tujuan:** challenge membuktikan *gerakan yang benar, berurutan, satu arah* — bukan kebetulan statistik min/max.

### P1.1 Sampling lebih rapat
- Ubah capture native dari **5 frame @ 300ms** menjadi **12 frame @ ~120ms** (total ±1,5s):
  - iOS: `LivenessCaptureViewController` (panggilan `frameCount: 5, interval: 0.3` di `FaceRecognitionPlugin.swift:883`).
  - Android: `LivenessCaptureActivity` + konstanta `NATIVE_FRAME_COUNT` / interval di `FaceRecognitionPlugin.java` (baris ~122-125, 956).
- Catatan risiko: `AVCapturePhotoOutput`/`ImageCapture` per-shot punya latensi; jika latensi melebihi 120ms di perangkat lambat, fallback yang disetujui: turunkan ke 10 frame @ 150ms. (Opsi lanjutan `AVCaptureVideoDataOutput`/`ImageAnalysis` dicatat, di luar scope P1.)
- Urutkan frame oleh timestamp capture asli, bukan urutan simpan.

### P1.2 Aturan lintasan (evaluator murni, dapat diuji)
Buat modul evaluasi baru yang **pure function** (tanpa dependensi Vision/ML Kit):
- iOS: `ios/Sources/FaceRecognitionPlugin/ChallengeTrajectory.swift`
- Android: `android/src/main/java/com/andrew/face/ChallengeTrajectory.java`

Input: deret yaw ternormalisasi + timestamps + `challenge`. Aturan lulus (semua harus benar):

| Aturan | Parameter (default) | Menolak |
|---|---|---|
| R1 `frontalStart` | 2 frame pertama \|yaw\| < `startFrontalMax` = 8° | Sesi yang dimulai sudah menoleh |
| R2 `deviationSesuai` | ∃ frame dengan yaw ke arah challenge ≥ `challengeYawMin` = 15°, **setelah** fase R1 | Gerakan ke arah lain / tak bergerak |
| R3 `returnFrontal` | 2 frame terakhir \|yaw\| < `returnFrontalMax` = 12° | Menoleh tanpa kembali lurus (video freeze di pose) |
| R4 `dominasiSatuArah` | Deviasi ke arah **salah** < `wrongWayMax` = 12° (LOOK_LEFT → semua yaw < +12°; LOOK_RIGHT → semua yaw > −12°) | **Head-shake kiri-kanan** |
| R5 `gerakanHalus` | ∃ ≥ 3 delta yaw berurutan dengan tanda sesuai arah challenge (median-filter 3 frame dulu untuk noise) | Lompatan noise/1-frame spike |

- Ganti aturan lama `frontalFrames >= 2 di mana saja` + min/max agregat (`FaceRecognitionPlugin.swift:746-771`, `FaceRecognitionPlugin.java:801-833`) sepenuhnya oleh evaluator ini.
- Alasan gagal baru: `CHALLENGE_WRONG_DIRECTION` (R4) dan `CHALLENGE_NO_RETURN` (R3) — dipisah dari `CHALLENGE_FAILED` agar QA/monitoring bisa membedakan.
- Threshold sebagai named constants terpusat per platform (ganti `challengeYawMin`/`challengeFrontMax` lama), disebutkan di README sebagai tabel.

### P1.3 Hasil & diagnostik lintasan
- `VerifyFaceSequenceResult` bertambah: `trajectory: { startYaw, endYaw, extremeYaw, extremeFrame, wrongWayMax, rulesPassed: {R1..R5} }` (ringkasan, selalu ada), `yawTrace` (debug saja).
- `thresholdVersion = 'liveness-v3'` sudah mencakup perubahan semantik ini (satu bump untuk P0+P1).

### P1.4 Unit test lintasan (menggantikan cakupan test nol saat ini)
- iOS: XCTest di `ios/Tests/FaceRecognitionPluginTests/` untuk `ChallengeTrajectory` (sekarang hanya `testEcho`).
- Android: JUnit di `android/src/test/` untuk `ChallengeTrajectory`.
- Fixture wajib (deret yaw sintetis): menoleh kiri benar, menoleh kanan benar, **head-shake**, menoleh tanpa kembali, gerakan < 15°, spike 1 frame, video-loop "terlalu sempurna" (delta konstan).

**Acceptance criteria P1**
- [ ] Semua fixture unit test hijau di 2 platform (aturan identik).
- [ ] Head-shake: 10/10 sesi GAGAL dengan `CHALLENGE_WRONG_DIRECTION` di kedua challenge.
- [ ] Menoleh tanpa kembali lurus: 10/10 GAGAL `CHALLENGE_NO_RETURN`.
- [ ] Gerakan benar: ≥ 9/10 LULUS (BPCER jangan sampai meledak — dipantau via `rulesPassed`).

---

## P2 — Anti-replay hardening

**Tujuan:** video yang diputar di depan kamera (screen replay) tidak lagi hanya bergantung pada klasifier pasif.

### P2.1 Veto per-frame untuk sesi challenge
- Dalam `finishSequence` (iOS) / `finishSequence` (Android): **satu pun frame** dengan `label == "REPLAY"` atau `spoofScore >= spoofHardVeto` (0.5) → sesi langsung gagal (`reason = 'SPOOF_FRAME_DETECTED'`).
- Voting lama tetap ada tapi diperketat untuk sesi: `minLiveVoteRatio` = **1.0** (semua frame `isLive`) untuk challenge session; `verifyFace` single-frame tetap di threshold lama. Konstanta terpisah + terdokumentasi dampak BPCER-nya.

### P2.2 Sinyal screen-suspect (soft signal dulu, bukan auto-fail)
- Pada crop wajah per frame, hitung heuristik artefak layar (ringan, on-device):
  - periodisitas piksel (autocorrelation puncak pada luma → pola grid panel/moiré),
  - anomali saturasi/warna (warna layar vs kulit),
  - ketajaman tepi ekstrem seragam (edge histogram).
- Hasilnya: `screenSuspectScore` per frame + agregat `screenSuspect` di ringkasan sesi, **dicatat ke AuditStore**, TIDAK auto-fail di rilis pertama (kumpulkan data dulu; threshold menyusul setelah ada angka). Konfigurasi `screenSuspectVeto?` sudah disediakan untuk mode ketat (opt-in).

### P2.3 Sinyal timing/behavioral
- Wajibkan durasi sesi dalam rentang gerakan manusia (±1,0–3,0s untuk 12 frame; capture native sudah punya timestamp — pindahkan ke hasil).
- Tandai lintasan "terlalu sempurna" (varians delta yaw ~0 antar segmen) sebagai soft signal `tooSmooth` — video loop cenderung demikian.

### P2.4 Gerbang jalur frame-inject (`verifyFaceSequence` JS base64)
- Dengan breaking change diizinkan: jalur base64 **hanya aktif pada build debug** (cek `DeviceIntegrity.isDebuggable` / flag inisialisasi plugin `enableInsecureFramePath`), dan hasil selalu membawa `native: false`.
- Produksi wajib memakai `captureLivenessSession` (frame native, sudah anti-inject). README memindahkan `verifyFaceSequence` ke bagian "Development & testing only".
- `createChallengeSession` + `challengeId` single-use (P0.3) tetap berlaku di jalur ini — dua lapis pertahanan.

### P2.5 Audit trail untuk pembuktian & tuning
- `AuditStore` (kedua platform): simpan ringkasan `trajectory`, `screenSuspect`, `SPOOF_FRAME_DETECTED` flags, versi threshold — agar setiap keputusan lulus/gagal bisa direkonstruksi pasca-insiden.

**Acceptance criteria P2**
- [ ] Replay video korban (menoleh/menggeleng) di depan kamera: **gagal ≥ 9/10** sesi di minimal 2 kombinasi perangkat-layar (diukur dengan setup uji sederhana, bukan asumsi).
- [ ] Head-shake video (dua arah): gagal via P1 (R4) — buktikan replay dua arah tidak lagi menolong.
- [ ] Foto cetak & layar statis: tetap ditolak oleh gate lama (regresi nol).
- [ ] User asli (gerakan benar, live): ≥ 9/10 lulus (BPCER terkendali; kalibrasi via slider example-app).
- [ ] Setiap sesi punya audit trail yang cukup untuk menjelaskan keputusan.

---

## 4. Peta perubahan file

| File | P0 | P1 | P2 |
|---|---|---|---|
| `ios/Sources/FaceRecognitionPlugin/FaceRecognitionPlugin.swift` | normalisasi yaw (`headAnglesDeg`), hapus `yawSign`/`challenge` input, challenge session state, kontrak hasil v3 | panggil evaluator lintasan, hapus aturan min/max, alasan gagal baru | veto per-frame, `minLiveVoteRatio`, timing check |
| `ios/Sources/FaceRecognitionPlugin/ChallengeTrajectory.swift` | — | **baru**: evaluator lintasan murni | (dipakai untuk sinyal `tooSmooth`) |
| `ios/Sources/FaceRecognitionPlugin/LivenessCaptureViewController.swift` | instruksi perspektif + fail-closed label | 12 frame @ 120ms, timestamps | timestamp ke hasil |
| `android/src/main/java/com/andrew/face/FaceRecognitionPlugin.java` | paritas iOS | paritas iOS | paritas iOS |
| `android/src/main/java/com/andrew/face/ChallengeTrajectory.java` | — | **baru** | — |
| `android/src/main/java/com/andrew/face/LivenessCaptureActivity.java` | instruksi perspektif | frame count/interval | timestamps |
| `android/.../AuditStore.java`, `ios/.../AuditStore.swift` | — | — | audit trajectory + screenSuspect |
| `src/definitions.ts` | kontrak v3 (breaking) | field `trajectory`, alasan gagal baru | `screenSuspect`, `native`, gating |
| `README.md` | konvensi yaw, alur challenge session | tabel threshold lintasan | panduan replay test, jalur dev-only |
| `ios/Tests/...`, `android/src/test/...` | test normalisasi yaw | test lintasan (fixture wajib) | test veto per-frame |
| `example-app/src/js/example.js` | halaman kalibrasi tanda yaw | visualisasi lintasan + `rulesPassed` | tombol mode replay-test |

---

## 5. Urutan eksekusi & estimasi

| Fase | Isi | Estimasi | Blokir |
|---|---|---|---|
| **P0-a** | Kalibrasi tanda yaw per platform (aktivitas device, isi konstanta) | 0,5 hari | — (blokir semua!) |
| **P0-b** | Normalisasi yaw + hapus `yawSign`/`challenge` input + challenge session state | 2 hari | P0-a |
| **P0-c** | Instruksi perspektif + kontrak hasil v3 + README/definitions | 1 hari | P0-b |
| **P1-a** | `ChallengeTrajectory` + unit test fixture (kedua platform) | 2 hari | P0-b |
| **P1-b** | Integrasi evaluator + hapus aturan min/max + sampling 12 frame | 1,5 hari | P1-a |
| **P1-c** | Uji perangkat: shake/wrong-way/no-return matrix | 1 hari | P1-b |
| **P2-a** | Veto per-frame + `minLiveVoteRatio` + alasan `SPOOF_FRAME_DETECTED` | 1 hari | P1-b |
| **P2-b** | Gerbang `verifyFaceSequence` (dev-only) + challenge single-use ketat | 1 hari | P0-b |
| **P2-c** | Heuristik screen-suspect (soft signal) + timing `tooSmooth` + audit | 2 hari | P2-a |
| **P2-d** | Uji replay video terukur (2 kombinasi perangkat-layar) | 1 hari | P2-c |

Total kasar: **±11–12 hari kerja** untuk P0–P2 lengkap termasuk uji.

---

## 6. Risiko & mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Tanda yaw ternyata berbeda antar versi OS / device vendor | Arah terbalik lagi | `yawSignOverride` + smoke-test `maxYaw/minYaw` di example-app; konvensi ternormalisasi membuat override jadi 1 titik saja |
| Latensi capture 120ms tidak terkejar di perangkat low-end | Lintasan renggang | Fallback 10 frame @ 150ms; evaluasi R5 toleran (median filter) |
| Veto per-frame + `minLiveVoteRatio` 1.0 menaikkan BPCER | User asli gagal | Ukur dulu di example-app; threshold veto & ratio jadi konfigurasi; soft signal screen-suspect tidak auto-fail di rilis pertama |
| Heuristik screen-suspect false-positive (layar bagus, kulit terang) | Salah veto | Soft signal dulu, threshold menyusul setelah data audit terkumpul |
| Behavior change liveness-v3 memutus konsumen lama | Integrasi rusak | Breaking disetujui; catat migrasi di README (`challenge` dihapus → `createChallengeSession`, `yawSign` dihapus → otomatis) |

---

## 7. Di luar scope (fase lanjutan)

1. **Server-side signed challenge** (token + capture window sempit) — butuh backend; upgrade path dari `challengeId` lokal sudah disiapkan di kontrak v3.
2. **Program pengukuran APCER/BPCER penuh** (dataset replay/Print, laporan resmi) — README §Fase 5; P2-d hanya uji terukur terbatas untuk acceptance.
3. **Ganti capture ke video stream** (`AVCaptureVideoDataOutput`/`ImageAnalysis`) bila latensi foto membatasi sampling.
4. **Depth/3D & sensor fusion** (ToF, TrueDepth) — arah jangka panjang untuk replay defense sejati.

---

## 8. Checklist sign-off

- [ ] P0: uji arah salah/benar 10/10 di iOS + Android
- [ ] P1: fixture unit test hijau; shake/no-return/wrong-way tertolak
- [ ] P2: replay video gagal ≥ 9/10 (2 kombinasi layar); user asli ≥ 9/10 lulus
- [ ] README + definitions.ts konsisten dengan perilaku v3
- [ ] Audit trail tersedia untuk setiap keputusan sesi
