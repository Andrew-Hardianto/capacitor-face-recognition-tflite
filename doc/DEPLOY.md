# Deploy ke npmjs

Paket: `capacitor-face-recognition-tflite`

| Jalur | Versi | dist-tag | Sumber kode | Install |
|-------|-------|----------|-------------|---------|
| **latest** | 1.0.1 (dan seterusnya) | `latest` | branch `fix-bug` (kelak `main`) | `npm i capacitor-face-recognition-tflite` |
| **legacy** | 0.0.1 | `v0` | branch `main` lama (sudah terbit) | `npm i capacitor-face-recognition-tflite@v0` |

Aturan dasar npm yang berlaku di sini:
- Versi yang sudah terbit **tidak bisa** ditimpa atau dipublish ulang.
- `npm publish` tanpa `--tag` otomatis memindahkan tag `latest`. Jalur legacy **wajib** memakai `--tag v0`, kalau tidak, versi lama akan menjadi default install.
- Saat ini 0.0.1 sudah terbit dan menjadi `latest`. Setelah 1.0.1 terbit, `latest` pindah ke 1.0.1.

---

## 0. Persiapan (sekali)

```bash
npm login                 # akun dengan hak publish paket ini
npm whoami                # harus menampilkan username, bukan error
```

Jika akun memakai 2FA, siapkan kode OTP; tambahkan `--otp=<kode>` pada perintah `npm publish` / `npm dist-tag` bila diminta.

---

## 1. Deploy LATEST (1.0.1)

Cek sebelum rilis:

1. Semua perubahan sudah di-commit (skrip menolak working tree kotor).
2. `version` di [package.json](../package.json) sudah naik dan belum ada di npm.
3. [CHANGELOG.md](../CHANGELOG.md) sudah memuat versi tersebut.
4. Verifikasi build:
   ```bash
   npm run verify:web                 # tsc + rollup
   npm run verify:android             # opsional, butuh Android SDK
   npm run verify:ios                 # opsional, butuh Xcode
   ```

Rilis:

```bash
npm run deploy:dry        # simulasi: build + isi tarball, TIDAK publish
npm run deploy            # publish dengan --tag latest
```

`npm run deploy` menjalankan [scripts/deploy.sh](../scripts/deploy.sh):
login dicek → working tree bersih → versi belum ada di npm → `prepublishOnly` (`npm run build`) → `npm publish --access public --tag latest`.

Setelah terbit:

```bash
npm view capacitor-face-recognition-tflite dist-tags     # latest → 1.0.1
git tag v1.0.1 && git push origin v1.0.1                 # tandai rilis di git
```

### Rilis berikutnya (latest)

```bash
npm version patch         # 1.0.1 → 1.0.2 (atau minor / major)
npm run deploy
```

`npm version` membuat commit + tag git sendiri, jadi working tree tetap bersih untuk skrip deploy.

---

## 2. Deploy LEGACY (0.0.1)

0.0.1 **sudah terbit**, jadi tidak perlu dipublish ulang. Yang perlu dilakukan hanya memberi tag `v0` agar mudah dipasang setelah `latest` pindah ke 1.0.1.

```bash
npm run tag:legacy
# = npm dist-tag add capacitor-face-recognition-tflite@0.0.1 v0
npm view capacitor-face-recognition-tflite dist-tags     # v0 → 0.0.1
```

Pemakai yang ingin tetap di perilaku lama:

```bash
npm i capacitor-face-recognition-tflite@v0        # atau @0.0.1 (terkunci pasti)
```

> Catatan: API lama (`extractFaceFeature`, `compareFaces`, `checkLiveness`) di 1.0.1
> **berperilaku sama** dengan 0.0.1, jadi upgrade ke 1.0.1 tidak memutus kode lama.
> Jalur legacy hanya untuk yang butuh paket persis 0.0.1 (mis. tanpa model tambahan,
> ukuran paket lebih kecil).

Perbaikan bug pada versi legacy (0.0.2, dst.): lihat [FIXING.md](FIXING.md).

---

## 3. Rollback / koreksi

| Situasi | Perintah |
|---------|----------|
| `latest` salah menunjuk | `npm dist-tag add capacitor-face-recognition-tflite@<versi> latest` |
| Versi terbit bermasalah | `npm deprecate capacitor-face-recognition-tflite@1.0.1 "alasan; gunakan 1.0.2"` |
| Salah menimpa tag legacy | `npm dist-tag add capacitor-face-recognition-tflite@0.0.1 v0` |
| Hapus tag | `npm dist-tag rm capacitor-face-recognition-tflite <tag>` |

`npm unpublish` hanya mungkin dalam 72 jam setelah terbit dan bila belum dipakai paket lain — lebih aman gunakan `deprecate` lalu rilis versi perbaikan.

---

## 4. Ringkasan perintah

```bash
npm login
npm run deploy:dry     # cek dulu
npm run deploy         # LATEST  (1.0.1)
npm run tag:legacy     # LEGACY  (0.0.1 → tag v0)
```
