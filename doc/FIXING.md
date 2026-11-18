# Fixing bug per versi

Paket: `capacitor-face-recognition-tflite`

Dokumen ini untuk **memperbaiki bug** lalu merilis hasilnya. Cara rilis rutin, login,
rollback, dan daftar dist-tag ada di [DEPLOY.md](DEPLOY.md).

| Jalur | Versi | dist-tag | Dikerjakan di | Dirilis dengan |
|-------|-------|----------|---------------|----------------|
| **latest** | 1.x | `latest` | branch kerja (`fix-bug`) | `npm run deploy` |
| **legacy** | 0.0.x | `v0` | branch `legacy/0.x` (dari `main`) | `npm publish --tag v0` |

Aturan penting: patch legacy **wajib** `--tag v0`. `npm run deploy` memakai
`--tag latest` dan akan menimpa versi terbaru dengan versi lama.

---

## A. Fix bug di LATEST (1.x)

```bash
git checkout fix-bug
# ...perbaiki kode, uji...
npm run verify:web                  # opsional: verify:android / verify:ios
git commit -am "fix: <deskripsi>"
# tambahkan catatan di CHANGELOG.md, commit
npm version patch                   # 1.0.1 → 1.0.2 (commit + tag)
npm run deploy:dry && npm run deploy
```

Jika bug-nya juga ada di legacy, lanjut ke bagian C.

---

## B. Fix bug di LEGACY (0.0.x)

Kerjakan di `main` lewat worktree terpisah supaya branch kerja tidak terganggu.

```bash
# 1. Siapkan worktree dari main
git worktree add ../face-legacy -b legacy/0.x main
cd ../face-legacy
npm ci

# 2. Perbaiki kode (ios/, android/, src/), lalu uji
npm run verify:web                  # opsional: verify:android / verify:ios

# 3. Naikkan versi (membuat commit + tag v0.0.2). Tetap 0.0.x, jangan 1.x
npm version 0.0.2 -m "fix legacy: <deskripsi>"

# 4. Cek, lalu publish. WAJIB --tag v0
npm publish --access public --tag v0 --dry-run
npm publish --access public --tag v0
npm view capacitor-face-recognition-tflite dist-tags   # latest tetap 1.x, v0 → 0.0.2

# 5. Push dan bersihkan
git push origin legacy/0.x --tags
cd - && git worktree remove ../face-legacy
```

Jika branch `legacy/0.x` sudah ada (patch kedua dan seterusnya), pakai
`git worktree add ../face-legacy legacy/0.x` tanpa `-b`.

---

## C. Sinkronkan perbaikan antar versi

`extractFaceFeature`, `compareFaces`, dan `checkLiveness` di 1.x adalah salinan
perilaku `main`. Bug di tiga method itu hampir pasti ada di kedua jalur, jadi
perbaiki di satu sisi lalu bawa ke sisi lain:

```bash
git checkout fix-bug                      # atau main, bila 1.x sudah di-merge
git cherry-pick <hash-commit-perbaikan>   # konflik mungkin terjadi
npm version patch && npm run deploy       # rilis 1.0.2
```

Perhatikan per method:

| Method | Cherry-pick dari legacy ke 1.x |
|--------|--------------------------------|
| `extractFaceFeature` | Biasanya bisa, kodenya sama persis dengan `main` (Android: ditambah `synchronized`) |
| `compareFaces` | Biasanya bisa |
| `checkLiveness` | **Perlu ditulis ulang.** Di 1.x ia memakai `runAntiSpoofInferenceLegacy` dan `antiSpoof27`, bukan `antiSpoofInterpreter` seperti `main` |
| `...V2`, `verifyFace`, dst. | Hanya ada di 1.x — perbaiki di jalur A saja |

Bug yang hanya ada di pipeline baru (`...V2`, `verifyFace`, `verifyFaceSequence`,
`captureLivenessSession`) tidak perlu dibawa ke legacy.

---

## D. Checklist sebelum rilis perbaikan

- [ ] Bug direproduksi, perbaikan diuji (unit test atau `example-app`).
- [ ] `npm run verify:web` lolos.
- [ ] Versi naik di jalur yang benar (0.0.x untuk legacy, 1.x untuk latest).
- [ ] Legacy: `--tag v0`. Latest: `npm run deploy`.
- [ ] Setelah publish: `npm view capacitor-face-recognition-tflite dist-tags` menunjukkan tag yang benar.
- [ ] Perbaikan sudah dibawa ke jalur lain bila bug-nya dipakai bersama (bagian C).
