package com.andrew.face;

/**
 * Heuristik artefak layar (P2.2) — SOFT SIGNAL EKSPERIMENTAL, bukan bukti.
 *
 * <p>Memeriksa tiga komponen pada crop wajah (downsample 32x32 ARGB):
 * <ul>
 *   <li>{@code periodicity} — puncak autocorrelation proyeksi luma (pola grid/moiré layar)</li>
 *   <li>{@code flatFraction} — fraksi blok 8x8 nyaris tanpa tekstur (panel datar)</li>
 *   <li>{@code saturationFlat} — fraksi piksel dengan saturasi ~0 (abu/putih terang khas layar)</li>
 * </ul>
 *
 * <p>Skor gabungan = max ketiganya (0..1). Pada rilis pertama sinyal ini HANYA
 * dicatat ke hasil & audit (pengumpulan data) — auto-fail hanya jika
 * {@code config.screenSuspectVeto} di-set eksplisit (mode ketat opt-in).
 * Threshold definitif menyusul setelah ada angka kalibrasi (lihat plan P2.2).
 */
public final class ScreenSuspect {

    private ScreenSuspect() {
    }

    /** Hasil analisis artefak layar pada satu crop wajah. */
    public static final class Result {
        /** Periodisitas piksel (puncak autocorrelation proyeksi luma) — 0..1 */
        public float periodicity;
        /** Fraksi blok 8x8 dengan varians sangat rendah (panel datar) — 0..1 */
        public float flatFraction;
        /** Fraksi piksel dengan saturasi ~0 (abu/putih terang khas layar) — 0..1 */
        public float saturationFlat;
        /** Skor gabungan = max ketiganya — 0..1 */
        public float score;
    }

    /**
     * @param argb piksel ARGB (0xAARRGGBB) hasil downsample crop wajah
     */
    public static Result analyze(int[] argb, int width, int height) {
        Result r = new Result();
        int n = width * height;
        if (argb == null || width <= 0 || height <= 0 || argb.length < n) {
            return r;
        }

        float[] gray = new float[n];
        int satFlat = 0;
        for (int i = 0; i < n; i++) {
            int px = argb[i];
            float red = (px >> 16) & 0xFF;
            float green = (px >> 8) & 0xFF;
            float blue = px & 0xFF;
            gray[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
            float max = Math.max(red, Math.max(green, blue));
            float min = Math.min(red, Math.min(green, blue));
            float sat = max <= 0f ? 0f : (max - min) / max;
            if (sat < 0.06f) {
                satFlat++;
            }
        }
        r.saturationFlat = satFlat / (float) n;

        // Proyeksi luma: rata-rata per baris & per kolom
        float[] rowMean = new float[height];
        for (int y = 0; y < height; y++) {
            float s = 0f;
            for (int x = 0; x < width; x++) {
                s += gray[y * width + x];
            }
            rowMean[y] = s / width;
        }
        float[] colMean = new float[width];
        for (int x = 0; x < width; x++) {
            float s = 0f;
            for (int y = 0; y < height; y++) {
                s += gray[y * width + x];
            }
            colMean[x] = s / height;
        }
        r.periodicity = Math.max(autocorrPeak(rowMean), autocorrPeak(colMean));

        r.flatFraction = flatFraction(gray, width, height);
        r.score = Math.max(r.periodicity, Math.max(r.flatFraction, r.saturationFlat));
        return r;
    }

    /** Puncak autocorrelation ternormalisasi (lag 2..min(12, n/2)) dari proyeksi 1D. */
    static float autocorrPeak(float[] x) {
        int n = x.length;
        if (n < 6) {
            return 0f;
        }
        float mean = 0f;
        for (float v : x) {
            mean += v;
        }
        mean /= n;
        float energy = 0f;
        for (float v : x) {
            energy += (v - mean) * (v - mean);
        }
        if (energy <= 1e-6f) {
            return 0f;
        }
        float best = 0f;
        int maxLag = Math.min(12, n / 2);
        for (int lag = 2; lag <= maxLag; lag++) {
            float acc = 0f;
            for (int i = 0; i + lag < n; i++) {
                acc += (x[i] - mean) * (x[i + lag] - mean);
            }
            float a = acc / energy;
            if (a > best) {
                best = a;
            }
        }
        return best;
    }

    private static float flatFraction(float[] gray, int width, int height) {
        int blocks = 0;
        int flat = 0;
        for (int by = 0; by + 8 <= height; by += 8) {
            for (int bx = 0; bx + 8 <= width; bx += 8) {
                blocks++;
                float mean = 0f;
                for (int y = by; y < by + 8; y++) {
                    for (int x = bx; x < bx + 8; x++) {
                        mean += gray[y * width + x];
                    }
                }
                mean /= 64f;
                float variance = 0f;
                for (int y = by; y < by + 8; y++) {
                    for (int x = bx; x < bx + 8; x++) {
                        float d = gray[y * width + x] - mean;
                        variance += d * d;
                    }
                }
                variance /= 64f;
                if (variance < 4f) {
                    flat++;
                }
            }
        }
        return blocks == 0 ? 0f : flat / (float) blocks;
    }
}
