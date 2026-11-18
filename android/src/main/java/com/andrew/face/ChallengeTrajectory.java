package com.andrew.face;

import java.util.List;

/**
 * Evaluator lintasan challenge (liveness-v3, P1) — PURE FUNCTION, dapat diuji
 * tanpa Vision/ML Kit.
 *
 * <p>Menggantikan aturan lama berbasis min/max agregat (yang bisa dilewati oleh
 * head-shake dan tidak memvalidasi urutan gerakan). Input: deret yaw
 * TERNORMALISASI ({@link YawConvention}: positif = menoleh ke kanan anatomis
 * user) beserta timestamp per frame, dalam urutan capture.
 *
 * <p>Aturan lulus (semua harus benar):
 * <ul>
 *   <li>R1 {@code frontalStart} — frame awal harus frontal (|yaw| &lt; 8°)</li>
 *   <li>R2 {@code deviation} — ∃ deviasi ≥ 15° ke arah challenge, SETELAH fase R1</li>
 *   <li>R3 {@code returnFrontal} — frame akhir kembali mendekati frontal (|yaw| &lt; 12°)</li>
 *   <li>R4 {@code singleDirection} — deviasi ke arah SALAH &lt; 12° (menolak head-shake)</li>
 *   <li>R5 {@code smoothMotion} — ∃ run ≥ 3 delta searah (pada nilai ter-filter
 *       median-3) dengan magnitudo ≥ 1°/frame (menolak spike 1 frame)</li>
 * </ul>
 *
 * <p>R1/R2/R3 dihitung dari nilai mentah (frame sudah lolos quality gate);
 * R4/R5 dari nilai ter-filter median-3 agar tahan spike noise tetapi tetap
 * menolak gerakan dua arah yang berkelanjutan.
 *
 * <p>Alasan gagal (prioritas): R4 → {@code CHALLENGE_WRONG_DIRECTION};
 * R3 → {@code CHALLENGE_NO_RETURN}; selain itu {@code CHALLENGE_FAILED}.
 */
public final class ChallengeTrajectory {

    private ChallengeTrajectory() {
    }

    /** Satu sampel yaw ternormalisasi (positif = menoleh ke kanan anatomis user). */
    public static final class Sample {
        public final float yaw;
        /** Epoch milliseconds; -1 jika tidak diketahui (jalur JS base64). */
        public final double timestampMs;

        public Sample(float yaw, double timestampMs) {
            this.yaw = yaw;
            this.timestampMs = timestampMs;
        }
    }

    /** Threshold aturan lintasan — SATU sumber kebenaran per platform. */
    public static final class Thresholds {
        public float startFrontalMax = 8f;      // R1: |yaw| maksimum frame awal
        public int startFrontalFrames = 2;      // R1: jumlah frame awal yang harus frontal
        public float challengeYawMin = 15f;     // R2: deviasi minimum ke arah challenge
        public float returnFrontalMax = 12f;    // R3: |yaw| maksimum frame akhir
        public int returnFrontalFrames = 2;     // R3: jumlah frame akhir yang harus kembali
        public float wrongWayMax = 12f;         // R4: deviasi maksimum ke arah SALAH
        public int smoothRunLength = 3;         // R5: run delta searah minimum
        public float smoothDeltaMin = 1f;       // R5: floor magnitudo delta (anti noise)
        public float tooSmoothMax = 0.05f;      // P2.3: ambang deltaVariance "terlalu sempurna"
    }

    /** Hasil evaluasi — semua angka yaw ternormalisasi. */
    public static final class Result {
        public boolean passed;
        /** null jika lulus; selain itu CHALLENGE_WRONG_DIRECTION | CHALLENGE_NO_RETURN | CHALLENGE_FAILED */
        public String failureReason;
        public boolean frontalStart;     // R1
        public boolean deviation;        // R2
        public boolean returnFrontal;    // R3
        public boolean singleDirection;  // R4
        public boolean smoothMotion;     // R5
        public float startYaw;
        public float endYaw;
        public float extremeYaw;
        public int extremeFrame = -1;
        /** Deviasi teramati ke arah SALAH (untuk diagnostik). */
        public float wrongWayObserved;
        /** Varians magnitudo delta yaw — sinyal "terlalu sempurna" untuk P2.3 (tooSmooth). */
        public float deltaVariance;
        /** Lintasan "terlalu sempurna" (deltaVariance ≈ 0) — soft signal dugaan video-loop (P2.3). */
        public boolean tooSmooth;
        /** Durasi sesi dalam ms; 0 jika timestamp tidak diketahui. */
        public double durationMs;
    }

    public static Result evaluate(List<Sample> samples, String challenge) {
        return evaluate(samples, challenge, new Thresholds());
    }

    public static Result evaluate(List<Sample> samples, String challenge, Thresholds t) {
        Result r = new Result();
        boolean lookLeft = "LOOK_LEFT".equals(challenge);
        int n = samples == null ? 0 : samples.size();
        if (n == 0) {
            r.passed = false;
            r.failureReason = "CHALLENGE_FAILED";
            return r;
        }

        float[] raw = new float[n];
        for (int i = 0; i < n; i++) {
            raw[i] = samples.get(i).yaw;
        }
        float[] filtered = medianFilter3(raw);

        // R1 — frontalStart: frame awal harus frontal
        r.frontalStart = true;
        for (int i = 0; i < Math.min(t.startFrontalFrames, n); i++) {
            if (Math.abs(raw[i]) >= t.startFrontalMax) {
                r.frontalStart = false;
                break;
            }
        }

        // R2 — deviation: ∃ deviasi ≥ challengeYawMin ke arah challenge, setelah fase R1
        float maxDev = -Float.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            float dev = lookLeft ? -raw[i] : raw[i];
            if (dev > maxDev) {
                maxDev = dev;
                r.extremeYaw = raw[i];
                r.extremeFrame = i;
            }
        }
        r.deviation = false;
        for (int i = t.startFrontalFrames; i < n; i++) {
            float dev = lookLeft ? -raw[i] : raw[i];
            if (dev >= t.challengeYawMin) {
                r.deviation = true;
                break;
            }
        }

        // R3 — returnFrontal: frame akhir kembali mendekati frontal
        r.returnFrontal = true;
        int from = Math.max(0, n - t.returnFrontalFrames);
        for (int i = from; i < n; i++) {
            if (Math.abs(raw[i]) >= t.returnFrontalMax) {
                r.returnFrontal = false;
                break;
            }
        }

        // R4 — singleDirection: deviasi ke arah SALAH < wrongWayMax (nilai ter-filter)
        float minF = Float.MAX_VALUE;
        float maxF = -Float.MAX_VALUE;
        for (float v : filtered) {
            minF = Math.min(minF, v);
            maxF = Math.max(maxF, v);
        }
        r.wrongWayObserved = lookLeft ? Math.max(0f, maxF) : Math.max(0f, -minF);
        r.singleDirection = r.wrongWayObserved < t.wrongWayMax;

        // R5 — smoothMotion: run delta searah pada nilai ter-filter + deltaVariance
        float[] deltas = new float[Math.max(0, n - 1)];
        for (int i = 0; i < deltas.length; i++) {
            deltas[i] = filtered[i + 1] - filtered[i];
        }
        int run = 0;
        int bestRun = 0;
        for (float d : deltas) {
            boolean searah = lookLeft ? d <= -t.smoothDeltaMin : d >= t.smoothDeltaMin;
            if (searah) {
                run++;
                bestRun = Math.max(bestRun, run);
            } else {
                run = 0;
            }
        }
        r.smoothMotion = bestRun >= t.smoothRunLength;
        r.deltaVariance = absVariance(deltas, t.smoothDeltaMin);
        r.tooSmooth = r.smoothMotion && r.deltaVariance < t.tooSmoothMax;

        // Rangkuman
        r.startYaw = raw[0];
        r.endYaw = raw[n - 1];
        double t0 = samples.get(0).timestampMs;
        double t1 = samples.get(n - 1).timestampMs;
        r.durationMs = (t0 >= 0 && t1 >= 0) ? Math.max(0, t1 - t0) : 0;

        r.passed = r.frontalStart && r.deviation && r.returnFrontal
                && r.singleDirection && r.smoothMotion;
        if (r.passed) {
            r.failureReason = null;
        } else if (!r.singleDirection) {
            r.failureReason = "CHALLENGE_WRONG_DIRECTION";
        } else if (!r.returnFrontal) {
            r.failureReason = "CHALLENGE_NO_RETURN";
        } else {
            r.failureReason = "CHALLENGE_FAILED";
        }
        return r;
    }

    /** Median filter jendela 3 (edge di-replicate) — meredam spike 1 frame. */
    static float[] medianFilter3(float[] v) {
        int n = v.length;
        if (n < 3) {
            return v.clone();
        }
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            float a = v[Math.max(0, i - 1)];
            float b = v[i];
            float c = v[Math.min(n - 1, i + 1)];
            out[i] = median3(a, b, c);
        }
        return out;
    }

    private static float median3(float a, float b, float c) {
        float lo = Math.min(a, Math.min(b, c));
        float hi = Math.max(a, Math.max(b, c));
        return a + b + c - lo - hi;
    }

    private static float absVariance(float[] v, float minAbs) {
        // Varians MAGNITUDO delta yang BUKAN diam/rest (|delta| >= minAbs) —
        // fase berbalik/diam dikecualikan. Gerakan video-loop sempurna
        // berkecepatan konstan menghasilkan langkah seragam (varians ≈ 0),
        // sedangkan gerakan manusia mengakselerasi/melambat (varians > 0).
        int count = 0;
        float sum = 0f;
        for (float x : v) {
            if (Math.abs(x) >= minAbs) {
                sum += Math.abs(x);
                count++;
            }
        }
        if (count < 2) {
            return 0f;
        }
        float mean = sum / count;
        float sq = 0f;
        for (float x : v) {
            if (Math.abs(x) >= minAbs) {
                float d = Math.abs(x) - mean;
                sq += d * d;
            }
        }
        return sq / count;
    }
}
