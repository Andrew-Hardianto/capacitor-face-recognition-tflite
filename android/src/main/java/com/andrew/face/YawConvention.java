package com.andrew.face;

/**
 * Konvensi yaw SDK (liveness-v3) — SATU kebenaran untuk iOS & Android.
 *
 * <p>Yaw TERNORMALISASI: <b>positif = user menoleh ke KANAN anatomisnya</b>,
 * negatif = ke KIRI anatomisnya. Semua aturan challenge bekerja pada konvensi
 * ini sehingga tidak lagi bergantung pada tanda mentah platform.
 *
 * <p>{@link #PLATFORM_YAW_SIGN} mengoreksi yaw MENTAH ML Kit ke konvensi anatomis:
 * {@code getHeadEulerAngleY} mentah positif = wajah berbalik ke sisi KANAN gambar
 * (frame tak termirror) = user menoleh ke KIRI anatomisnya (dokumen resmi ML Kit:
 * "Positive y euler angle is when the face is turned towards the right side of
 * the image") → dikalikan -1.
 *
 * <p>VERIFIKASI WAJIB via halaman "Kalibrasi Yaw" di example-app (P0-a):
 * "Dagu ke bahu KIRI Anda" → yaw ternormalisasi harus <b>negatif</b>.
 * Jika positif pada perangkat tertentu (buffer kamera ter-flip, dilaporkan pada
 * beberapa Samsung/OnePlus), set {@code config.yawSignOverride = -1} — JANGAN
 * mengubah konstanta ini.
 */
public final class YawConvention {

    /** Pengali koreksi yaw mentah ML Kit → konvensi anatomis SDK. */
    public static final float PLATFORM_YAW_SIGN = -1f;

    private YawConvention() {
    }

    /** Normalisasi yaw mentah platform → konvensi anatomis SDK (positif = menoleh ke kanan anatomis user). */
    public static float normalize(float rawYaw) {
        return rawYaw * PLATFORM_YAW_SIGN;
    }

    /**
     * Override per-perangkat ({@code config.yawSignOverride}) untuk kasus buffer
     * ter-flip. Diterapkan ke yaw ternormalisasi SEBELUM agregasi challenge.
     */
    public static float applyOverride(float normalizedYaw, int overrideSign) {
        return overrideSign >= 0 ? normalizedYaw : -normalizedYaw;
    }
}
