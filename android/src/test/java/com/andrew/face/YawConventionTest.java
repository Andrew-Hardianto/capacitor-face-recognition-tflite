package com.andrew.face;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * YawConvention (liveness-v3) — menjaga kontrak arah challenge.
 */
public class YawConventionTest {

    @Test
    public void normalizationFollowsPlatformSign() {
        // Normalisasi = yaw mentah × PLATFORM_YAW_SIGN (sifat hukum, bebas nilai konstanta).
        assertEquals(45f * YawConvention.PLATFORM_YAW_SIGN, YawConvention.normalize(45f), 0.0001f);
        assertEquals(-30f * YawConvention.PLATFORM_YAW_SIGN, YawConvention.normalize(-30f), 0.0001f);
        assertEquals(0f, YawConvention.normalize(0f), 0.0001f);
    }

    @Test
    public void platformSignForMlKitIsNegative() {
        // Asumsi hasil kalibrasi P0-a untuk ML Kit (dokumen resmi: "Positive y
        // euler angle is when the face is turned towards the right side of the
        // image" = user menoleh ke KIRI anatomis) → koreksi ×(-1).
        // JIKA halaman "Kalibrasi Yaw" example-app menunjukkan sebaliknya pada
        // perangkat uji, perbarui konstanta YawConvention DAN test ini bersamaan.
        assertEquals(-1f, YawConvention.PLATFORM_YAW_SIGN, 0.0001f);
    }

    @Test
    public void overrideFlipsOnlyWhenNegative() {
        assertEquals(15f, YawConvention.applyOverride(15f, 1), 0.0001f);
        assertEquals(15f, YawConvention.applyOverride(15f, 0), 0.0001f);
        assertEquals(-15f, YawConvention.applyOverride(15f, -1), 0.0001f);
        assertEquals(15f, YawConvention.applyOverride(-15f, -1), 0.0001f);
    }

    @Test
    public void lookLeftChallengeWantsNegativeYaw() {
        // LOOK_LEFT = menoleh ke kiri anatomis user = yaw ternormalisasi NEGATIF.
        // Simulasi: user menoleh ke kiri anatomis → yaw mentah +20 (kanan gambar).
        float leftTurnNormalized = YawConvention.normalize(20f);
        // Dengan PLATFORM_YAW_SIGN = -1, hasilnya -20 → memenuhi aturan LOOK_LEFT
        // (minYaw <= -15) dan TIDAK memenuhi LOOK_RIGHT (maxYaw >= +15).
        float challengeYawMin = 15f;
        assertTrue(leftTurnNormalized <= -challengeYawMin);
        assertTrue(leftTurnNormalized < challengeYawMin);
    }
}
