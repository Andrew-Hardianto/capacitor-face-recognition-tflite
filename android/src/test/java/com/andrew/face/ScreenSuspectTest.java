package com.andrew.face;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * ScreenSuspect (P2.2) — heuristik artefak layar (soft signal).
 * Fixtures sintetis deterministik.
 */
public class ScreenSuspectTest {

    private static int[] solid(int width, int height, int argb) {
        int[] px = new int[width * height];
        for (int i = 0; i < px.length; i++) {
            px[i] = argb;
        }
        return px;
    }

    @Test
    public void periodicityDetectsStripes() {
        // Stripes abu-abu period 4 (pola grid/moiré layar)
        int[] px = new int[32 * 32];
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                px[y * 32 + x] = ((x / 4) % 2 == 0) ? 0xFF202020 : 0xFFE0E0E0;
            }
        }
        ScreenSuspect.Result r = ScreenSuspect.analyze(px, 32, 32);
        assertTrue("stripes periodik harus terdeteksi (periodicity > 0.7)", r.periodicity > 0.7f);
    }

    @Test
    public void periodicityLowForNoise() {
        // Noise deterministik (LCG) — tidak boleh dianggap periodik
        int[] px = new int[32 * 32];
        long seed = 42;
        for (int i = 0; i < px.length; i++) {
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
            int red = (int) (seed % 256);
            int green = (int) ((seed >> 8) % 256);
            int blue = (int) ((seed >> 16) % 256);
            px[i] = 0xFF000000 | (red << 16) | (green << 8) | blue;
        }
        ScreenSuspect.Result r = ScreenSuspect.analyze(px, 32, 32);
        assertTrue("noise harus periodisitasnya rendah (< 0.5)", r.periodicity < 0.5f);
        assertTrue("noise tidak boleh dianggap panel datar", r.flatFraction < 0.3f);
    }

    @Test
    public void flatFractionDetectsUniformPanel() {
        ScreenSuspect.Result r = ScreenSuspect.analyze(solid(32, 32, 0xFF808080), 32, 32);
        assertEquals(1f, r.flatFraction, 0.001f);
        assertEquals(1f, r.saturationFlat, 0.001f);
        assertEquals(1f, r.score, 0.001f);
    }

    @Test
    public void saturationFlatLowForSaturatedColors() {
        int[] px = new int[32 * 32];
        for (int i = 0; i < px.length; i++) {
            px[i] = (i % 2 == 0) ? 0xFFFF0000 : 0xFF0000FF; // merah/biru murni
        }
        ScreenSuspect.Result r = ScreenSuspect.analyze(px, 32, 32);
        assertTrue("warna jenuh bukan sinyal layar-abu", r.saturationFlat < 0.05f);
    }

    @Test
    public void invalidInputIsZero() {
        ScreenSuspect.Result r = ScreenSuspect.analyze(null, 32, 32);
        assertEquals(0f, r.score, 0.001f);
        ScreenSuspect.Result empty = ScreenSuspect.analyze(new int[0], 0, 0);
        assertEquals(0f, empty.score, 0.001f);
    }
}
