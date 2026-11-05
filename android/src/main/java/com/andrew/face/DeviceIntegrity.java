package com.andrew.face;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.os.Build;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

/**
 * Pemeriksaan integritas perangkat (Fase 4) — heuristik sisi klien.
 *
 * CATATAN PENTING: semua pemeriksaan di sini adalah SINYAL RISIKO, bukan bukti.
 * Deteksi root/emulator/hook yang benar-benar kuat membutuhkan device attestation
 * terverifikasi server (Play Integrity) — di luar scope plugin ini karena backend
 * hanya menyediakan penyimpanan vector (lihat plan Fase 3.4).
 *
 * Hasil dipakai oleh aplikasi untuk mengambil keputusan (tolak, minta tantangan
 * tambahan, atau catat untuk audit), bukan diandalkan sebagai gerbang tunggal.
 */
public final class DeviceIntegrity {

    private DeviceIntegrity() {}

    private static final String[] SU_PATHS = {
            "/system/bin/su",
            "/system/xbin/su",
            "/sbin/su",
            "/system/app/Superuser.apk",
            "/data/local/xbin/su",
            "/data/local/bin/su",
            "/data/local/su",
            "/su/bin/su",
            "/system/bin/failsafe/su",
    };

    private static final String[] MAGISK_PATHS = {
            "/sbin/.magisk",
            "/data/adb/magisk",
            "/cache/magisk.log",
    };

    /** Build/prop yang khas pada emulator. */
    public static boolean isEmulator() {
        return Build.FINGERPRINT.startsWith("generic")
                || Build.FINGERPRINT.startsWith("unknown")
                || Build.FINGERPRINT.contains("emulator")
                || Build.FINGERPRINT.contains("sdk_gphone")
                || Build.MODEL.contains("Emulator")
                || Build.MODEL.contains("Android SDK built for x86")
                || Build.MODEL.contains("google_sdk")
                || Build.MANUFACTURER.contains("Genymotion")
                || Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic")
                || Build.HARDWARE.contains("goldfish")
                || Build.HARDWARE.contains("ranchu")
                || Build.PRODUCT.contains("sdk_gphone")
                || Build.PRODUCT.contains("emulator")
                || Build.PRODUCT.contains("simulator");
    }

    /** Heuristik root: binary su, Magisk, atau build test-keys. */
    public static boolean isRooted() {
        for (String path : SU_PATHS) {
            if (new File(path).exists()) return true;
        }
        for (String path : MAGISK_PATHS) {
            if (new File(path).exists()) return true;
        }
        String tags = Build.TAGS;
        if (tags != null && tags.contains("test-keys")) return true;

        // which su
        Process process = null;
        try {
            process = Runtime.getRuntime().exec(new String[]{"/system/xbin/which", "su"});
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                if (reader.readLine() != null) return true;
            }
        } catch (Exception ignored) {
            // tidak ada which/su — anggap tidak rooted dari jalur ini
        } finally {
            if (process != null) process.destroy();
        }
        return false;
    }

    /** Build debuggable (development) — hasil liveness dari build debug tidak untuk produksi. */
    public static boolean isDebuggable(Context context) {
        return (context.getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
    }

    /** Heuristik framework hook (Xposed / Frida / Substrate) via classloader dan /proc/self/maps. */
    public static boolean hooksDetected() {
        String[] hookClasses = {
                "de.robv.android.xposed.XposedBridge",
                "de.robv.android.xposed.XposedHelpers",
                "com.saurik.substrate.MS$2",
                "com.topjohnwu.magisk.core",
        };
        for (String className : hookClasses) {
            try {
                Class.forName(className);
                return true;
            } catch (ClassNotFoundException ignored) {
                // normal — kelas tidak ada
            }
        }

        // Cari pustaka mencurigakan di /proc/self/maps (Frida dll)
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new java.io.FileInputStream("/proc/self/maps")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String lower = line.toLowerCase();
                if (lower.contains("frida") || lower.contains("xposed")
                        || lower.contains("substrate") || lower.contains("libgadget")) {
                    return true;
                }
            }
        } catch (Exception ignored) {
            // tidak bisa membaca maps — lewati
        }
        return false;
    }

    /** Snapshot lengkap untuk disertakan di hasil verifikasi. */
    public static org.json.JSONObject snapshot(Context context) {
        org.json.JSONObject o = new org.json.JSONObject();
        try {
            o.put("isEmulator", isEmulator());
            o.put("isRooted", isRooted());
            o.put("isDebuggable", isDebuggable(context));
            o.put("hooksDetected", hooksDetected());
        } catch (Exception ignored) {
            // JSONObject.put tidak melempar untuk tipe ini
        }
        return o;
    }
}
