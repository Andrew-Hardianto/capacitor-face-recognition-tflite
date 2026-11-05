package com.andrew.face;

import android.content.Context;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.KeyStore;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/**
 * Jejak audit lokal terenkripsi (Fase 4).
 *
 * Setiap sesi liveness native menyimpan hasil + frame yang dipakai dalam bentuk
 * AES-256-GCM dengan kunci di AndroidKeyStore (tidak pernah keluar dari perangkat,
 * tidak ikut backup). File berada di penyimpanan privat aplikasi.
 *
 * Format file: [1 byte panjang IV][IV][ciphertext + GCM tag].
 * Dipakai untuk pemeriksaan fraud HRIS — kebijakan retensi mengikuti persetujuan
 * dan UU PDP (hak hapus: cukup hapus folder liveness_audit).
 */
public final class AuditStore {

    private static final String KEY_ALIAS = "andrew.face.audit.v1";
    private static final String AUDIT_DIR = "liveness_audit";
    private static final String KEYSTORE = "AndroidKeyStore";
    private static final String TRANSFORMATION = "AES/GCM/NoPadding";
    private static final int GCM_TAG_BITS = 128;
    private static final int IV_BYTES = 12;

    private AuditStore() {}

    /** Simpan JSON sesi + frame yang dipakai, semuanya terenkripsi. Mengembalikan sessionId. */
    public static String storeSession(Context context, String sessionJson, List<byte[]> frames)
            throws Exception {
        String sessionId = "session_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                .format(new Date());
        File dir = new File(context.getFilesDir(), AUDIT_DIR);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("Gagal membuat direktori audit");
        }

        encryptToFile(sessionJson.getBytes(StandardCharsets.UTF_8),
                new File(dir, sessionId + ".json.enc"));
        for (int i = 0; i < frames.size(); i++) {
            encryptToFile(frames.get(i), new File(dir, sessionId + "_frame_" + i + ".jpg.enc"));
        }
        return sessionId;
    }

    /** Dekripsi file audit — untuk alat pemeriksa (dijalankan di perangkat yang sama). */
    public static byte[] readEncryptedFile(File file) throws Exception {
        byte[] blob;
        try (FileInputStream in = new FileInputStream(file)) {
            blob = new byte[(int) file.length()];
            int offset = 0;
            while (offset < blob.length) {
                int read = in.read(blob, offset, blob.length - offset);
                if (read < 0) break;
                offset += read;
            }
        }
        if (blob.length < IV_BYTES + 1) {
            throw new IllegalStateException("File audit tidak valid");
        }
        int ivLen = blob[0] & 0xFF;
        byte[] iv = new byte[ivLen];
        System.arraycopy(blob, 1, iv, 0, ivLen);
        byte[] ciphertext = new byte[blob.length - 1 - ivLen];
        System.arraycopy(blob, 1 + ivLen, ciphertext, 0, ciphertext.length);

        Cipher cipher = Cipher.getInstance(TRANSFORMATION);
        cipher.init(Cipher.DECRYPT_MODE, getOrCreateKey(), new GCMParameterSpec(GCM_TAG_BITS, iv));
        return cipher.doFinal(ciphertext);
    }

    private static void encryptToFile(byte[] plaintext, File outFile) throws Exception {
        Cipher cipher = Cipher.getInstance(TRANSFORMATION);
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey());
        byte[] iv = cipher.getIV();
        byte[] ciphertext = cipher.doFinal(plaintext);

        try (FileOutputStream out = new FileOutputStream(outFile)) {
            out.write(iv.length);
            out.write(iv);
            out.write(ciphertext);
        }
    }

    private static SecretKey getOrCreateKey() throws Exception {
        KeyStore keyStore = KeyStore.getInstance(KEYSTORE);
        keyStore.load(null);
        if (keyStore.containsAlias(KEY_ALIAS)) {
            KeyStore.Entry entry = keyStore.getEntry(KEY_ALIAS, null);
            if (entry instanceof KeyStore.SecretKeyEntry) {
                return ((KeyStore.SecretKeyEntry) entry).getSecretKey();
            }
        }

        KeyGenerator keyGenerator =
                KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE);
        keyGenerator.init(new KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build());
        return keyGenerator.generateKey();
    }
}
