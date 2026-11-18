package com.andrew.face;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

/**
 * Activity kamera untuk captureLivenessSession (Fase 4 — native capture).
 *
 * Frame diambil DI DALAM plugin dan disimpan ke file cache privat,
 * TIDAK pernah melewati layer JS — menutup celah injeksi frame.
 * Pipeline verifikasi (quality gate + liveness + challenge) tetap dijalankan
 * oleh FaceRecognitionPlugin terhadap frame-file ini.
 *
 * Hasil dikembalikan lewat setResult():
 *   RESULT_OK     → EXTRA_FRAME_PATHS (ArrayList&lt;String&gt;), EXTRA_CHALLENGE
 *   RESULT_CANCELED → EXTRA_ERROR (String)
 */
public class LivenessCaptureActivity extends Activity {

    public static final String EXTRA_CHALLENGE = "challenge";
    public static final String EXTRA_FRAME_COUNT = "frameCount";
    public static final String EXTRA_INTERVAL_MS = "intervalMs";
    public static final String EXTRA_WARMUP_MS = "warmupMs";
    public static final String EXTRA_FRAME_PATHS = "framePaths";
    public static final String EXTRA_FRAME_TIMESTAMPS = "frameTimestamps";
    public static final String EXTRA_ERROR = "error";

    private static final String FRAME_DIR = "liveness_frames";

    private PreviewView previewView;
    private TextView instructionView;
    private TextView progressView;
    private ImageCapture imageCapture;
    private final Handler handler = new Handler(Looper.getMainLooper());

    private String challenge;
    private int frameCount;
    private long intervalMs;
    private long warmupMs;
    private int captured = 0;
    private final ArrayList<String> framePaths = new ArrayList<>();
    private final ArrayList<Double> frameTimestamps = new ArrayList<>();

    public static Intent createIntent(Context context, String challenge, int frameCount,
                                      long intervalMs, long warmupMs) {
        Intent intent = new Intent(context, LivenessCaptureActivity.class);
        intent.putExtra(EXTRA_CHALLENGE, challenge);
        intent.putExtra(EXTRA_FRAME_COUNT, frameCount);
        intent.putExtra(EXTRA_INTERVAL_MS, intervalMs);
        intent.putExtra(EXTRA_WARMUP_MS, warmupMs);
        return intent;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        challenge = getIntent().getStringExtra(EXTRA_CHALLENGE);
        frameCount = Math.max(2, getIntent().getIntExtra(EXTRA_FRAME_COUNT, 5));
        intervalMs = getIntent().getLongExtra(EXTRA_INTERVAL_MS, 300L);
        warmupMs = getIntent().getLongExtra(EXTRA_WARMUP_MS, 1200L);

        // FAIL-CLOSED: challenge tak dikenal tidak boleh diam-diam memakai
        // instruksi default — sesi dibatalkan sebelum kamera dibuka.
        if (!"LOOK_LEFT".equals(challenge) && !"LOOK_RIGHT".equals(challenge)) {
            failWith("Challenge tidak dikenal");
            return;
        }

        buildUi();
        startCamera();
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);

        previewView = new PreviewView(this);
        root.addView(previewView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        instructionView = new TextView(this);
        // FAIL-CLOSED: mapping eksplisit per challenge — nilai tak dikenal sudah
        // ditolak di onCreate, TIDAK ada default diam-diam ke "ke kanan".
        String instruction;
        switch (challenge) {
            case "LOOK_LEFT":
                instruction = "TOLEH KE KIRI ANDA (dagu ke bahu kiri) perlahan, lalu kembali lurus";
                break;
            case "LOOK_RIGHT":
                instruction = "TOLEH KE KANAN ANDA (dagu ke bahu kanan) perlahan, lalu kembali lurus";
                break;
            default:
                instruction = ""; // tidak tercapai — onCreate sudah fail-closed
        }
        instructionView.setText(instruction);
        instructionView.setTextColor(Color.WHITE);
        instructionView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        instructionView.setGravity(Gravity.CENTER);
        instructionView.setBackgroundColor(Color.argb(160, 0, 0, 0));
        instructionView.setPadding(24, 32, 24, 32);
        FrameLayout.LayoutParams instrParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        instrParams.gravity = Gravity.TOP;
        root.addView(instructionView, instrParams);

        progressView = new TextView(this);
        progressView.setTextColor(Color.WHITE);
        progressView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        progressView.setGravity(Gravity.CENTER);
        progressView.setBackgroundColor(Color.argb(160, 0, 0, 0));
        progressView.setPadding(24, 24, 24, 24);
        FrameLayout.LayoutParams progressParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        progressParams.gravity = Gravity.BOTTOM;
        root.addView(progressView, progressParams);

        setContentView(root);
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();
                bindCamera(provider);
            } catch (ExecutionException | InterruptedException e) {
                failWith("Gagal memulai kamera: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCamera(ProcessCameraProvider provider) {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build();

        CameraSelector selector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();

        provider.unbindAll();
        provider.bindToLifecycle((LifecycleOwner) this, selector, preview, imageCapture);

        // Beri waktu user memosisikan wajah, lalu capture frame berkala
        handler.postDelayed(this::captureNext, warmupMs);
    }

    private void captureNext() {
        if (imageCapture == null) {
            failWith("ImageCapture tidak tersedia");
            return;
        }

        File dir = new File(getCacheDir(), FRAME_DIR);
        if (!dir.exists() && !dir.mkdirs()) {
            failWith("Gagal membuat direktori frame");
            return;
        }
        File outFile = new File(dir, "frame_" + captured + ".jpg");
        ImageCapture.OutputFileOptions options =
                new ImageCapture.OutputFileOptions.Builder(outFile).build();

        Executor executor = ContextCompat.getMainExecutor(this);
        imageCapture.takePicture(options, executor, new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults results) {
                framePaths.add(outFile.getAbsolutePath());
                frameTimestamps.add((double) System.currentTimeMillis());
                captured++;
                progressView.setText("Mengambil frame " + captured + "/" + frameCount);

                if (captured >= frameCount) {
                    double[] timestamps = new double[frameTimestamps.size()];
                    for (int i = 0; i < timestamps.length; i++) {
                        timestamps[i] = frameTimestamps.get(i);
                    }
                    Intent data = new Intent();
                    data.putStringArrayListExtra(EXTRA_FRAME_PATHS, framePaths);
                    data.putExtra(EXTRA_FRAME_TIMESTAMPS, timestamps);
                    data.putExtra(EXTRA_CHALLENGE, challenge);
                    setResult(RESULT_OK, data);
                    finish();
                } else {
                    handler.postDelayed(LivenessCaptureActivity.this::captureNext, intervalMs);
                }
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                failWith("Gagal mengambil frame: " + exception.getMessage());
            }
        });
    }

    private void failWith(String message) {
        Intent data = new Intent();
        data.putExtra(EXTRA_ERROR, message);
        setResult(RESULT_CANCELED, data);
        finish();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        handler.removeCallbacksAndMessages(null);
    }
}
