export type LivenessLabel = 'LIVE' | 'PRINT' | 'REPLAY';

export type QualityReason =
  | 'NO_FACE'
  | 'MULTIPLE_FACES'
  | 'LOW_RESOLUTION'
  | 'FACE_TOO_SMALL'
  | 'FACE_AT_EDGE'
  | 'BAD_POSE'
  | 'BLURRY'
  | 'BAD_LIGHTING';

export type ChallengeType = 'LOOK_LEFT' | 'LOOK_RIGHT';

export type SequenceFailureReason =
  | 'QUALITY_FAILED'
  | 'IDENTITY_INCONSISTENT'
  | 'CHALLENGE_FAILED'
  | 'CHALLENGE_WRONG_DIRECTION'
  | 'CHALLENGE_NO_RETURN'
  | 'SPOOF_FRAME_DETECTED'
  | 'SCREEN_SUSPECTED'
  | 'TIMING_SUSPECT'
  | 'NOT_LIVE_ENOUGH';

/** Konfigurasi liveness & quality gate — semua opsional (default lihat README). */
export interface LivenessConfig {
  /** Threshold liveScore minimum. Default 0.80 */
  liveThreshold?: number;
  /** Veto: maksimum skor spoof yang ditoleransi. Default 0.20 */
  spoofVeto?: number;
  /** Sisi terpendek frame minimum (px). Default 480 */
  minFrameSide?: number;
  /** faceWidth / frameWidth minimum. Default 0.25 */
  minFaceRatio?: number;
  /** Margin tepi frame (0..1). Default 0.05 */
  edgeMargin?: number;
  /** Maksimum |yaw|/|pitch|/|roll| (derajat). Default 20 */
  maxPose?: number;
  /** Variance of Laplacian minimum (perlu kalibrasi). Default 50 */
  blurMin?: number;
  /** Mean luminance minimum. Default 60 */
  exposureMin?: number;
  /** Mean luminance maksimum. Default 200 */
  exposureMax?: number;
  /** Threshold cosine konsistensi identitas antar frame sebanding. Default 0.7 */
  identityThreshold?: number;
  /** Override tanda yaw per-perangkat (1 atau -1) untuk kasus buffer kamera
   *  ter-flip (mis. beberapa Samsung/OnePlus). Default 1. Konvensi arah normal
   *  sudah dinormalisasi plugin (yaw positif = menoleh ke kanan anatomis user) —
   *  jangan isi kecuali halaman "Kalibrasi Yaw" example-app menyuruhnya. */
  yawSignOverride?: number;
  /** Rasio minimum frame yang harus isLive per sesi (P2.1). Default 1.0 = SEMUA
   *  frame harus isLive. Turunkan (mis. 0.8) jika pengukuran BPCER menunjukkan
   *  user asli sering gagal — veto per-frame SPOOF_FRAME_DETECTED tetap aktif. */
  minLiveVoteRatio?: number;
  /** Veto screen-suspect per sesi (P2.2), 0..1. Default tidak di-set = soft
   *  signal saja (skor tetap dilaporkan di hasil & audit). Set eksplisit untuk
   *  mode ketat: sesi gagal (SCREEN_SUSPECTED) jika skor >= veto. */
  screenSuspectVeto?: number;
  /** Kembalikan data debug (probs per model). Default false */
  debug?: boolean;
}

export interface LivenessResult {
  label: LivenessLabel;
  /** Probabilitas live (0..1) */
  liveScore: number;
  /** Probabilitas print spoof (0..1) */
  printScore: number;
  /** Probabilitas replay spoof (0..1) */
  replayScore: number;
  /** Sinyal artefak layar per frame (P2.2) — soft signal eksperimental (0..1) */
  screenSuspectScore: number;
}

export interface QualityResult {
  passed: boolean;
  reason?: QualityReason;
  faceRatio: number;
  blur: number;
}

export interface MatchResult {
  isMatch: boolean;
  score: number;
  threshold: number;
  thresholdVersion: string;
}

export interface VerifyFaceResult {
  isLive: boolean;
  thresholdVersion: string;
  /** null jika quality gate gagal */
  liveness: LivenessResult | null;
  quality: QualityResult;
  /** HANYA diisi jika isLive — dari frame & bbox yang sama */
  embedding?: number[];
  /** Hanya jika referenceEmbedding diberikan dan isLive */
  match?: MatchResult;
  /** Hanya jika config.debug */
  debug?: { probs27: number[]; probs40: number[] };
}

/** Hasil per-aturan lintasan challenge (P1) — lihat ChallengeTrajectory. */
export interface TrajectoryRulesPassed {
  /** R1 — frame awal mendekati frontal (|yaw| < 8°) */
  frontalStart: boolean;
  /** R2 — ada deviasi ≥ 15° ke arah challenge, setelah fase awal */
  deviation: boolean;
  /** R3 — frame akhir kembali mendekati frontal (|yaw| < 12°) */
  returnFrontal: boolean;
  /** R4 — tanpa gerakan berkelanjutan ke arah SALAH (deviasi < 12°) — menolak head-shake */
  singleDirection: boolean;
  /** R5 — gerakan halus berkelanjutan (≥ 3 delta searah pada nilai ter-filter median-3) */
  smoothMotion: boolean;
}

/** Ringkasan lintasan challenge (P1) — selalu ada di hasil sesi. */
export interface TrajectorySummary {
  startYaw: number;
  endYaw: number;
  /** Deviasi ekstrem ke arah challenge */
  extremeYaw: number;
  extremeFrame: number;
  /** Deviasi teramati ke arah SALAH (diagnostik) */
  wrongWayMax: number;
  /** Varians magnitudo delta yaw — mendekati 0 menandai gerakan konstan sempurna (dugaan video-loop) */
  deltaVariance: number;
  /** Lintasan "terlalu sempurna" (deltaVariance ≈ 0) — soft signal dugaan video-loop (P2.3) */
  tooSmooth: boolean;
  /** Durasi sesi dalam ms (0 jika timestamp tidak tersedia) */
  durationMs: number;
  rulesPassed: TrajectoryRulesPassed;
}

export interface VerifyFaceSequenceResult {
  passed: boolean;
  reason?: SequenceFailureReason;
  challenge: ChallengeType;
  /** Present on verifyFaceSequence results (from createChallengeSession). */
  challengeId?: string;
  /** false = jalur JS base64 (dev-only), true = native capture (produksi) */
  native: boolean;
  medianLiveScore: number;
  liveVotes: number;
  frameCount: number;
  /** Max normalized yaw across quality-passed frames (positive = user's anatomical right) */
  maxYaw: number;
  /** Min normalized yaw across quality-passed frames (negative = user's anatomical left) */
  minYaw: number;
  frontalFrames: number;
  identityConsistent: boolean;
  /** Cosine minimum antara frame sebanding terhadap frame referensi (paling frontal).
   *  null jika kurang dari 2 frame sebanding. Untuk diagnostik kalibrasi. */
  minIdentityCosine: number | null;
  /** Ringkasan lintasan challenge (aturan R1–R5 + statistik yaw) */
  trajectory: TrajectorySummary;
  /** Sinyal artefak layar agregat (P2.2) — max antar frame; soft signal kecuali
   *  screenSuspectVeto di-set (mode ketat → SCREEN_SUSPECTED). */
  screenSuspect: number;
  /** Trace yaw per frame — HANYA jika config.debug (build debug) */
  yawTrace?: { index: number; yaw: number; timestampMs?: number }[];
  thresholdVersion: string;
  match?: MatchResult;
}

/**
 * Sinyal risiko perangkat (Fase 4) — heuristik sisi klien, BUKAN bukti.
 * Keputusan akhir tetap di aplikasi/server; jangan jadikan gerbang tunggal.
 */
export interface DeviceIntegrity {
  /** Berjalan di emulator / simulator */
  isEmulator: boolean;
  /** Perangkat rooted / jailbroken (heuristik) */
  isRooted: boolean;
  /** Build debuggable (development) */
  isDebuggable: boolean;
  /** Framework hook terdeteksi (Xposed / Frida / Substrate, heuristik) */
  hooksDetected: boolean;
}

/** Hasil captureLivenessSession — frame diambil native, tidak melewati JS. */
export interface NativeSessionResult extends VerifyFaceSequenceResult {
  /** Selalu true — penanda bahwa frame diambil native */
  native: true;
  deviceIntegrity: DeviceIntegrity;
  /** ID jejak audit terenkripsi lokal (frame + hasil tersimpan untuk audit HRIS) */
  auditId: string | null;
}

export interface FaceRecognitionPlugin {
  /**
   * Sends base64 image to Native, returning an array of numbers (embeddings).
   * Embedding size: 192 (MobileFaceNet).
   */
  extractFaceFeature(options: { imageBase64: string }): Promise<{ embedding: number[] }>;

  /**
   * Compares two face embeddings using Cosine Similarity.
   * Returns isMatch (0.75 default threshold, NOT yet calibrated), cosine score,
   * similarity percentage, and threshold metadata.
   */
  compareFaces(options: { vector1: number[], vector2: number[] }): Promise<{
    isMatch: boolean;
    score: number;
    similarityPercentage: number;
    threshold: number;
    thresholdVersion: string;
  }>;

  /**
   * Checks if the face in the image is a real (live) face or a spoof
   * (printed photo, phone screen, video).
   *
   * Implementation (liveness-v2, Silent-Face reference pipeline):
   * two models (crop 2.7x and 4.0x with shift, BGR raw input), summed softmax,
   * 3-class argmax ([PRINT, LIVE, REPLAY]) plus liveScore threshold and spoof veto.
   *
   * NOTE: the "score" field is now mapped to post-argmax liveScore — its values
   * are NOT comparable with the previous (liveness-v1) implementation.
   *
   * @returns isLive - true only if label === 'LIVE' && liveScore >= 0.80 && spoofScore < 0.20
   * @returns score  - alias of liveScore (0.0 = spoof, 1.0 = live)
   * @returns confidence - "HIGH" (>0.85 or <0.15), "MEDIUM" (0.6–0.85 or 0.15–0.4), "LOW" (0.4–0.6)
   */
  checkLiveness(options: { imageBase64: string }): Promise<{
    isLive: boolean;
    score: number;
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
    label: LivenessLabel;
    liveScore: number;
    printScore: number;
    replayScore: number;
    thresholdVersion: string;
  }>;

  /**
   * Detects all faces in the image without performing recognition.
   * Useful for early validation and active challenge in JS layer.
   *
   * iOS: uses VNDetectFaceLandmarksRequest for eye landmarks (EAR-based).
   * Android: uses ML Kit with CLASSIFICATION_MODE_ALL.
   * Angles are in DEGREES on both platforms (iOS converts from radians).
   * headEulerAngleY (yaw) is NORMALIZED: positive = user's anatomical right,
   * negative = user's anatomical left (same convention as the challenge).
   *
   * @param options.imageBase64 - Image in Base64 format
   * @returns count - Number of detected faces
   * @returns faces - Array of data for each face
   */
  detectFaces(options: { imageBase64: string }): Promise<{
    count: number;
    faces: {
      x: number;
      y: number;
      width: number;
      height: number;
      /** Up-down head rotation in degrees (pitch) */
      headEulerAngleX: number;
      /** Left-right head rotation in degrees (yaw, normalized: positive = user's anatomical right) */
      headEulerAngleY: number;
      /** Tilt head rotation in degrees (roll) */
      headEulerAngleZ: number;
      /** Left eye open probability (0.0 = closed, 1.0 = open). Null if uncalculable. */
      leftEyeOpenProbability: number | null;
      /** Right eye open probability (0.0 = closed, 1.0 = open). Null if uncalculable. */
      rightEyeOpenProbability: number | null;
    }[];
  }>;

  /**
   * Unified verification (TOCTOU-safe): quality gate + liveness + embedding,
   * all computed from the SAME frame and bounding box. The embedding is only
   * returned when isLive is true.
   */
  verifyFace(options: {
    imageBase64: string;
    referenceEmbedding?: number[];
    config?: LivenessConfig;
  }): Promise<VerifyFaceResult>;

  /**
   * Creates a native challenge session (random direction, single-use, 60s expiry).
   * Show the instruction to the user, capture frames, then call verifyFaceSequence
   * with the returned challengeId. The challenge direction CANNOT be chosen
   * from JS — it is always generated natively.
   */
  createChallengeSession(): Promise<{
    challengeId: string;
    challenge: ChallengeType;
    /** Expiry as Unix epoch milliseconds */
    expiresAt: number;
  }>;

  /**
   * Multi-frame verification with active challenge (helper in plugin).
   *
   * DEV/QA-ONLY-BY-DEFAULT: this path accepts base64 frames from JS (a
   * frame-injection surface) and is therefore only available on DEBUG builds,
   * OR when the host app explicitly opts in via the Capacitor config flag
   * `plugins.FaceRecognition.enableInsecureFramePath: true` (legacy webview
   * integrations that accept the risk). Production should prefer
   * captureLivenessSession. The result always carries `native: false`.
   *
   * Pass >= 5 frames captured over ~1.5 seconds. Passing rules:
   * - every frame must pass the quality gate
   * - anti-replay (P2): NO frame may be a hard spoof (label PRINT/REPLAY or
   *   spoofScore >= 0.5 → SPOOF_FRAME_DETECTED), the session duration must look
   *   human (TIMING_SUSPECT), and screenSuspectVeto, when configured, must not
   *   be reached (SCREEN_SUSPECTED)
   * - live voting: `minLiveVoteRatio` of frames must be isLive (default 1.0 =
   *   ALL frames) AND median liveScore >= liveThreshold
   * - identity consistent among pose-comparable frames (|yaw| <= 12 deg):
   *   cosine >= identityThreshold (default 0.7) vs the MOST FRONTAL frame.
   *   Frames with |yaw| > 12 deg serve the challenge only and are excluded —
   *   their embeddings are not pose-invariant.
   * - challenge satisfied — verified as a TEMPORAL TRAJECTORY (not min/max):
   *   R1 start near frontal (|yaw| < 8 deg) → R2 deviation >= 15 deg toward the
   *   requested direction (yaw is normalized: positive = user's anatomical
   *   right; LOOK_LEFT needs a NEGATIVE deviation) → R3 return near frontal
   *   (|yaw| < 12 deg), with R4 no sustained movement toward the wrong
   *   direction (< 12 deg — rejects head-shakes) and R5 sustained smooth
   *   motion (>= 3 same-direction deltas, median-3 filtered).
   *   Failure reasons: CHALLENGE_WRONG_DIRECTION (R4), CHALLENGE_NO_RETURN (R3),
   *   CHALLENGE_FAILED (R1/R2/R5).
   *
   * The challenge direction is read from the native challenge session
   * (single-use) — never from JS input.
   */
  verifyFaceSequence(options: {
    images: string[];
    /** From createChallengeSession — consumed on the FIRST attempt (single-use). */
    challengeId: string;
    referenceEmbedding?: number[];
    config?: LivenessConfig;
  }): Promise<VerifyFaceSequenceResult>;

  /**
   * Native liveness session (anti frame-injection, Fase 4).
   *
   * Opens the camera natively (CameraX / AVFoundation), captures >= 5 frames
   * over ~1.5s WITHOUT passing frames through JS, then runs the full
   * verification pipeline (quality gate + liveness + active challenge).
   * Only the RESULT is returned — never the raw frames.
   *
   * The session is rate-limited (max 10 attempts per 5 minutes) and logged to
   * an encrypted local audit store for HRIS fraud review.
   *
   * The challenge direction is ALWAYS generated natively at random — it cannot
   * be chosen by the caller.
   *
   * Requires camera permission (requested automatically).
   */
  captureLivenessSession(options: {
    referenceEmbedding?: number[];
    config?: LivenessConfig;
  }): Promise<NativeSessionResult>;

  /**
   * Device integrity signals (Fase 4) — root/emulator/hook heuristics.
   * These are RISK SIGNALS, not proof. Use them to trigger extra checks or
   * audit logging, not as a single gate.
   */
  getDeviceIntegrity(): Promise<{ deviceIntegrity: DeviceIntegrity }>;
}
