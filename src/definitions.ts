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
  /** Konvensi tanda yaw untuk challenge (1 atau -1). Default 1 */
  yawSign?: number;
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

export interface VerifyFaceSequenceResult {
  passed: boolean;
  reason?: SequenceFailureReason;
  challenge: ChallengeType;
  medianLiveScore: number;
  liveVotes: number;
  frameCount: number;
  maxYaw: number;
  minYaw: number;
  frontalFrames: number;
  identityConsistent: boolean;
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
      /** Left-right head rotation in degrees (yaw) */
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
   * Issues a random active challenge for a liveness session.
   * Show the instruction to the user, capture frames, then call verifyFaceSequence.
   */
  getLivenessChallenge(): Promise<{ challenge: ChallengeType }>;

  /**
   * Multi-frame verification with active challenge (helper in plugin).
   *
   * Pass >= 5 frames captured over ~1.5 seconds. Passing rules:
   * - every frame must pass the quality gate
   * - median liveScore >= liveThreshold AND majority of frames isLive (voting,
   *   NOT "all frames must live")
   * - identity consistent across frames (cosine >= 0.8 vs first frame)
   * - challenge satisfied: yaw deviates >= 15 deg in the requested direction,
   *   then >= 2 frames return near frontal (|yaw| < 15 deg)
   */
  verifyFaceSequence(options: {
    images: string[];
    challenge: ChallengeType;
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
   * Requires camera permission (requested automatically).
   */
  captureLivenessSession(options: {
    challenge?: ChallengeType;
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
