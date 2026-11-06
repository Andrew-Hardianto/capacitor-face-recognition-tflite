import { WebPlugin } from '@capacitor/core';
import type {
  ChallengeType,
  FaceRecognitionPlugin,
  NativeSessionResult,
  VerifyFaceResult,
  VerifyFaceSequenceResult,
} from './definitions';

// Threshold — satu sumber kebenaran untuk implementasi web (match-v1)
const MATCH_THRESHOLD = 0.75;
const MATCH_THRESHOLD_VERSION = 'match-v1';

export class FaceRecognitionWeb extends WebPlugin implements FaceRecognitionPlugin {
  async extractFaceFeature(_options: { imageBase64: string }): Promise<{ embedding: number[] }> {
    console.warn('Face recognition saat ini hanya berjalan di Android & iOS native.');
    return { embedding: [] };
  }

  async compareFaces(options: { vector1: number[], vector2: number[] }): Promise<{
    isMatch: boolean;
    score: number;
    similarityPercentage: number;
    threshold: number;
    thresholdVersion: string;
  }> {
    if (options.vector1.length !== options.vector2.length) {
      throw new Error('Panjang vector tidak sama');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    // Cosine Similarity (konsisten dengan Android & iOS)
    for (let i = 0; i < options.vector1.length; i++) {
      dotProduct += options.vector1[i] * options.vector2[i];
      norm1 += options.vector1[i] * options.vector1[i];
      norm2 += options.vector2[i] * options.vector2[i];
    }

    if (norm1 === 0 || norm2 === 0) {
      throw new Error('Data vektor ada yang bernilai nol semua, cek apakah embedding berhasil');
    }

    const score = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

    // THRESHOLD / AMBANG BATAS — default 0.75, BELUM dikalibrasi (Fase 5)
    const isMatch = score > MATCH_THRESHOLD;
    const similarityPercentage = Math.max(0, score * 100);

    return {
      isMatch,
      score,
      similarityPercentage,
      threshold: MATCH_THRESHOLD,
      thresholdVersion: MATCH_THRESHOLD_VERSION,
    };
  }

  async checkLiveness(_options: { imageBase64: string }): Promise<{
    isLive: boolean;
    score: number;
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
    label: 'LIVE' | 'PRINT' | 'REPLAY';
    liveScore: number;
    printScore: number;
    replayScore: number;
    thresholdVersion: string;
  }> {
    console.warn('checkLiveness (Anti-Spoofing) hanya berjalan di Android & iOS native.');
    // FAIL-CLOSED: web tidak pernah mengembalikan isLive true
    return {
      isLive: false,
      score: 0,
      confidence: 'LOW',
      label: 'REPLAY',
      liveScore: 0,
      printScore: 0,
      replayScore: 1,
      thresholdVersion: 'liveness-v2',
    };
  }

  async detectFaces(_options: { imageBase64: string }): Promise<{
    count: number;
    faces: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      headEulerAngleX: number;
      headEulerAngleY: number;
      headEulerAngleZ: number;
      leftEyeOpenProbability: number | null;
      rightEyeOpenProbability: number | null;
    }>;
  }> {
    console.warn('detectFaces hanya berjalan di Android & iOS native.');
    return { count: 0, faces: [] };
  }

  async verifyFace(_options: {
    imageBase64: string;
    referenceEmbedding?: number[];
  }): Promise<VerifyFaceResult> {
    // FAIL-CLOSED: jangan pernah mengembalikan isLive true / embedding dari web
    throw new Error('verifyFace hanya berjalan di Android & iOS native.');
  }

  async createChallengeSession(): Promise<{
    challengeId: string;
    challenge: ChallengeType;
    expiresAt: number;
  }> {
    console.warn('createChallengeSession hanya berjalan penuh di Android & iOS native.');
    const challenge: ChallengeType = Math.random() < 0.5 ? 'LOOK_LEFT' : 'LOOK_RIGHT';
    return {
      challengeId: typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `web-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      challenge,
      expiresAt: Date.now() + 60_000,
    };
  }

  async verifyFaceSequence(_options: {
    images: string[];
    challengeId: string;
    referenceEmbedding?: number[];
  }): Promise<VerifyFaceSequenceResult> {
    // FAIL-CLOSED: jangan pernah mengembalikan passed true dari web
    throw new Error('verifyFaceSequence hanya berjalan di Android & iOS native.');
  }

  async captureLivenessSession(_options: {
    referenceEmbedding?: number[];
  }): Promise<NativeSessionResult> {
    // FAIL-CLOSED: native capture hanya tersedia di Android & iOS
    throw new Error('captureLivenessSession hanya berjalan di Android & iOS native.');
  }

  async getDeviceIntegrity(): Promise<{ deviceIntegrity: import('./definitions').DeviceIntegrity }> {
    throw new Error('getDeviceIntegrity hanya berjalan di Android & iOS native.');
  }
}
