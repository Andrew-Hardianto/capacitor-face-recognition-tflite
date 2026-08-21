import { WebPlugin } from '@capacitor/core';
import type { FaceRecognitionPlugin } from './definitions';

export class FaceRecognitionWeb extends WebPlugin implements FaceRecognitionPlugin {
  async extractFaceFeature(_options: { imageBase64: string }): Promise<{ embedding: number[] }> {
    console.warn('Face recognition saat ini hanya berjalan di Android & iOS native.');
    return { embedding: [] };
  }

  async compareFaces(options: { vector1: number[], vector2: number[] }): Promise<{
    isMatch: boolean;
    score: number;
    similarityPercentage: number;
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

    // THRESHOLD / AMBANG BATAS
    // 0.75 adalah standar yang baik.
    // Jika masih tembus wajah orang lain, naikkan ke 0.80 atau 0.85
    const isMatch = score > 0.75;
    const similarityPercentage = Math.max(0, score * 100);

    return { isMatch, score, similarityPercentage };
  }

  async checkLiveness(_options: { imageBase64: string }): Promise<{
    isLive: boolean;
    score: number;
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  }> {
    console.warn('checkLiveness (Anti-Spoofing) hanya berjalan di Android & iOS native.');
    return { isLive: false, score: 0, confidence: 'LOW' };
  }

  async detectFaces(_options: { imageBase64: string }): Promise<{
    count: number;
    faces: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      headEulerAngleY: number;
      headEulerAngleZ: number;
      leftEyeOpenProbability: number | null;
      rightEyeOpenProbability: number | null;
    }>;
  }> {
    console.warn('detectFaces hanya berjalan di Android & iOS native.');
    return { count: 0, faces: [] };
  }
}
