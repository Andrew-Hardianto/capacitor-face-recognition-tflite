import { WebPlugin } from '@capacitor/core';
import type { FaceRecognitionPlugin } from './definitions';
export class FaceRecognitionWeb extends WebPlugin implements FaceRecognitionPlugin {
  async extractFaceFeature(_options: { imageBase64: string }): Promise<{ embedding: number[] }> {
    console.warn('Face recognition saat ini hanya berjalan di Android & iOS native.');
    return { embedding: [] };
  }

  async compareFaces(options: { vector1: number[], vector2: number[] }): Promise<{ distance: number, similarityPercentage: number }> {
    if (options.vector1.length !== options.vector2.length) {
      throw new Error("Panjang vector tidak sama");
    }
    let sum = 0;
    for (let i = 0; i < options.vector1.length; i++) {
      sum += Math.pow(options.vector1[i] - options.vector2[i], 2);
    }
    const distance = Math.sqrt(sum);

    let similarityPercentage = 100 - (distance * 30);
    if (similarityPercentage < 0) similarityPercentage = 0;

    return { distance, similarityPercentage };
  }
}
