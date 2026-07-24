import { WebPlugin } from '@capacitor/core';
import type { FaceRecognitionPlugin } from './definitions';
export class FaceRecognitionWeb extends WebPlugin implements FaceRecognitionPlugin {
  async extractFaceFeature(_options: { imageBase64: string }): Promise<{ embedding: number[] }> {
    console.warn('Face recognition saat ini hanya berjalan di Android & iOS native.');
    return { embedding: [] };
  }
}
