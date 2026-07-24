export interface FaceRecognitionPlugin {
  /**
   * Mengirim path gambar ke Native, mengembalikan array of numbers (embeddings)
   */
  extractFaceFeature(options: { imageBase64: string }): Promise<{ embedding: number[] }>;
}