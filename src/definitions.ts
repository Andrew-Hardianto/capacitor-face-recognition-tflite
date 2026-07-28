export interface FaceRecognitionPlugin {
  /**
   * Mengirim base64 gambar ke Native, mengembalikan array of numbers (embeddings)
   */
  extractFaceFeature(options: { imageBase64: string }): Promise<{ embedding: number[] }>;

  /**
   * Membandingkan dua embedding wajah menggunakan Cosine Similarity.
   * Mengembalikan isMatch (threshold 0.75), score cosine, dan persentase kemiripan.
   */
  compareFaces(options: { vector1: number[], vector2: number[] }): Promise<{
    isMatch: boolean;
    score: number;
    similarityPercentage: number;
  }>;
}