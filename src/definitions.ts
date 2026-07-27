export interface FaceRecognitionPlugin {
  /**
   * Mengirim base64 gambar ke Native, mengembalikan array of numbers (embeddings)
   */
  extractFaceFeature(options: { imageBase64: string }): Promise<{ embedding: number[] }>;

  /**
   * Mengirim vector gambar ke Native, mengembalikan ismatch boolean
   */
  compareFaces(options: { vector1: number[], vector2: number[] }): Promise<{ distance: number, similarityPercentage: number }>;
}