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

  /**
   * Memeriksa apakah wajah dalam gambar adalah wajah asli (live) atau spoofing
   * (foto cetak, layar HP, video, topeng).
   *
   * Memerlukan model `anti_spoof.tflite` di folder assets (Android) atau bundle (iOS).
   *
   * @param options.imageBase64 - Gambar dalam format Base64
   * @returns isLive - true jika wajah asli, false jika terdeteksi spoofing
   * @returns score - Skor liveness antara 0.0 (spoof) hingga 1.0 (live)
   * @returns confidence - Tingkat kepercayaan: "HIGH" (>0.85), "MEDIUM" (0.6–0.85), "LOW" (<0.6)
   */
  checkLiveness(options: { imageBase64: string }): Promise<{
    isLive: boolean;
    score: number;
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  }>;
}