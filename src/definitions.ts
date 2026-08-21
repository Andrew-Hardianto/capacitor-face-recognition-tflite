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
   * Implementasi: Multi-Scale MiniFASNet — inference dua kali (scale 1.0x dan 2.7x),
   * hasilnya di-average. Threshold 0.75 untuk HRIS production.
   *
   * @param options.imageBase64 - Gambar dalam format Base64
   * @returns isLive - true jika wajah asli, false jika terdeteksi spoofing
   * @returns score  - Skor liveness antara 0.0 (spoof) hingga 1.0 (live)
   * @returns confidence - "HIGH" (>0.85 atau <0.15), "MEDIUM" (0.6–0.85 atau 0.15–0.4), "LOW" (0.4–0.6)
   */
  checkLiveness(options: { imageBase64: string }): Promise<{
    isLive: boolean;
    score: number;
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  }>;

  /**
   * Mendeteksi semua wajah dalam gambar tanpa melakukan recognition.
   * Berguna untuk validasi awal dan blink challenge di layer JS (HRIS production).
   *
   * iOS: menggunakan VNDetectFaceLandmarksRequest untuk eye landmark (EAR-based).
   * Android: menggunakan ML Kit dengan CLASSIFICATION_MODE_ALL.
   *
   * @param options.imageBase64 - Gambar dalam format Base64
   * @returns count - Jumlah wajah yang terdeteksi
   * @returns faces - Array data tiap wajah
   */
  detectFaces(options: { imageBase64: string }): Promise<{
    count: number;
    faces: {
      x: number;
      y: number;
      width: number;
      height: number;
      /** Rotasi kepala kiri-kanan dalam derajat (Android: ML Kit, iOS: selalu 0) */
      headEulerAngleY: number;
      /** Rotasi kepala miring dalam derajat */
      headEulerAngleZ: number;
      /** Probabilitas mata kiri terbuka (0.0 = tutup, 1.0 = buka). Null jika tidak dapat dihitung. */
      leftEyeOpenProbability: number | null;
      /** Probabilitas mata kanan terbuka (0.0 = tutup, 1.0 = buka). Null jika tidak dapat dihitung. */
      rightEyeOpenProbability: number | null;
    }[];
  }>;
}