export interface FaceRecognitionPlugin {
  /**
   * Sends base64 image to Native, returning an array of numbers (embeddings)
   */
  extractFaceFeature(options: { imageBase64: string }): Promise<{ embedding: number[] }>;

  /**
   * Compares two face embeddings using Cosine Similarity.
   * Returns isMatch (0.75 threshold), cosine score, and similarity percentage.
   */
  compareFaces(options: { vector1: number[], vector2: number[] }): Promise<{
    isMatch: boolean;
    score: number;
    similarityPercentage: number;
  }>;

  /**
   * Checks if the face in the image is a real (live) face or a spoof
   * (printed photo, phone screen, video, mask).
   *
   * Implementation: Multi-Scale MiniFASNet — inference twice (1.0x and 2.7x scales),
   * results are averaged. Threshold 0.75.
   *
   * @param options.imageBase64 - Image in Base64 format
   * @returns isLive - true if real face, false if spoofing detected
   * @returns score  - Liveness score between 0.0 (spoof) and 1.0 (live)
   * @returns confidence - "HIGH" (>0.85 or <0.15), "MEDIUM" (0.6–0.85 or 0.15–0.4), "LOW" (0.4–0.6)
   */
  checkLiveness(options: { imageBase64: string }): Promise<{
    isLive: boolean;
    score: number;
    confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  }>;

  /**
   * Detects all faces in the image without performing recognition.
   * Useful for early validation and blink challenge in JS layer.
   *
   * iOS: uses VNDetectFaceLandmarksRequest for eye landmarks (EAR-based).
   * Android: uses ML Kit with CLASSIFICATION_MODE_ALL.
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
      /** Left-right head rotation in degrees (Android: ML Kit, iOS: always 0) */
      headEulerAngleY: number;
      /** Tilt head rotation in degrees */
      headEulerAngleZ: number;
      /** Left eye open probability (0.0 = closed, 1.0 = open). Null if uncalculable. */
      leftEyeOpenProbability: number | null;
      /** Right eye open probability (0.0 = closed, 1.0 = open). Null if uncalculable. */
      rightEyeOpenProbability: number | null;
    }[];
  }>;
}