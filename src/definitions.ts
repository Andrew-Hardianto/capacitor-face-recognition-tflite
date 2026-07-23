export interface FaceRecognitionPlugin {
  echo(options: { value: string }): Promise<{ value: string }>;
}
