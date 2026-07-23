import { registerPlugin } from '@capacitor/core';

import type { FaceRecognitionPlugin } from './definitions';

const FaceRecognition = registerPlugin<FaceRecognitionPlugin>('FaceRecognition', {
  web: () => import('./web').then((m) => new m.FaceRecognitionWeb()),
});

export * from './definitions';
export { FaceRecognition };
