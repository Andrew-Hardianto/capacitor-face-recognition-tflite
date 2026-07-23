import { WebPlugin } from '@capacitor/core';

import type { FaceRecognitionPlugin } from './definitions';

export class FaceRecognitionWeb extends WebPlugin implements FaceRecognitionPlugin {
  async echo(options: { value: string }): Promise<{ value: string }> {
    console.log('ECHO', options);
    return options;
  }
}
