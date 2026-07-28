# capacitor-face-recognition-tflite

Capacitor plugin for ML Kit + TFLite face recognition

## Install

To use npm

```bash
npm install capacitor-face-recognition-tflite
````

To use yarn

```bash
yarn add capacitor-face-recognition-tflite
```

Sync native files

```bash
npx cap sync
```

## API

<docgen-index>

* [`extractFaceFeature(...)`](#extractfacefeature)
* [`compareFaces(...)`](#comparefaces)

</docgen-index>

<docgen-api>
<!--Update the source file JSDoc comments and rerun docgen to update the docs below-->

### extractFaceFeature(...)

```typescript
extractFaceFeature(options: { imageBase64: string; }) => Promise<{ embedding: number[]; }>
```

Mengirim base64 gambar ke Native, mengembalikan array of numbers (embeddings)

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ embedding: number[]; }&gt;</code>

--------------------


### compareFaces(...)

```typescript
compareFaces(options: { vector1: number[]; vector2: number[]; }) => Promise<{ isMatch: boolean; score: number; similarityPercentage: number; }>
```

Membandingkan dua embedding wajah menggunakan Cosine Similarity.
Mengembalikan isMatch (threshold 0.75), score cosine, dan persentase kemiripan.

| Param         | Type                                                   |
| ------------- | ------------------------------------------------------ |
| **`options`** | <code>{ vector1: number[]; vector2: number[]; }</code> |

**Returns:** <code>Promise&lt;{ isMatch: boolean; score: number; similarityPercentage: number; }&gt;</code>

--------------------

</docgen-api>
