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

</docgen-index>

<docgen-api>
<!--Update the source file JSDoc comments and rerun docgen to update the docs below-->

### extractFaceFeature(...)

```typescript
extractFaceFeature(options: { imageBase64: string; }) => Promise<{ embedding: number[]; }>
```

Mengirim path gambar ke Native, mengembalikan array of numbers (embeddings)

| Param         | Type                                  |
| ------------- | ------------------------------------- |
| **`options`** | <code>{ imageBase64: string; }</code> |

**Returns:** <code>Promise&lt;{ embedding: number[]; }&gt;</code>

--------------------

</docgen-api>
