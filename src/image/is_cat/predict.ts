import * as tf from "@tensorflow/tfjs-node-gpu";
import { loadAndPreprocessImage } from "./data";

export const THRESHOLD = 0.5;

export interface PredictionResult {
  isCat: boolean;
  confidence: number;
  label: string;
}

export function predictImage(
  model: tf.LayersModel,
  imagePath: string
): PredictionResult {
  const imageTensor = loadAndPreprocessImage(imagePath);
  const batched = imageTensor.expandDims(0) as tf.Tensor4D;
  const prediction = model.predict(batched) as tf.Tensor;
  const confidence = prediction.dataSync()[0];

  imageTensor.dispose();
  batched.dispose();
  prediction.dispose();

  const isCat = confidence >= THRESHOLD;
  return {
    isCat,
    confidence,
    label: isCat ? "cat" : "not cat",
  };
}
