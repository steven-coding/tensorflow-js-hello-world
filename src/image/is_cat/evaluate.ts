import * as tf from "@tensorflow/tfjs-node-gpu";
import { ImageDataset } from "./data";
import { THRESHOLD } from "./predict";

export interface EvaluationResult {
  loss: number;
  accuracy: number;
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
}

export function evaluateModel(
  model: tf.LayersModel,
  testData: ImageDataset
): EvaluationResult {
  const [lossTensor, accTensor] = model.evaluate(testData.xs, testData.ys) as tf.Scalar[];
  const loss = lossTensor.dataSync()[0];
  const accuracy = accTensor.dataSync()[0];
  lossTensor.dispose();
  accTensor.dispose();

  // Confusion matrix
  const predictions = model.predict(testData.xs) as tf.Tensor;
  const predValues = predictions.dataSync();
  const labelValues = testData.ys.dataSync();
  predictions.dispose();

  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < predValues.length; i++) {
    const predicted = predValues[i] >= THRESHOLD ? 1 : 0;
    const actual = labelValues[i];
    if (predicted === 1 && actual === 1) tp++;
    else if (predicted === 1 && actual === 0) fp++;
    else if (predicted === 0 && actual === 0) tn++;
    else fn++;
  }

  console.log(`\nTest Evaluation:`);
  console.log(`  Loss: ${loss.toFixed(4)}`);
  console.log(`  Accuracy: ${(accuracy * 100).toFixed(1)}%`);
  console.log(`  Confusion Matrix:`);
  console.log(`    TP: ${tp}  FP: ${fp}`);
  console.log(`    FN: ${fn}  TN: ${tn}`);

  return { loss, accuracy, truePositives: tp, falsePositives: fp, trueNegatives: tn, falseNegatives: fn };
}
