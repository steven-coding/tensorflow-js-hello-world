import * as tf from "@tensorflow/tfjs-node-gpu";
import { linearRegression } from "./linear_regression";
import { isCatClassifier } from "./image/is_cat";

async function main() {
  console.log(`TensorFlow.js backend: ${tf.getBackend()}`);

  //await linearRegression();
  await isCatClassifier();
}

main();
