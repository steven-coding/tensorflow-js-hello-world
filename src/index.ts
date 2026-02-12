import * as tf from "@tensorflow/tfjs-node-gpu";
import { linearRegression } from "./linear_regression";

async function main() {
  console.log(`TensorFlow.js backend: ${tf.getBackend()}`);

  await linearRegression();
}

main();
