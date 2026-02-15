import * as tf from "@tensorflow/tfjs-node-gpu";
import { linearRegression } from "./linear_regression/simple";
import { priceOfWine } from "./linear_regression/price-of-wine";
import { isCatClassifier, predict } from "./image/is_cat";
import { predictImage } from "./image/is_cat/predict";

async function main() {
  console.log(`TensorFlow.js backend: ${tf.getBackend()}`);

  //LINEAR REGRESSION - SIMPLE
  //await linearRegression();

  //LINEAR REGRESSION - PRICE OF WINE
  await priceOfWine();

  //CAT - CLASSIFIER
  //await isCatClassifier();
  // const result = await predict(
  //   "models/is_cat/model.json",
  //   "data/is_cat/test/cat/999.jpg"
  // );
  // console.log(result.label, result.confidence);
  
}

main();
