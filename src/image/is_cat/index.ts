import * as tf from "@tensorflow/tfjs-node-gpu";
import * as fs from "fs";
import * as path from "path";
import { createCatModel } from "./model";
import { loadSplit, DATA_DIR } from "./data";
import { trainModel, MODEL_SAVE_PATH } from "./train";
import { evaluateModel } from "./evaluate";
import { predictImage } from "./predict";

export async function isCatClassifier(): Promise<void> {
  console.log("\n=== Cat/Not-Cat Image Classifier ===\n");

  const modelJsonPath = path.join(MODEL_SAVE_PATH, "model.json");
  let model: tf.LayersModel;

  if (fs.existsSync(modelJsonPath)) {
    console.log("Loading saved model...");
    model = await tf.loadLayersModel(`file://${modelJsonPath}`);

    // Re-compile the model after loading
    model.compile({
      optimizer: "adam",
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    console.log("Model loaded and compiled.");
  } else {
    const trainDir = path.join(DATA_DIR, "train");
    if (!fs.existsSync(trainDir)) {
      console.log(`Training data not found at ${DATA_DIR}`);
      console.log("Please create the following directory structure:");
      console.log(`  ${DATA_DIR}${path.sep}train${path.sep}cat${path.sep}`);
      console.log(`  ${DATA_DIR}${path.sep}train${path.sep}not_cat${path.sep}`);
      console.log(`  ${DATA_DIR}${path.sep}validation${path.sep}cat${path.sep}`);
      console.log(`  ${DATA_DIR}${path.sep}validation${path.sep}not_cat${path.sep}`);
      console.log(`  ${DATA_DIR}${path.sep}test${path.sep}cat${path.sep}`);
      console.log(`  ${DATA_DIR}${path.sep}test${path.sep}not_cat${path.sep}`);
      console.log("Then run again.");
      return;
    }

    const sequentialModel = createCatModel();
    sequentialModel.summary();

    const trainData = loadSplit("train");
    const validationData = loadSplit("validation");

    await trainModel(sequentialModel, trainData, validationData);

    trainData.xs.dispose();
    trainData.ys.dispose();
    validationData.xs.dispose();
    validationData.ys.dispose();

    model = sequentialModel;
  }

  // Evaluate on test data
  const testDir = path.join(DATA_DIR, "test", "cat");
  if (fs.existsSync(testDir)) {
    const testData = loadSplit("test");
    evaluateModel(model, testData);
    testData.xs.dispose();
    testData.ys.dispose();
  }

  // Demo prediction on first test cat image
  if (fs.existsSync(testDir)) {
    const files = fs.readdirSync(testDir).filter((f) => /\.(jpg|jpeg|png)$/i.test(f));
    if (files.length > 0) {
      const samplePath = path.join(testDir, files[0]);
      console.log(`\nDemo prediction on: ${samplePath}`);
      const result = predictImage(model, samplePath);
      console.log(`  Result: ${result.label}`);
      console.log(`  Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    }
  }

  model.dispose();
  console.log("\nDone. GPU memory cleaned up.");
}

export async function predict(
  modelPath: string,
  imagePath: string
): Promise<{ isCat: boolean; confidence: number; label: string }> {
  const absoluteModelPath = path.resolve(modelPath);
  const absoluteImagePath = path.resolve(imagePath);

  const model = await tf.loadLayersModel(`file://${absoluteModelPath}`);

  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  const result = predictImage(model, absoluteImagePath);

  model.dispose();

  return result;
}
