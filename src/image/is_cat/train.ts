import * as tf from "@tensorflow/tfjs-node-gpu";
import * as path from "path";
import { ImageDataset } from "./data";

export const MODEL_SAVE_PATH = path.resolve("models", "is_cat");
export const EPOCHS = 20;
export const BATCH_SIZE = 32;

export async function trainModel(
  model: tf.Sequential,
  trainData: ImageDataset,
  validationData: ImageDataset
): Promise<tf.History> {
  console.log(`Training on ${trainData.xs.shape[0]} images...`);
  console.log(`  Validation: ${validationData.xs.shape[0]} images`);
  console.log(`  Epochs: ${EPOCHS}, Batch size: ${BATCH_SIZE}`);

  const history = await model.fit(trainData.xs, trainData.ys, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [validationData.xs, validationData.ys],
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `  Epoch ${epoch + 1}/${EPOCHS} — ` +
            `loss: ${logs?.loss?.toFixed(4)} — ` +
            `acc: ${logs?.acc?.toFixed(4)} — ` +
            `val_loss: ${logs?.val_loss?.toFixed(4)} — ` +
            `val_acc: ${logs?.val_acc?.toFixed(4)}`
        );
      },
    },
  });

  const saveUrl = `file://${MODEL_SAVE_PATH}`;
  await model.save(saveUrl);
  console.log(`Model saved to ${MODEL_SAVE_PATH}`);

  return history;
}
