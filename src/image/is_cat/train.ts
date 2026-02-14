import * as tf from "@tensorflow/tfjs-node-gpu";
import * as path from "path";
import { ImageDataset } from "./data";

export const MODEL_SAVE_PATH = path.resolve("models", "is_cat");
export const EPOCHS = 50; // Increased, but early stopping will stop earlier
export const BATCH_SIZE = 32;

export async function trainModel(
  model: tf.Sequential,
  trainData: ImageDataset,
  validationData: ImageDataset
): Promise<tf.History> {
  console.log(`Training on ${trainData.xs.shape[0]} images...`);
  console.log(`  Validation: ${validationData.xs.shape[0]} images`);
  console.log(`  Epochs: ${EPOCHS}, Batch size: ${BATCH_SIZE}`);
  console.log(`  Early stopping: patience 5 epochs`);

  // Manual early stopping with best weights restoration
  let bestEpoch = 0;
  let bestValLoss = Infinity;
  let bestWeights: tf.Tensor[] = [];
  let patience = 5;
  let epochsWithoutImprovement = 0;

  const history = await model.fit(trainData.xs, trainData.ys, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [validationData.xs, validationData.ys],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
        console.log(
          `  Epoch ${epoch + 1}/${EPOCHS} — ` +
            `loss: ${logs?.loss?.toFixed(4)} — ` +
            `acc: ${logs?.acc?.toFixed(4)} — ` +
            `val_loss: ${logs?.val_loss?.toFixed(4)} — ` +
            `val_acc: ${logs?.val_acc?.toFixed(4)}`
        );

        // Track best epoch and save weights
        if (logs?.val_loss !== undefined && logs.val_loss < bestValLoss) {
          bestValLoss = logs.val_loss;
          bestEpoch = epoch + 1;
          epochsWithoutImprovement = 0;

          // Dispose old best weights
          bestWeights.forEach((w) => w.dispose());

          // Save new best weights (clone them)
          bestWeights = model.getWeights().map((w) => tf.clone(w));

          console.log(`    → New best model!`);
        } else {
          epochsWithoutImprovement++;
          if (epochsWithoutImprovement >= patience) {
            console.log(`\n  Early stopping triggered (no improvement for ${patience} epochs)`);
            model.stopTraining = true;
          }
        }
      },
    },
  });

  // Restore best weights if we found any improvement
  if (bestWeights.length > 0) {
    console.log(`\nRestoring best model from epoch ${bestEpoch} (val_loss: ${bestValLoss.toFixed(4)})`);
    model.setWeights(bestWeights);
    bestWeights.forEach((w) => w.dispose());
  }

  const saveUrl = `file://${MODEL_SAVE_PATH}`;
  await model.save(saveUrl);
  console.log(`Model saved to ${MODEL_SAVE_PATH}`);

  return history;
}
