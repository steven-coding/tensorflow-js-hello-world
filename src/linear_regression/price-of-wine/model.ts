import * as tf from "@tensorflow/tfjs-node-gpu";

export const EPOCHS = 200;
export const BATCH_SIZE = 8;

export function createWineModel(): tf.Sequential {
  const model = tf.sequential();

  // Hidden layer with 16 units
  model.add(
    tf.layers.dense({
      units: 16,
      activation: "relu",
      inputShape: [4], // 4 features: AGST, HarvestRain, WinterRain, Age
    })
  );

  // Output layer: single linear output for price prediction
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "linear",
    })
  );

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError",
    metrics: ["mse"],
  });

  return model;
}
