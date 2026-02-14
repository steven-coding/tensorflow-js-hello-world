import * as tf from "@tensorflow/tfjs-node-gpu";

export const IMAGE_SIZE = 128;
export const CHANNELS = 3;

export function createCatModel(): tf.Sequential {
  const model = tf.sequential();

  // L2 regularization for all layers
  const l2Reg = tf.regularizers.l2({ l2: 0.001 });

  // Conv Block 1: 128x128x3 -> 63x63x16 (reduced from 32)
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_SIZE, IMAGE_SIZE, CHANNELS],
      filters: 16,
      kernelSize: 3,
      activation: "relu",
      kernelRegularizer: l2Reg,
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.dropout({ rate: 0.25 })); // Dropout after conv

  // Conv Block 2: 63x63x16 -> 30x30x32 (reduced from 64)
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      kernelRegularizer: l2Reg,
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.dropout({ rate: 0.25 })); // Dropout after conv

  // Classification Head
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.5 })); // Increased dropout
  model.add(
    tf.layers.dense({
      units: 64, // Reduced from 128
      activation: "relu",
      kernelRegularizer: l2Reg,
    })
  );
  model.add(tf.layers.dropout({ rate: 0.3 })); // Additional dropout
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
