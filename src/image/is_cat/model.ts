import * as tf from "@tensorflow/tfjs-node-gpu";

export const IMAGE_SIZE = 128;
export const CHANNELS = 3;

export function createCatModel(): tf.Sequential {
  const model = tf.sequential();

  // Conv Block 1: 128x128x3 -> 63x63x32
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_SIZE, IMAGE_SIZE, CHANNELS],
      filters: 32,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Conv Block 2: 63x63x32 -> 30x30x64
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Classification Head
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
