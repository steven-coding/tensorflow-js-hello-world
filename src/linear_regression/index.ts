import * as tf from "@tensorflow/tfjs-node-gpu";

export async function linearRegression() {
  // Define a simple sequential model: learn y = 2x - 1
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

  // Training data
  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // Train
  await model.fit(xs, ys, {
    epochs: 250,
    verbose: 0,
  });

  // Predict y for x = 10 (expected: ~19)
  const prediction = model.predict(tf.tensor2d([10], [1, 1])) as tf.Tensor;
  console.log(`Prediction for x=10: ${prediction.dataSync()[0]}`);

  prediction.dispose();
  xs.dispose();
  ys.dispose();
}
