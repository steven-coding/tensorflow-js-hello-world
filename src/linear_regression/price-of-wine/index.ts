import * as tf from "@tensorflow/tfjs-node-gpu";
import { loadWineData } from "./data";
import { createWineModel, EPOCHS, BATCH_SIZE } from "./model";

export async function priceOfWine() {
  console.log("\n=== Bordeaux Wine Price Prediction (Linear Regression) ===\n");

  // Load and split data
  const { trainXs, trainYs, testXs, testYs, normParams } = loadWineData();
  console.log(`Training samples: ${trainXs.shape[0]}`);
  console.log(`Test samples: ${testXs.shape[0]}`);
  console.log(`Features: AGST, HarvestRain, WinterRain, Age\n`);

  // Create model
  const model = createWineModel();
  model.summary();

  // Train model
  console.log(`\nTraining for ${EPOCHS} epochs (batch size: ${BATCH_SIZE})...\n`);
  await model.fit(trainXs, trainYs, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    verbose: 1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 50 === 0) {
          console.log(
            `Epoch ${epoch + 1}/${EPOCHS} - ` +
              `loss: ${logs?.loss.toFixed(4)} - ` +
              `val_loss: ${logs?.val_loss?.toFixed(4)}`
          );
        }
      },
    },
  });

  // Evaluate on test set
  console.log("\n=== Evaluation on Test Set ===");
  const evalResult = model.evaluate(testXs, testYs) as tf.Scalar[];
  const testMSE = await evalResult[0].data();
  console.log(`Test MSE: ${testMSE[0].toFixed(4)}`);
  console.log(`Test RMSE: ${Math.sqrt(testMSE[0]).toFixed(4)}`);

  // Calculate R² on test set
  const predictions = model.predict(testXs) as tf.Tensor2D;
  const testYsMean = testYs.mean();
  const totalSS = testYs.sub(testYsMean).square().sum();
  const residualSS = testYs.sub(predictions).square().sum();
  const r2 = tf.scalar(1).sub(residualSS.div(totalSS));
  const r2Value = await r2.data();
  console.log(`Test R²: ${r2Value[0].toFixed(4)}`);

  // Demo prediction: Predict for first test sample (1976)
  console.log("\n=== Demo Prediction (1976 vintage) ===");
  const samplePrediction = predictions.slice([0, 0], [1, 1]);
  const actualPrice = await testYs.slice([0, 0], [1, 1]).data();
  const predictedPrice = await samplePrediction.data();
  console.log(`Actual Price (log): ${actualPrice[0].toFixed(4)}`);
  console.log(`Predicted Price (log): ${predictedPrice[0].toFixed(4)}`);
  console.log(`Difference: ${Math.abs(actualPrice[0] - predictedPrice[0]).toFixed(4)}`);

  // Cleanup
  evalResult.forEach((t) => t.dispose());
  predictions.dispose();
  testYsMean.dispose();
  totalSS.dispose();
  residualSS.dispose();
  r2.dispose();
  trainXs.dispose();
  trainYs.dispose();
  testXs.dispose();
  testYs.dispose();

  console.log("\n=== Done ===\n");
}
