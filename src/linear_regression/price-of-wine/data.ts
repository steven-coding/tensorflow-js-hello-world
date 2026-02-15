import * as tf from "@tensorflow/tfjs-node-gpu";

/**
 * Represents a single Bordeaux wine vintage data point from Ashenfelter's dataset.
 */
export interface WineDataPoint {
  /** Vintage year (e.g., 1952) */
  Year: number;
  /** Log of relative market price (relative to 1961 vintage) */
  Price: number;
  /** Winter rainfall in ml (October to March) */
  WinterRain: number;
  /** Average Growing Season Temperature in °C (April to September) */
  AGST: number;
  /** Harvest rainfall in ml (August and September) */
  HarvestRain: number;
  /** Age of wine in years (time stored in cask) */
  Age: number;
}

/**
 * Min-Max normalization parameters for each feature.
 * Calculated from training data only to prevent data leakage.
 */
export interface NormalizationParams {
  /** AGST (temperature) min/max for scaling to [0, 1] */
  AGST: { min: number; max: number };
  /** Harvest rainfall min/max for scaling to [0, 1] */
  HarvestRain: { min: number; max: number };
  /** Winter rainfall min/max for scaling to [0, 1] */
  WinterRain: { min: number; max: number };
  /** Wine age min/max for scaling to [0, 1] */
  Age: { min: number; max: number };
}

/**
 * Wine dataset split into training and test sets with normalization parameters.
 * Features are normalized to [0, 1] using min-max scaling.
 */
export interface WineDataset {
  /** Training features: [22, 4] tensor (AGST, HarvestRain, WinterRain, Age) */
  trainXs: tf.Tensor2D;
  /** Training labels: [22, 1] tensor (log price) */
  trainYs: tf.Tensor2D;
  /** Test features: [5, 4] tensor (normalized with training params) */
  testXs: tf.Tensor2D;
  /** Test labels: [5, 1] tensor (log price) */
  testYs: tf.Tensor2D;
  /** Normalization parameters from training data */
  normParams: NormalizationParams;
}

/**
 * Complete Bordeaux wine dataset (1952-1980).
 * Source: Ashenfelter's Bordeaux equation
 * https://raw.githubusercontent.com/egarpor/handy/master/datasets/wine.csv
 */
const WINE_DATA: WineDataPoint[] = [
  { Year: 1952, Price: 7.495, WinterRain: 600, AGST: 17.1167, HarvestRain: 160, Age: 31 },
  { Year: 1953, Price: 8.0393, WinterRain: 690, AGST: 16.7333, HarvestRain: 80, Age: 30 },
  { Year: 1955, Price: 7.6858, WinterRain: 502, AGST: 17.15, HarvestRain: 130, Age: 28 },
  { Year: 1957, Price: 6.9845, WinterRain: 420, AGST: 16.1333, HarvestRain: 110, Age: 26 },
  { Year: 1958, Price: 6.7772, WinterRain: 582, AGST: 16.4167, HarvestRain: 187, Age: 25 },
  { Year: 1959, Price: 8.0757, WinterRain: 485, AGST: 17.4833, HarvestRain: 187, Age: 24 },
  { Year: 1960, Price: 6.5188, WinterRain: 763, AGST: 16.4167, HarvestRain: 290, Age: 23 },
  { Year: 1961, Price: 8.4937, WinterRain: 830, AGST: 17.3333, HarvestRain: 38, Age: 22 },
  { Year: 1962, Price: 7.388, WinterRain: 697, AGST: 16.3, HarvestRain: 52, Age: 21 },
  { Year: 1963, Price: 6.7127, WinterRain: 608, AGST: 15.7167, HarvestRain: 155, Age: 20 },
  { Year: 1964, Price: 7.3094, WinterRain: 402, AGST: 17.2667, HarvestRain: 96, Age: 19 },
  { Year: 1965, Price: 6.2518, WinterRain: 602, AGST: 15.3667, HarvestRain: 267, Age: 18 },
  { Year: 1966, Price: 7.7443, WinterRain: 819, AGST: 16.5333, HarvestRain: 86, Age: 17 },
  { Year: 1967, Price: 6.8398, WinterRain: 714, AGST: 16.2333, HarvestRain: 118, Age: 16 },
  { Year: 1968, Price: 6.2435, WinterRain: 610, AGST: 16.2, HarvestRain: 292, Age: 15 },
  { Year: 1969, Price: 6.3459, WinterRain: 575, AGST: 16.55, HarvestRain: 244, Age: 14 },
  { Year: 1970, Price: 7.5883, WinterRain: 622, AGST: 16.6667, HarvestRain: 89, Age: 13 },
  { Year: 1971, Price: 7.1934, WinterRain: 551, AGST: 16.7667, HarvestRain: 112, Age: 12 },
  { Year: 1972, Price: 6.2049, WinterRain: 536, AGST: 14.9833, HarvestRain: 158, Age: 11 },
  { Year: 1973, Price: 6.6367, WinterRain: 376, AGST: 17.0667, HarvestRain: 123, Age: 10 },
  { Year: 1974, Price: 6.2941, WinterRain: 574, AGST: 16.3, HarvestRain: 184, Age: 9 },
  { Year: 1975, Price: 7.292, WinterRain: 572, AGST: 16.95, HarvestRain: 171, Age: 8 },
  { Year: 1976, Price: 7.1211, WinterRain: 418, AGST: 17.65, HarvestRain: 247, Age: 7 },
  { Year: 1977, Price: 6.2587, WinterRain: 821, AGST: 15.5833, HarvestRain: 87, Age: 6 },
  { Year: 1978, Price: 7.186, WinterRain: 763, AGST: 15.8167, HarvestRain: 51, Age: 5 },
  { Year: 1979, Price: 6.9541, WinterRain: 717, AGST: 16.1667, HarvestRain: 122, Age: 4 },
  { Year: 1980, Price: 6.4979, WinterRain: 578, AGST: 16.0, HarvestRain: 74, Age: 3 },
];

/**
 * Loads and prepares the wine dataset for training and testing.
 *
 * Splits data into:
 * - Training: First 22 vintages (1952-1975)
 * - Test: Last 5 vintages (1976-1980)
 *
 * All features are normalized to [0, 1] using min-max scaling.
 * Normalization parameters are calculated from training data only.
 *
 * @returns Dataset with normalized training/test tensors and normalization params
 */
export function loadWineData(): WineDataset {
  // Split: First 22 vintages (1952-1975) for training, last 5 (1976-1980) for testing
  const trainData = WINE_DATA.slice(0, 22);
  const testData = WINE_DATA.slice(22);

  // Calculate normalization parameters from training data only
  const trainAGST = trainData.map((d) => d.AGST);
  const trainHarvestRain = trainData.map((d) => d.HarvestRain);
  const trainWinterRain = trainData.map((d) => d.WinterRain);
  const trainAge = trainData.map((d) => d.Age);

  const normParams: NormalizationParams = {
    AGST: { min: Math.min(...trainAGST), max: Math.max(...trainAGST) },
    HarvestRain: { min: Math.min(...trainHarvestRain), max: Math.max(...trainHarvestRain) },
    WinterRain: { min: Math.min(...trainWinterRain), max: Math.max(...trainWinterRain) },
    Age: { min: Math.min(...trainAge), max: Math.max(...trainAge) },
  };

  // Normalize function
  const normalize = (value: number, min: number, max: number) =>
    max === min ? 0 : (value - min) / (max - min);

  // Prepare training features (normalized)
  const trainXsData = trainData.map((d) => [
    normalize(d.AGST, normParams.AGST.min, normParams.AGST.max),
    normalize(d.HarvestRain, normParams.HarvestRain.min, normParams.HarvestRain.max),
    normalize(d.WinterRain, normParams.WinterRain.min, normParams.WinterRain.max),
    normalize(d.Age, normParams.Age.min, normParams.Age.max),
  ]);

  const trainYsData = trainData.map((d) => [d.Price]);

  // Prepare test features (normalized with training params)
  const testXsData = testData.map((d) => [
    normalize(d.AGST, normParams.AGST.min, normParams.AGST.max),
    normalize(d.HarvestRain, normParams.HarvestRain.min, normParams.HarvestRain.max),
    normalize(d.WinterRain, normParams.WinterRain.min, normParams.WinterRain.max),
    normalize(d.Age, normParams.Age.min, normParams.Age.max),
  ]);

  const testYsData = testData.map((d) => [d.Price]);

  return {
    trainXs: tf.tensor2d(trainXsData),
    trainYs: tf.tensor2d(trainYsData),
    testXs: tf.tensor2d(testXsData),
    testYs: tf.tensor2d(testYsData),
    normParams,
  };
}
