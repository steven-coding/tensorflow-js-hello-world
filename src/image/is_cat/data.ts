import * as tf from "@tensorflow/tfjs-node-gpu";
import * as fs from "fs";
import * as path from "path";
import { IMAGE_SIZE, CHANNELS } from "./model";

export const DATA_DIR = path.resolve("data", "is_cat");
export const MAX_IMAGES_PER_CLASS = 1000;

const IMAGE_EXTENSIONS = /\.(jpg|jpeg|png)$/i;

export type SplitName = "train" | "validation" | "test";

export interface ImageDataset {
  xs: tf.Tensor4D;
  ys: tf.Tensor2D;
}

export function loadAndPreprocessImage(filePath: string): tf.Tensor3D {
  const buffer = fs.readFileSync(filePath);
  const decoded = tf.node.decodeImage(new Uint8Array(buffer), CHANNELS) as tf.Tensor3D;
  const resized = tf.image.resizeBilinear(decoded, [IMAGE_SIZE, IMAGE_SIZE]);
  const normalized = resized.div(255.0) as tf.Tensor3D;

  decoded.dispose();
  resized.dispose();

  return normalized;
}

function shuffleArray<T>(array: T[]): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function listImages(dir: string, maxCount: number): string[] {
  if (!fs.existsSync(dir)) return [];
  const files = fs.readdirSync(dir).filter((f) => IMAGE_EXTENSIONS.test(f));
  return files.slice(0, maxCount).map((f) => path.join(dir, f));
}

export function loadSplit(split: SplitName): ImageDataset {
  const catDir = path.join(DATA_DIR, split, "cat");
  const notCatDir = path.join(DATA_DIR, split, "not_cat");

  const catFiles = listImages(catDir, MAX_IMAGES_PER_CLASS);
  const notCatFiles = listImages(notCatDir, MAX_IMAGES_PER_CLASS);

  console.log(`  [${split}] Found ${catFiles.length} cat, ${notCatFiles.length} not-cat images`);

  const samples: { filePath: string; label: number }[] = [
    ...catFiles.map((filePath) => ({ filePath, label: 1 })),
    ...notCatFiles.map((filePath) => ({ filePath, label: 0 })),
  ];

  shuffleArray(samples);

  const images: tf.Tensor3D[] = [];
  const labels: number[] = [];
  let skipped = 0;

  for (const sample of samples) {
    try {
      const tensor = loadAndPreprocessImage(sample.filePath);
      images.push(tensor);
      labels.push(sample.label);
    } catch (err) {
      skipped++;
      console.warn(`  Skipped ${sample.filePath}: ${err instanceof Error ? err.message : err}`);
    }
  }

  if (skipped > 0) {
    console.log(`  [${split}] Skipped ${skipped} corrupt images`);
  }

  const xs = tf.stack(images) as tf.Tensor4D;
  const ys = tf.tensor2d(labels, [labels.length, 1]);

  for (const img of images) {
    img.dispose();
  }

  console.log(`  [${split}] Dataset shape: xs=${xs.shape}, ys=${ys.shape}`);
  return { xs, ys };
}
