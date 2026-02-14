import * as tf from "@tensorflow/tfjs-node-gpu";
import * as fs from "fs";
import * as path from "path";

const IMAGE_EXTENSIONS = /\.(jpg|jpeg|png)$/i;

export async function resizeSingleImage(
  srcPath: string,
  destPath: string,
  width: number,
  height: number
): Promise<void> {
  const buffer = fs.readFileSync(srcPath);
  const decoded = tf.node.decodeImage(new Uint8Array(buffer), 3) as tf.Tensor3D;
  const resized = tf.image.resizeBilinear(decoded, [height, width]);
  const uint8 = resized.cast("int32") as tf.Tensor3D;

  const encoded = await tf.node.encodeJpeg(uint8);
  fs.writeFileSync(destPath, encoded);

  decoded.dispose();
  resized.dispose();
  uint8.dispose();
}

export async function resizeImages(
  srcDir: string,
  destDir: string,
  width: number,
  height: number
): Promise<number> {
  const files = fs.readdirSync(srcDir).filter((f) => IMAGE_EXTENSIONS.test(f));

  if (files.length === 0) {
    console.log(`No images found in ${srcDir}`);
    return 0;
  }

  fs.mkdirSync(destDir, { recursive: true });

  let processed = 0;
  let skipped = 0;

  for (const file of files) {
    const srcPath = path.join(srcDir, file);
    const destName = path.parse(file).name + ".jpg";
    const destPath = path.join(destDir, destName);

    try {
      await resizeSingleImage(srcPath, destPath, width, height);
      processed++;

      if (processed % 100 === 0) {
        console.log(`  Resized ${processed}/${files.length} images...`);
      }
    } catch (err) {
      skipped++;
      console.warn(`  Skipped ${file}: ${err instanceof Error ? err.message : err}`);
    }
  }

  console.log(`Resize complete: ${processed} processed, ${skipped} skipped`);
  return processed;
}
