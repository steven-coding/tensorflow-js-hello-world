# tensorflow-js-hello-world

Minimal TypeScript TensorFlow.js hello world with CUDA GPU support. Trains a simple linear regression model (`y = 2x - 1`) on the GPU using `@tensorflow/tfjs-node-gpu`.

## Prerequisites

- Node.js v18+
- NVIDIA GPU with CUDA support

## WSL Setup (Ubuntu 22.04)

### 1. NVIDIA Driver (Windows side)

Install the latest NVIDIA Game Ready or Studio Driver on Windows. The driver is automatically passed through to WSL — **do not install a separate Linux driver**.

### 2. CUDA Toolkit 11.8

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# If libtinfo5 is missing (dependency error with nsight-systems):
wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb

# Install CUDA Toolkit
sudo apt install cuda-toolkit-11-8
```

### 3. cuDNN 8

```bash
sudo apt install libcudnn8 libcudnn8-dev
```

### 4. Cleanup (optional)

Free up disk space by clearing APT caches after installation:

```bash
sudo apt clean
sudo apt autoremove
rm -f ~/cuda-keyring*.deb ~/libtinfo5*.deb
```

## Usage

```bash
npm install

# Set library path to CUDA 11.8 before running
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
npm start
```

### Expected output

```
TensorFlow.js backend: tensorflow
Prediction for x=10: ~19
```

## Scripts

| Command | Description |
|---|---|
| `npm start` | Run TypeScript directly (via tsx) |
| `npm run build` | Compile TypeScript to `dist/` |
| `npm run start:built` | Run the compiled output |
