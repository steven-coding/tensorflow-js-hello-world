# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minimal TypeScript TensorFlow.js hello world with CUDA GPU support for model training. Runs on Node.js using `@tensorflow/tfjs-node-gpu`.

## Commands

- `npm start` — run the hello world script (uses tsx)
- `npm run build` — compile TypeScript to `dist/`
- `npm run start:built` — run the compiled output

## Prerequisites

- Node.js v18+
- NVIDIA GPU with CUDA Toolkit and cuDNN installed

## Architecture

Single-file project in `src/index.ts`. Uses `@tensorflow/tfjs-node-gpu` which binds to the native TensorFlow C library with CUDA support. The `tsx` runner executes TypeScript directly without a separate compile step.
