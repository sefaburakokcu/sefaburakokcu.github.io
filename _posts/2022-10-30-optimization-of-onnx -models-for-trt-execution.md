---
title: Optimization of Onnx Models for TensorRT Execution
date: 2022-10-30 14:20:00 +0300
categories: [Posts, Python]
tags: [python, tensorrt, trtexec, onnx]
---

## Introduction

TensorRT is utilized for running deep learning models on Nvidia GPUs efficiently. TensorRT supports parsing Onnx and Caffe models. In addition to TensorRT, trtexec which is a tool for using TensorRT without any development environment provides serialized engines from Onnx, Caffe or UFF models and benchmarking of networks. Further information can be obtained on [Tensorrt documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html).

In this post, optimization of Onnx models for TensorRT execution for faster inference and efficient memory usage will be investigated by using trtexec tool. 

Trtexec(also TensorRT) has different configuration options for building models(serializing) and running inference. Fistly, we will examine building options for efficient models. 

## Build Options

__A. TacticSource__

It provides tactics for TensorRT for efficient inference. Nevertheless, it leads to incrase in GPU memory. Therefore, if you need smaller memory allocation, available tactic sources should be disabled. Available options:
- CUBLAS
- CUBLAS_LT
- CUDNN
- EDGE_MASK_CONVOLUTIONS

Usage:

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan 
--tacticSources=+CUBLAS,-CUBLAS_LT,-CUDNN,+EDGE_MASK_CONVOLUTIONS
```

+:enable selected tactics                                                                                           
-:disable selected tactics

__B. fp16__

It enables float16 support in addition to float32. Weights will be converted to fp16 if fp16 flag is enabled and layers are supported. It decreases size of serialized engine file and accelerates inference.

Usage:

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16
```

There are many other available options i.e. int8, sparsity, maxBatch etc. Available options can be listed via

```bash
trtexec --help
```

## Inference Options

__A. noDataTransfers__

It disables DMA transfers to and from device and decreases GPU speed fluactuations.

Usage:

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --noDataTransfers
```

__B. useSpinWait__

It decreases synchronization time and GPU speed fluactuations but increases CPU usage and power. 

Usage:

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --useSpinWait
```

__C. streams__

It instantiates N engines to use concurrently and increases throughput per seconds(tps). However, it increas a bit of GPU memory usage.

Note: It should be set 1 which is default value if dynamic batch size is used in models.

Usage:

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --streams=2
```

__D. useCudaGraph__

It enables CUDA graph to capture engine execution and then launches inference. It increases Gpu runtime speed greatly.

Usage:

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --useCudaGraph
```

There are many other available options i.e. threads, exposeDMA etc. Available options can be listed via

```bash
trtexec --help
```

