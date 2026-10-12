# llama.cpp server built with CUDA, for machines with an NVIDIA GPU (see compose.gpu.yaml).
# The CPU build in llama.Dockerfile is pinned to the release the evaluation runs used.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda
