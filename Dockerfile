# ---- build stage ----
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY vram_lock.cpp /src/vram_lock.cpp

# CUDA paths inside the image
ENV CUDA_HOME=/usr/local/cuda

# Build (explicit include + lib paths for cuda.h / libcuda stubs)
RUN g++ -O2 -std=c++17 \
    -I${CUDA_HOME}/include \
    -I${CUDA_HOME}/targets/x86_64-linux/include \
    vram_lock.cpp -o vram_lock \
    -L${CUDA_HOME}/lib64 \
    -L${CUDA_HOME}/targets/x86_64-linux/lib \
    -lcuda -lcrypto


# ---- runtime stage ----
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/vram_lock /app/vram_lock

CMD ["./vram_lock"]