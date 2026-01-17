# vram_lock – GPU VRAM Corruption Finder / “Lock” Tool

`vram_lock` is a minimal C++ utility that tries to **find VRAM allocations that
do not reliably hold data** and then **keeps (“locks”) the suspected-bad chunk
allocated** so it can’t be reused.

## What it’s for

This tool is useful when a GPU has **VRAM errors** (often due to aging hardware,
unstable overclocks/undervolts, or marginal memory). By finding a slice of VRAM
that fails to retain data and then keeping that slice allocated, you can sometimes
reduce or eliminate crashes/artifacts by preventing the driver from reusing the problematic pages.

It can also be used to **make a marginal VRAM overclock usable** by effectively
leaving the least-capable part of memory behind: if one region of VRAM is the first
to fail at a given memory clock, `vram_lock` may find that region and keep it
allocated so the driver is less likely to place new allocations there.

## How it works

It works by repeatedly allocating a configurable “slice” of VRAM on a selected
GPU, writing a deterministic byte pattern (`0xA5`), then copying the same slice
back to host memory multiple times and comparing the results.

If the copies ever differ for the same allocation, the program treats that slice
as corrupted/unstable, **frees all previously successful allocations**, keeps the
faulty allocation resident (“locks” it), and then sleeps forever so the bad pages
cannot be reused.

```
GPU 0 (NVIDIA GeForce RTX 4090)
Slice size: 64 MiB (67108864 bytes)
Slices held (locked faulty): 1   OK: 358   Faulty locked: 1   In-progress: 0
Slices held (allocations): 1
Map entries: 359
Total held: 64 MiB
Elapsed: 47s
Next slice index: 359
Last status: STOP: cuMemAlloc failed (likely OOM). Freeing all OK slices; keeping only faulty locked.
Last check: mismatch detected between repeated readbacks

VRAM slice map ('#'=allocated OK, 'X'=faulty locked, '?'=in-progress, '.'=freed after OOM)
     0: ..................X.............................................
    64: ................................................................
   128: ................................................................
   192: ................................................................
   256: ................................................................
   320: .......................................

cuMemAlloc failed at slice #359: CUDA_ERROR_OUT_OF_MEMORY (2) - out of memory
Sleeping forever holding only faulty VRAM allocations.
```

---

### Command-line usage

```
./<binary> [gpu_index] [slice_mebibytes]
Defaults: gpu_index=0 slice_mebibytes=512
```

Examples:

```bash
./vram_lock              # GPU 0, 512 MiB slices
./vram_lock 1            # GPU 1, 512 MiB slices
./vram_lock 0 1024       # GPU 0, 1 GiB slices
```

---

## Requirements

- NVIDIA GPU with a working CUDA driver
- CUDA Driver API available at runtime (`libcuda`)

### Platforms / binaries

This project is intended to run on Linux and Windows (where a CUDA driver is
available).

A prebuilt Windows binary is available here:
https://github.com/Keeo/vram-lock/blob/master/vram_lock.exe

---

## Native build & run

Build however you prefer (Makefile/CMake/manual). A typical manual build looks like:

```bash
g++ -O2 -std=c++17 vram_lock.cpp -o vram_lock -lcuda
./vram_lock
```

### Note on `cuCtxCreate` signature

This project uses the 4-argument form:

```cpp
cuCtxCreate(&ctx, nullptr, 0, dev);
```

Some older CUDA headers expose a different signature (e.g. 3 arguments). If you
see a compile error around `cuCtxCreate`, adjust the call to match your installed
CUDA headers.

---

## Docker build & run

This repo includes a `Dockerfile`:

```bash
docker build -t vram_lock .
docker run --rm --gpus all vram_lock
```

To test if your GPU is faulty, you can also use `gpu_burn`:

```bash
https://github.com/wilicc/gpu-burn
docker run --rm --gpus all gpu_burn
```

---

## Cleaning up / stopping

Press <kbd>Ctrl+C</kbd> to terminate the program.

Note: the program does not currently install a signal handler to explicitly free
allocations on SIGINT; on exit, the OS/driver will reclaim resources when the
process terminates.

---

## License

Distributed under the terms of the **MIT License** (see `LICENSE`).
