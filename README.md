# vram_lock – GPU VRAM Corruption Finder / “Lock” Tool

`vram_lock` is a minimal C++ utility that tries to **find VRAM allocations that
do not reliably hold data** and then **keeps (“locks”) the suspected-bad chunk
allocated** so it can’t be reused.

It works by repeatedly allocating a configurable “slice” of VRAM on a selected
GPU, writing a deterministic byte pattern (`0xA5`), then copying the same slice
back to host memory multiple times and comparing MD5 hashes of the copies.

If the hashes ever differ for the same allocation, the program treats that slice
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
Last md5: f1c88a14a4e6020ad4121468b7365c4f

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

## What it can do (matches the code)

* Select GPU by index (`gpu_index`, default `0`)
* Select slice size in MiB (`slice_mebibytes`, default `512`)
* Allocate slices until allocation fails or corruption is detected
* For each slice:
  * fill with `0xA5`
  * copy device → host twice
  * compute MD5 of each host copy
  * compare the two MD5 hashes
* On corruption (MD5 mismatch for the same slice):
  * prints both MD5 hashes
  * frees all earlier allocations
  * keeps the “broken” allocation
  * sleeps forever holding it

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

* NVIDIA GPU with a working CUDA driver
* CUDA toolkit / CUDA Driver API headers and `-lcuda`
* OpenSSL development libraries (for the MD5 helper)

---

## Native build & run

Build however you prefer (Makefile/CMake/manual). A typical manual build looks like:

```bash
g++ -O2 -std=c++17 vram_lock.cpp -o vram_lock -lcuda -lcrypto
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

If you have a `Dockerfile` in the repo:

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

## Expected output (excerpt)

```
Starting on GPU 0 (NVIDIA ...). Slice size = 512 MiB
OK slice #0  md5=7d06eab8...  kept=1
OK slice #1  md5=7d06eab8...  kept=2
...
```

On corruption you will see something like:

```
MISMATCH at slice #42!
  md5 #1: 7d06eab8...
  md5 #2: 1be3f9c4...
Freeing all previous allocations; keeping the broken one.

Sleeping forever with the broken VRAM allocation held.
```

If VRAM is exhausted before any mismatch is detected, allocation will stop with:

```
STOP: cuMemAlloc failed at slice #N: <CUDA_ERROR_...> (<code>)
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
