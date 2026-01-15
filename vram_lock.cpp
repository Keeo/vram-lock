/*
vram_lock.cpp
VRAM integrity stress / "lock" test (CUDA Driver API)

Behavior:
- Allocate a configurable slice size on a configurable GPU repeatedly.
- Fill with deterministic byte pattern.
- Copy device->host twice and MD5 hash both copies.
- If hashes match: keep allocation, allocate another slice, repeat.
- If mismatch: free all previous allocations, keep the broken one,
  then sleep forever holding that allocation.

Usage:
  ./gpu-lock [gpu_index] [slice_mebibytes]
Defaults:
  gpu_index = 0
  slice_mebibytes = 512

Build:
  g++ -O2 -std=c++17 vram_lock.cpp -o gpu-lock -lcuda -lcrypto
*/

#include <cuda.h>

#include <openssl/md5.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static constexpr unsigned int FILL_BYTE = 0xA5;

static void die_cuda(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;

  const char* name = nullptr;
  const char* desc = nullptr;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &desc);

  std::fprintf(stderr, "ERROR: %s failed: %s (%d) - %s\n",
               what,
               name ? name : "UNKNOWN",
               static_cast<int>(r),
               desc ? desc : "no description");
  std::fflush(stderr);
  std::exit(1);
}

static std::string md5_hex(const uint8_t* data, size_t n) {
  unsigned char digest[MD5_DIGEST_LENGTH];
  MD5(data, n, digest);

  static const char* hex = "0123456789abcdef";
  std::string out;
  out.resize(MD5_DIGEST_LENGTH * 2);
  for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
    out[i * 2 + 0] = hex[(digest[i] >> 4) & 0xF];
    out[i * 2 + 1] = hex[(digest[i] >> 0) & 0xF];
  }
  return out;
}

[[noreturn]] static void sleep_forever(const char* msg) {
  std::printf("%s\n", msg);
  std::fflush(stdout);
  while (true) {
    std::this_thread::sleep_for(std::chrono::hours(1));
  }
}

static void usage(const char* argv0) {
  std::printf(
      "Usage: %s [gpu_index] [slice_mebibytes]\n"
      "Defaults: gpu_index=0 slice_mebibytes=512\n",
      argv0);
}

static bool parse_u32(const char* s, unsigned int* out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  unsigned long v = std::strtoul(s, &end, 10);
  if (!end || *end != '\0') return false;
  if (v > 0xFFFFFFFFul) return false;
  *out = static_cast<unsigned int>(v);
  return true;
}

int main(int argc, char** argv) {
  unsigned int gpu_index = 0;
  unsigned int slice_mib = 512;

  if (argc >= 2) {
    if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
      usage(argv[0]);
      return 0;
    }
    if (!parse_u32(argv[1], &gpu_index)) {
      std::fprintf(stderr, "Invalid gpu_index: '%s'\n", argv[1]);
      usage(argv[0]);
      return 2;
    }
  }

  if (argc >= 3) {
    if (!parse_u32(argv[2], &slice_mib) || slice_mib == 0) {
      std::fprintf(stderr, "Invalid slice_mebibytes: '%s'\n", argv[2]);
      usage(argv[0]);
      return 2;
    }
  }

  if (argc >= 4) {
    std::fprintf(stderr, "Too many arguments.\n");
    usage(argv[0]);
    return 2;
  }

  const size_t slice_bytes = static_cast<size_t>(slice_mib) * 1024ull * 1024ull;

  // --- init + context on selected device ---
  die_cuda(cuInit(0), "cuInit");

  int device_count = 0;
  die_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
  if (device_count <= 0) {
    std::fprintf(stderr, "No CUDA devices found.\n");
    return 1;
  }
  if (gpu_index >= static_cast<unsigned int>(device_count)) {
    std::fprintf(stderr, "Invalid gpu_index %u (device count = %d)\n", gpu_index, device_count);
    return 2;
  }

  CUdevice dev = 0;
  die_cuda(cuDeviceGet(&dev, static_cast<int>(gpu_index)), "cuDeviceGet");

  char dev_name[256];
  dev_name[0] = '\0';
  (void)cuDeviceGetName(dev_name, sizeof(dev_name), dev);

  CUcontext ctx = nullptr;
  die_cuda(cuCtxCreate(&ctx, nullptr, 0, dev), "cuCtxCreate");

  std::vector<CUdeviceptr> allocations;
  allocations.reserve(64);

  std::printf("Starting on GPU %u (%s). Slice size = %u MiB\n", gpu_index, dev_name, slice_mib);
  std::fflush(stdout);

  size_t idx = 0;

  // Host buffers reused each iteration
  std::vector<uint8_t> host1(slice_bytes);
  std::vector<uint8_t> host2(slice_bytes);

  auto cleanup = [&]() {
    std::printf("\nCleaning up allocations...\n");
    std::fflush(stdout);
    for (auto p : allocations) {
      (void)cuMemFree(p);
    }
    allocations.clear();
    (void)cuCtxDestroy(ctx);
  };

  try {
    while (true) {
      CUdeviceptr dptr = 0;
      CUresult r = cuMemAlloc(&dptr, slice_bytes);

      if (r != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(r, &name);
        std::printf("STOP: cuMemAlloc failed at slice #%zu: %s (%d)\n",
                    idx, name ? name : "UNKNOWN", static_cast<int>(r));
        std::fflush(stdout);
        break;
      }

      // fill pattern
      die_cuda(cuMemsetD8(dptr, static_cast<unsigned char>(FILL_BYTE), slice_bytes), "cuMemsetD8");

      // copy twice
      die_cuda(cuMemcpyDtoH(host1.data(), dptr, slice_bytes), "cuMemcpyDtoH #1");
      std::string h1 = md5_hex(host1.data(), host1.size());

      die_cuda(cuMemcpyDtoH(host2.data(), dptr, slice_bytes), "cuMemcpyDtoH #2");
      std::string h2 = md5_hex(host2.data(), host2.size());

      if (h1 != h2) {
        std::printf(
            "\nMISMATCH at slice #%zu!\n"
            "  md5 #1: %s\n"
            "  md5 #2: %s\n"
            "Freeing all previous allocations; keeping the broken one.\n\n",
            idx, h1.c_str(), h2.c_str());
        std::fflush(stdout);

        // Free all previously successful allocations.
        for (CUdeviceptr p : allocations) {
          (void)cuMemFree(p);
        }
        allocations.clear();

        // Keep the broken allocation held.
        allocations.push_back(dptr);
        sleep_forever("Sleeping forever with the broken VRAM allocation held.");
      }

      allocations.push_back(dptr);
      std::printf("OK slice #%zu  md5=%s  kept=%zu\n", idx, h1.c_str(), allocations.size());
      std::fflush(stdout);
      ++idx;
    }
  } catch (...) {
    // Shouldn't really happen (we exit on CUDA errors), but keep symmetry.
  }

  cleanup();
  return 0;
}
