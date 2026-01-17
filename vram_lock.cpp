/*
vram_lock.cpp
VRAM integrity stress / "lock" test (CUDA Driver API)

Behavior:
- Allocate a configurable slice size on a configurable GPU repeatedly.
- Fill with deterministic byte pattern.
- Copy device->host twice and compare the two host copies byte-for-byte.
- If copies match: keep allocation, allocate another slice, repeat.
- If mismatch: keep the allocation (lock it), mark it as faulty, and continue.
- Continue until cuMemAlloc fails (OOM). Then:
    NEW BEHAVIOR: free all non-faulty allocations and keep only faulty chunks locked.
    Sleep forever holding only the faulty allocations.

UI:
- Simple ANSI terminal UI showing an ASCII "VRAM map" of slices:
    '#' = allocated + verified OK (still held)
    'X' = mismatch detected (faulty chunk locked)
    '?' = allocated and currently being processed (in-progress)
    '.' = freed after OOM (visual "cleared" state)
- Shows counters and last result.

Usage:
  ./gpu-lock [gpu_index] [slice_mebibytes]
Defaults:
  gpu_index = 0
  slice_mebibytes = 512

Build:
  g++ -O2 -std=c++17 vram_lock.cpp -o gpu-lock -lcuda
*/

#include <cuda.h>

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

static void ansi_clear_screen() {
  // Clear screen + move cursor to home.
  std::fputs("\x1b[2J\x1b[H", stdout);
}

static void ansi_hide_cursor() {
  std::fputs("\x1b[?25l", stdout);
}

static void ansi_show_cursor() {
  std::fputs("\x1b[?25h", stdout);
}

static size_t count_char(const std::vector<char>& v, char c) {
  size_t n = 0;
  for (char x : v) {
    if (x == c) ++n;
  }
  return n;
}

static void finalize_map_after_oom(std::vector<char>& map) {
  // After OOM we free all non-faulty allocations; show freed blocks as '.'
  // Keep 'X' as-is.
  for (char& c : map) {
    if (c == '#' || c == '?') c = '.';
  }
}

static void render_ui(unsigned int gpu_index,
                      const char* dev_name,
                      unsigned int slice_mib,
                      size_t slice_bytes,
                      size_t idx_next,
                      size_t allocations_held,
                      bool finalized_after_oom,
                      const std::vector<char>& map,
                      size_t ok_count,
                      size_t bad_count,
                      const std::string& last_status,
                      const std::string& last_md5_ok,
                      const std::string& last_md5_1,
                      const std::string& last_md5_2,
                      std::chrono::steady_clock::time_point t0) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - t0).count();

  const size_t in_progress = finalized_after_oom ? 0 : count_char(map, '?');
  const size_t locked_faulty = count_char(map, 'X');

  ansi_clear_screen();

  std::printf("GPU %u (%s)\n", gpu_index, dev_name ? dev_name : "");
  std::printf("Slice size: %u MiB (%zu bytes)\n", slice_mib, slice_bytes);

  if (finalized_after_oom) {
    std::printf("Slices held (locked faulty): %zu   OK: %zu   Faulty locked: %zu   In-progress: %zu\n",
                locked_faulty, ok_count, bad_count, in_progress);
    std::printf("Slices held (allocations): %zu\n", allocations_held);
  } else {
    std::printf("Slices held (allocations): %zu   OK: %zu   Faulty locked: %zu   In-progress: %zu\n",
                allocations_held, ok_count, bad_count, in_progress);
  }

  std::printf("Map entries: %zu\n", map.size());

  std::printf("Total held: %zu MiB\n", (allocations_held * static_cast<size_t>(slice_mib)));
  std::printf("Elapsed: %llds\n", static_cast<long long>(elapsed));
  std::printf("Next slice index: %zu\n", idx_next);
  std::printf("Last status: %s\n", last_status.c_str());

  // These fields are kept for UI compatibility, but now carry simple compare info.
  if (!last_md5_1.empty() && !last_md5_2.empty()) {
    std::printf("Last compare #1: %s\n", last_md5_1.c_str());
    std::printf("Last compare #2: %s\n", last_md5_2.c_str());
  } else if (!last_md5_ok.empty()) {
    std::printf("Last compare: %s\n", last_md5_ok.c_str());
  }

  std::printf("\nVRAM slice map ('#'=allocated OK, 'X'=faulty locked, '?'=in-progress, '.'=freed after OOM)\n");

  constexpr size_t COLS = 64;
  for (size_t i = 0; i < map.size(); i += COLS) {
    const size_t end = (i + COLS < map.size()) ? (i + COLS) : map.size();
    std::printf("%6zu: ", i);
    for (size_t j = i; j < end; ++j) {
      std::putchar(map[j]);
    }
    std::putchar('\n');
  }

  std::fflush(stdout);
}

struct VramLockState {
  unsigned int gpu_index = 0;
  const char* dev_name = nullptr;

  unsigned int slice_mib = 0;
  size_t slice_bytes = 0;

  std::vector<CUdeviceptr> allocations;
  std::vector<char> map;

  size_t ok_count = 0;
  size_t bad_count = 0;

  std::vector<uint8_t> host1;
  std::vector<uint8_t> host2;

  std::string last_status = "Starting...";

  // Kept names to minimize UI churn; now used for compare status strings.
  std::string last_md5_ok;
  std::string last_md5_1;
  std::string last_md5_2;

  bool finalized_after_oom = false;

  VramLockState(unsigned int gpu_index_,
                const char* dev_name_,
                unsigned int slice_mib_,
                size_t slice_bytes_)
      : gpu_index(gpu_index_),
        dev_name(dev_name_),
        slice_mib(slice_mib_),
        slice_bytes(slice_bytes_),
        host1(slice_bytes_),
        host2(slice_bytes_) {
    allocations.reserve(64);
    map.reserve(256);
  }

  CUresult make_allocation() {
    CUdeviceptr dptr = 0;
    CUresult r = cuMemAlloc(&dptr, slice_bytes);
    if (r != CUDA_SUCCESS) return r;

    allocations.push_back(dptr);
    map.push_back('?');
    return CUDA_SUCCESS;
  }

  void test_pointer(size_t idx) {
    if (idx >= allocations.size() || idx >= map.size()) {
      std::fprintf(stderr, "Internal error: test_pointer idx out of range.\n");
      std::fflush(stderr);
      std::exit(1);
    }

    CUdeviceptr dptr = allocations[idx];

    last_status = "Allocated slice; filling pattern...";
    last_md5_1.clear();
    last_md5_2.clear();

    die_cuda(cuMemsetD8(dptr, static_cast<unsigned char>(FILL_BYTE), slice_bytes), "cuMemsetD8");

    last_status = "Copying (pass 1)...";
    die_cuda(cuMemcpyDtoH(host1.data(), dptr, slice_bytes), "cuMemcpyDtoH #1");

    last_status = "Copying (pass 2)...";
    die_cuda(cuMemcpyDtoH(host2.data(), dptr, slice_bytes), "cuMemcpyDtoH #2");

    last_status = "Comparing host copies...";
    const int cmp = std::memcmp(host1.data(), host2.data(), slice_bytes);

    if (cmp != 0) {
      last_md5_1 = "DIFFERENT";
      last_md5_2 = "DIFFERENT";

      map[idx] = 'X';
      ++bad_count;
      last_status = "MISMATCH detected: locking faulty chunk and continuing...";
      return;
    }

    last_md5_ok = "MATCH";
    last_md5_1.clear();
    last_md5_2.clear();

    map[idx] = '#';
    ++ok_count;

    last_status = "OK";
  }

  // After OOM: free all non-faulty allocations and keep only the faulty ones ('X').
  void free_all_except_faulty() {
    if (allocations.size() != map.size()) {
      std::fprintf(stderr, "Internal error: allocations/map size mismatch.\n");
      std::fflush(stderr);
      std::exit(1);
    }

    std::vector<CUdeviceptr> kept;
    kept.reserve(count_char(map, 'X'));

    for (size_t i = 0; i < allocations.size(); ++i) {
      CUdeviceptr dptr = allocations[i];
      if (map[i] == 'X') {
        kept.push_back(dptr);
      } else {
        // Best-effort free; if it fails, treat as fatal because we want to release VRAM.
        die_cuda(cuMemFree(dptr), "cuMemFree");
      }
    }

    allocations.swap(kept);
  }
};

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

  auto cleanup = [&]() {
    ansi_show_cursor();
    std::printf("\nCleaning up allocations...\n");
    std::fflush(stdout);
    (void)cuCtxDestroy(ctx);
  };

  const auto t0 = std::chrono::steady_clock::now();

  ansi_hide_cursor();

  VramLockState state(gpu_index, dev_name, slice_mib, slice_bytes);

  size_t idx = 0;

  try {
    while (true) {
      render_ui(state.gpu_index, state.dev_name, state.slice_mib, state.slice_bytes, idx,
                state.allocations.size(), state.finalized_after_oom, state.map, state.ok_count,
                state.bad_count, state.last_status, state.last_md5_ok, state.last_md5_1,
                state.last_md5_2, t0);

      CUresult r = state.make_allocation();
      if (r != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* desc = nullptr;
        cuGetErrorName(r, &name);
        cuGetErrorString(r, &desc);

        state.last_status =
            "STOP: cuMemAlloc failed (likely OOM). Freeing all OK slices; keeping only faulty locked.";
        state.last_md5_1.clear();
        state.last_md5_2.clear();

        // Free everything except faulty chunks.
        state.free_all_except_faulty();

        // Update map visualization to show freed blocks.
        finalize_map_after_oom(state.map);
        state.finalized_after_oom = true;

        render_ui(state.gpu_index, state.dev_name, state.slice_mib, state.slice_bytes, idx,
                  state.allocations.size(), state.finalized_after_oom, state.map, state.ok_count,
                  state.bad_count, state.last_status, state.last_md5_ok, state.last_md5_1,
                  state.last_md5_2, t0);

        std::printf("\ncuMemAlloc failed at slice #%zu: %s (%d) - %s\n",
                    idx, name ? name : "UNKNOWN", static_cast<int>(r), desc ? desc : "");
        std::fflush(stdout);

        sleep_forever("Sleeping forever holding only faulty VRAM allocations.");
      }

      const size_t this_idx = state.allocations.size() - 1;

      state.last_status = "Allocated slice; filling pattern...";
      state.last_md5_1.clear();
      state.last_md5_2.clear();

      render_ui(state.gpu_index, state.dev_name, state.slice_mib, state.slice_bytes, idx,
                state.allocations.size(), state.finalized_after_oom, state.map, state.ok_count,
                state.bad_count, state.last_status, state.last_md5_ok, state.last_md5_1,
                state.last_md5_2, t0);

      state.test_pointer(this_idx);

      render_ui(state.gpu_index, state.dev_name, state.slice_mib, state.slice_bytes, idx,
                state.allocations.size(), state.finalized_after_oom, state.map, state.ok_count,
                state.bad_count, state.last_status, state.last_md5_ok, state.last_md5_1,
                state.last_md5_2, t0);

      ++idx;
    }
  } catch (...) {
    // Shouldn't really happen (we exit on CUDA errors), but keep symmetry.
  }

  cleanup();
  return 0;
}
