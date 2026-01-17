/*
vram_lock.cpp
VRAM integrity stress / "lock" test (Vulkan)

Behavior:
- Allocate a configurable slice size on a configurable GPU repeatedly.
- Fill with deterministic byte pattern.
- Copy device->host twice and compare the two host copies byte-for-byte.
- If copies match: keep allocation, allocate another slice, repeat.
- If mismatch: keep the allocation (lock it), mark it as faulty, and continue.
- Continue until allocation fails (OOM). Then:
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

Build (Linux):
  g++ -O2 -std=c++17 vram_lock.cpp -o gpu-lock -lvulkan

Build (Windows, MSVC):
  cl /O2 /std:c++17 vram_lock.cpp /I path\to\vulkan\include vulkan-1.lib
*/

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#endif

static constexpr uint8_t FILL_BYTE = 0xA5;

static void die(const char* what) {
  std::fprintf(stderr, "ERROR: %s\n", what);
  std::fflush(stderr);
  std::exit(1);
}

static void die_vk(VkResult r, const char* what) {
  if (r == VK_SUCCESS) return;
  std::fprintf(stderr, "ERROR: %s failed: VkResult=%d\n", what, static_cast<int>(r));
  std::fflush(stderr);
  std::exit(1);
}

[[noreturn]] static void sleep_forever(const char* msg) {
  std::printf("%s\n", msg);
  std::fflush(stdout);
  while (true) {
#ifdef _WIN32
    Sleep(60UL * 60UL * 1000UL); // 1 hour
#else
    std::this_thread::sleep_for(std::chrono::hours(1));
#endif
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

static uint32_t find_memory_type(uint32_t type_bits,
                                VkMemoryPropertyFlags required,
                                const VkPhysicalDeviceMemoryProperties& mem_props) {
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) == 0) continue;
    const VkMemoryPropertyFlags flags = mem_props.memoryTypes[i].propertyFlags;
    if ((flags & required) == required) return i;
  }
  return UINT32_MAX;
}

struct VulkanCtx {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = UINT32_MAX;

  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  VkPhysicalDeviceMemoryProperties mem_props{};
  VkPhysicalDeviceProperties props{};

  std::string device_name;

  void destroy() {
    if (device != VK_NULL_HANDLE) {
      if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
      if (cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmd_pool, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) {
      vkDestroyInstance(instance, nullptr);
    }
    instance = VK_NULL_HANDLE;
    phys = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
    queue_family = UINT32_MAX;
    cmd_pool = VK_NULL_HANDLE;
    cmd = VK_NULL_HANDLE;
    fence = VK_NULL_HANDLE;
  }
};

static void begin_one_time(VulkanCtx& vk) {
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  die_vk(vkBeginCommandBuffer(vk.cmd, &bi), "vkBeginCommandBuffer");
}

static void end_submit_wait(VulkanCtx& vk) {
  die_vk(vkEndCommandBuffer(vk.cmd), "vkEndCommandBuffer");

  die_vk(vkResetFences(vk.device, 1, &vk.fence), "vkResetFences");

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &vk.cmd;

  die_vk(vkQueueSubmit(vk.queue, 1, &si, vk.fence), "vkQueueSubmit");
  die_vk(vkWaitForFences(vk.device, 1, &vk.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

  die_vk(vkResetCommandBuffer(vk.cmd, 0), "vkResetCommandBuffer");
}

struct Slice {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};

struct Staging {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  void* mapped = nullptr;
  bool coherent = false;
};

static void destroy_slice(VulkanCtx& vk, Slice& s) {
  if (s.buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk.device, s.buffer, nullptr);
  if (s.memory != VK_NULL_HANDLE) vkFreeMemory(vk.device, s.memory, nullptr);
  s.buffer = VK_NULL_HANDLE;
  s.memory = VK_NULL_HANDLE;
  s.size = 0;
}

static void destroy_staging(VulkanCtx& vk, Staging& s) {
  if (s.mapped) {
    vkUnmapMemory(vk.device, s.memory);
    s.mapped = nullptr;
  }
  if (s.buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk.device, s.buffer, nullptr);
  if (s.memory != VK_NULL_HANDLE) vkFreeMemory(vk.device, s.memory, nullptr);
  s.buffer = VK_NULL_HANDLE;
  s.memory = VK_NULL_HANDLE;
  s.size = 0;
  s.coherent = false;
}

static VkResult create_buffer_and_memory(VulkanCtx& vk,
                                        VkDeviceSize size,
                                        VkBufferUsageFlags usage,
                                        VkMemoryPropertyFlags mem_flags,
                                        VkBuffer& out_buf,
                                        VkDeviceMemory& out_mem) {
  out_buf = VK_NULL_HANDLE;
  out_mem = VK_NULL_HANDLE;

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult r = vkCreateBuffer(vk.device, &bci, nullptr, &out_buf);
  if (r != VK_SUCCESS) return r;

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(vk.device, out_buf, &req);

  const uint32_t mt = find_memory_type(req.memoryTypeBits, mem_flags, vk.mem_props);
  if (mt == UINT32_MAX) {
    vkDestroyBuffer(vk.device, out_buf, nullptr);
    out_buf = VK_NULL_HANDLE;
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mt;

  r = vkAllocateMemory(vk.device, &mai, nullptr, &out_mem);
  if (r != VK_SUCCESS) {
    vkDestroyBuffer(vk.device, out_buf, nullptr);
    out_buf = VK_NULL_HANDLE;
    out_mem = VK_NULL_HANDLE;
    return r;
  }

  r = vkBindBufferMemory(vk.device, out_buf, out_mem, 0);
  if (r != VK_SUCCESS) {
    vkFreeMemory(vk.device, out_mem, nullptr);
    vkDestroyBuffer(vk.device, out_buf, nullptr);
    out_buf = VK_NULL_HANDLE;
    out_mem = VK_NULL_HANDLE;
    return r;
  }

  return VK_SUCCESS;
}

static void init_vulkan(VulkanCtx& vk, unsigned int gpu_index) {
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.pApplicationName = "vram_lock";
  ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  ai.pEngineName = "none";
  ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  ai.apiVersion = VK_API_VERSION_1_0; // "any version is fine"

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &ai;

  die_vk(vkCreateInstance(&ici, nullptr, &vk.instance), "vkCreateInstance");

  uint32_t phys_count = 0;
  die_vk(vkEnumeratePhysicalDevices(vk.instance, &phys_count, nullptr), "vkEnumeratePhysicalDevices(count)");
  if (phys_count == 0) die("No Vulkan physical devices found.");

  std::vector<VkPhysicalDevice> phys(phys_count);
  die_vk(vkEnumeratePhysicalDevices(vk.instance, &phys_count, phys.data()), "vkEnumeratePhysicalDevices(list)");

  if (gpu_index >= phys_count) {
    std::fprintf(stderr, "Invalid gpu_index %u (device count = %u)\n", gpu_index, phys_count);
    std::fflush(stderr);
    std::exit(2);
  }

  vk.phys = phys[gpu_index];
  vkGetPhysicalDeviceProperties(vk.phys, &vk.props);
  vkGetPhysicalDeviceMemoryProperties(vk.phys, &vk.mem_props);
  vk.device_name = vk.props.deviceName ? vk.props.deviceName : "";

  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vk.phys, &qf_count, nullptr);
  if (qf_count == 0) die("No queue families found.");

  std::vector<VkQueueFamilyProperties> qfps(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(vk.phys, &qf_count, qfps.data());

  // Prefer a queue that supports TRANSFER; COMPUTE/GRAPHICS also fine.
  uint32_t chosen = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; ++i) {
    const VkQueueFlags f = qfps[i].queueFlags;
    if (f & VK_QUEUE_TRANSFER_BIT) {
      chosen = i;
      break;
    }
  }
  if (chosen == UINT32_MAX) {
    for (uint32_t i = 0; i < qf_count; ++i) {
      const VkQueueFlags f = qfps[i].queueFlags;
      if (f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
        chosen = i;
        break;
      }
    }
  }
  if (chosen == UINT32_MAX) die("No suitable queue family found.");

  vk.queue_family = chosen;

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo dqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  dqci.queueFamilyIndex = vk.queue_family;
  dqci.queueCount = 1;
  dqci.pQueuePriorities = &prio;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &dqci;

  die_vk(vkCreateDevice(vk.phys, &dci, nullptr, &vk.device), "vkCreateDevice");
  vkGetDeviceQueue(vk.device, vk.queue_family, 0, &vk.queue);

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpci.queueFamilyIndex = vk.queue_family;
  die_vk(vkCreateCommandPool(vk.device, &cpci, nullptr, &vk.cmd_pool), "vkCreateCommandPool");

  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = vk.cmd_pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  die_vk(vkAllocateCommandBuffers(vk.device, &cbai, &vk.cmd), "vkAllocateCommandBuffers");

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  die_vk(vkCreateFence(vk.device, &fci, nullptr, &vk.fence), "vkCreateFence");
}

struct VramLockState {
  unsigned int gpu_index = 0;
  std::string dev_name;

  unsigned int slice_mib = 0;
  size_t slice_bytes = 0;

  VulkanCtx vk;

  std::vector<Slice> slices;
  std::vector<char> map;

  size_t ok_count = 0;
  size_t bad_count = 0;

  Staging staging;

  std::vector<uint8_t> host1;
  std::vector<uint8_t> host2;

  std::string last_status = "Starting...";

  std::string last_md5_ok;
  std::string last_md5_1;
  std::string last_md5_2;

  bool finalized_after_oom = false;

  VramLockState(unsigned int gpu_index_,
                unsigned int slice_mib_,
                size_t slice_bytes_)
      : gpu_index(gpu_index_),
        slice_mib(slice_mib_),
        slice_bytes(slice_bytes_),
        host1(slice_bytes_),
        host2(slice_bytes_) {
    slices.reserve(64);
    map.reserve(256);
  }

  void init() {
    init_vulkan(vk, gpu_index);
    dev_name = vk.device_name;

    // Create a single reusable staging buffer (host-visible) for readback.
    // We prefer HOST_COHERENT; if not available, we still work with invalidate.
    VkBuffer sbuf = VK_NULL_HANDLE;
    VkDeviceMemory स्मem = VK_NULL_HANDLE;

    VkResult r = create_buffer_and_memory(
        vk,
        static_cast<VkDeviceSize>(slice_bytes),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        sbuf,
        स्मem);

    bool coherent = true;
    if (r != VK_SUCCESS) {
      coherent = false;
      r = create_buffer_and_memory(
          vk,
          static_cast<VkDeviceSize>(slice_bytes),
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
          sbuf,
          स्मem);
    }
    die_vk(r, "create staging buffer");

    staging.buffer = sbuf;
    staging.memory = स्मem;
    staging.size = static_cast<VkDeviceSize>(slice_bytes);
    staging.coherent = coherent;

    void* mapped = nullptr;
    die_vk(vkMapMemory(vk.device, staging.memory, 0, staging.size, 0, &mapped), "vkMapMemory(staging)");
    staging.mapped = mapped;
  }

  VkResult make_allocation() {
    Slice s{};
    s.size = static_cast<VkDeviceSize>(slice_bytes);

    VkResult r = create_buffer_and_memory(
        vk,
        s.size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        s.buffer,
        s.memory);

    if (r != VK_SUCCESS) {
      // Ensure no partial resources remain.
      if (s.buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk.device, s.buffer, nullptr);
      if (s.memory != VK_NULL_HANDLE) vkFreeMemory(vk.device, s.memory, nullptr);
      return r;
    }

    slices.push_back(s);
    map.push_back('?');
    return VK_SUCCESS;
  }

  void readback_to_host(VkBuffer src, uint8_t* dst) {
    begin_one_time(vk);

    // Fill/Copy are transfer operations; use a barrier to make sure fill is visible to copy,
    // and copy is visible to host.
    VkBufferMemoryBarrier b1{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.buffer = src;
    b1.offset = 0;
    b1.size = VK_WHOLE_SIZE;

    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = static_cast<VkDeviceSize>(slice_bytes);

    // Barrier before copy (after fill).
    vkCmdPipelineBarrier(vk.cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         1, &b1,
                         0, nullptr);

    vkCmdCopyBuffer(vk.cmd, src, staging.buffer, 1, &region);

    VkBufferMemoryBarrier b2{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    b2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.buffer = staging.buffer;
    b2.offset = 0;
    b2.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(vk.cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0, nullptr,
                         1, &b2,
                         0, nullptr);

    end_submit_wait(vk);

    if (!staging.coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = staging.memory;
      range.offset = 0;
      range.size = staging.size;
      die_vk(vkInvalidateMappedMemoryRanges(vk.device, 1, &range), "vkInvalidateMappedMemoryRanges");
    }

    std::memcpy(dst, staging.mapped, slice_bytes);
  }

  void fill_pattern(VkBuffer buf) {
    begin_one_time(vk);

    // vkCmdFillBuffer fills with a 32-bit pattern. Replicate FILL_BYTE across 4 bytes.
    const uint32_t pattern = (static_cast<uint32_t>(FILL_BYTE) << 24) |
                             (static_cast<uint32_t>(FILL_BYTE) << 16) |
                             (static_cast<uint32_t>(FILL_BYTE) << 8) |
                             (static_cast<uint32_t>(FILL_BYTE) << 0);

    vkCmdFillBuffer(vk.cmd, buf, 0, static_cast<VkDeviceSize>(slice_bytes), pattern);

    // Barrier to make fill visible to subsequent transfer reads (copy).
    VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buf;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(vk.cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         1, &b,
                         0, nullptr);

    end_submit_wait(vk);
  }

  void test_slice(size_t idx) {
    if (idx >= slices.size() || idx >= map.size()) {
      die("Internal error: test_slice idx out of range.");
    }

    Slice& s = slices[idx];

    last_status = "Allocated slice; filling pattern...";
    last_md5_ok.clear();
    last_md5_1.clear();
    last_md5_2.clear();

    fill_pattern(s.buffer);

    last_status = "Copying (pass 1)...";
    readback_to_host(s.buffer, host1.data());

    last_status = "Copying (pass 2)...";
    readback_to_host(s.buffer, host2.data());

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
    map[idx] = '#';
    ++ok_count;
    last_status = "OK";
  }

  void free_all_except_faulty() {
    if (slices.size() != map.size()) die("Internal error: slices/map size mismatch.");

    std::vector<Slice> kept;
    kept.reserve(count_char(map, 'X'));

    for (size_t i = 0; i < slices.size(); ++i) {
      if (map[i] == 'X') {
        kept.push_back(slices[i]);
      } else {
        destroy_slice(vk, slices[i]);
      }
    }

    slices.swap(kept);
  }

  void shutdown() {
    // Note: if we are "sleeping forever", we intentionally do not call shutdown.
    for (auto& s : slices) destroy_slice(vk, s);
    destroy_staging(vk, staging);
    vk.destroy();
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

  const auto t0 = std::chrono::steady_clock::now();
  ansi_hide_cursor();

  VramLockState state(gpu_index, slice_mib, slice_bytes);
  state.init();

  size_t idx = 0;

  while (true) {
    render_ui(state.gpu_index, state.dev_name.c_str(), state.slice_mib, state.slice_bytes, idx,
              state.slices.size(), state.finalized_after_oom, state.map, state.ok_count,
              state.bad_count, state.last_status, state.last_md5_ok, state.last_md5_1,
              state.last_md5_2, t0);

    VkResult r = state.make_allocation();
    if (r != VK_SUCCESS) {
      state.last_status =
          "STOP: allocation failed (likely OOM). Freeing all OK slices; keeping only faulty locked.";
      state.last_md5_1.clear();
      state.last_md5_2.clear();

      state.free_all_except_faulty();
      finalize_map_after_oom(state.map);
      state.finalized_after_oom = true;

      render_ui(state.gpu_index, state.dev_name.c_str(), state.slice_mib, state.slice_bytes, idx,
                state.slices.size(), state.finalized_after_oom, state.map, state.ok_count,
                state.bad_count, state.last_status, state.last_md5_ok, state.last_md5_1,
                state.last_md5_2, t0);

      std::printf("\nAllocation failed at slice #%zu: VkResult=%d\n", idx, static_cast<int>(r));
      std::fflush(stdout);

      // Intentionally do NOT destroy Vulkan objects; we want to keep faulty allocations locked.
      sleep_forever("Sleeping forever holding only faulty VRAM allocations.");
    }

    const size_t this_idx = state.slices.size() - 1;

    state.last_status = "Allocated slice; filling pattern...";
    state.last_md5_1.clear();
    state.last_md5_2.clear();

    render_ui(state.gpu_index, state.dev_name.c_str(), state.slice_mib, state.slice_bytes, idx,
              state.slices.size(), state.finalized_after_oom, state.map, state.ok_count,
              state.bad_count, state.last_status, state.last_md5_ok, state.last_md5_1,
              state.last_md5_2, t0);

    state.test_slice(this_idx);

    render_ui(state.gpu_index, state.dev_name.c_str(), state.slice_mib, state.slice_bytes, idx,
              state.slices.size(), state.finalized_after_oom, state.map, state.ok_count,
              state.bad_count, state.last_status, state.last_md5_ok, state.last_md5_1,
              state.last_md5_2, t0);

    ++idx;
  }

  // Unreachable, but keep symmetry.
  state.shutdown();
  ansi_show_cursor();
  return 0;
}
