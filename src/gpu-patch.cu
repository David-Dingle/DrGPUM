/*
 * Use C style programming in this file
 */
#include "gpu-patch.h"
#include "gpu-queue.h"
#include "utils.h"

#include <sanitizer_patching.h>


/*
 * Monitor each shared and global memory access.
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  gpu_patch_buffer_t *buffer = (gpu_patch_buffer_t *)user_data;

  if (!sample_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  // 1. Init values
  uint32_t active_mask = __ballot_sync(0xFFFFFFFF, 1);
  uint32_t laneid = get_laneid();
  uint32_t first_laneid = __ffs(active_mask) - 1;

  // 2. Read memory values
  uint8_t buf[MAX_ACCESS_SIZE];
  if (new_value == NULL) {
    // Read operation, old value can be on local memory, shared memory, or global memory
    if (flags & SANITIZER_MEMORY_DEVICE_FLAG_SHARED) {
      read_shared_memory(size, (uint32_t)address, buf);
    } else if (flags & SANITIZER_MEMORY_DEVICE_FLAG_LOCAL) {
      read_local_memory(size, (uint32_t)address, buf);
    } else if (flags != SANITIZER_MEMORY_DEVICE_FLAG_FORCE_INT) {
      read_global_memory(size, (uint64_t)address, buf);
    }
  } else {
    // Write operation, new value is on global memory
    read_global_memory(size, (uint64_t)new_value, buf);
  }

  if (laneid == first_laneid) {
    // 3. Get a record
    gpu_patch_record_t *record = gpu_queue_get(buffer); 

    // 4. Assign basic values
    record->pc = pc;
    record->size = size;
    record->flat_thread_id = get_flat_thread_id();
    record->flat_block_id = get_flat_block_id();
    for (uint32_t i = 0; i < WARP_SIZE; i++) {
      uint64_t addr = (uint64_t)address;
      record->address[i] = shfl(addr, i);
      record->flags[i] = shfl(flags, i);
      for (uint32_t j = 0; j < size; ++j) {
        record->value[i][j] = shfl(buf[j], i);
      }
    }

    // 5. Push a record
    gpu_queue_push(buffer);
  }

  return SANITIZER_PATCH_SUCCESS;
}


/*
 * Lock the corresponding hash entry for a block
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_block_exit_callback
(
 void *user_data,
 uint64_t pc
)
{
  gpu_patch_buffer_t* buffer = (gpu_patch_buffer_t *)user_data;

  if (!sample_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  // Finish one block
  atomicAdd(&buffer->num_blocks, -1);

  return SANITIZER_PATCH_SUCCESS;
}


/*
 * Sample the corresponding blocks
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_block_enter_callback
(
 void *user_data,
 uint64_t pc
)
{
  gpu_patch_buffer_t* buffer = (gpu_patch_buffer_t *)user_data;

  if (!sample_callback(buffer->block_sampling_frequency)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  return SANITIZER_PATCH_SUCCESS;
}