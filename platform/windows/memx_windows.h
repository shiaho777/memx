#ifndef MEMX_WINDOWS_H
#define MEMX_WINDOWS_H

#include <stddef.h>
#include <stdint.h>

int memx_win_init(size_t region_bytes, uint16_t tensor_role, uint16_t tensor_dtype,
                  uint32_t tensor_flags);
void memx_win_shutdown(void);
uint8_t *memx_win_region(void);
size_t memx_win_region_bytes(void);
uint64_t memx_win_compressed_pages(void);
uint64_t memx_win_faults(void);
void memx_win_dump_states(void);

#endif