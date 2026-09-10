#ifndef MEMX_POSIX_H
#define MEMX_POSIX_H

#include <stddef.h>
#include <stdint.h>

int memx_posix_init(size_t region_bytes, uint16_t tensor_role, uint16_t tensor_dtype,
                   uint32_t tensor_flags);
void memx_posix_shutdown(void);
uint8_t *memx_posix_region(void);
size_t memx_posix_region_bytes(void);
uint64_t memx_posix_compressed_pages(void);
uint64_t memx_posix_faults(void);
void memx_posix_dump_states(void);

#endif
