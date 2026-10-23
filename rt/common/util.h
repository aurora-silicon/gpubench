// SPDX-License-Identifier: GPL-2.0-only
/* Clock, PNG writer and small helpers shared by both executors. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* Monotonic clock in nanoseconds (CLOCK_MONOTONIC on Linux, mach_absolute_time on macOS). */
uint64_t rt_now_ns(void);
void rt_sleep_ms(unsigned ms);
uint32_t rt_crc32(uint32_t crc, const void *data, size_t n);
/* Writes an 8-bit RGB PNG (stored deflate blocks, no compression library) from RGBA8 pixels.
 * Returns 0 on success. */
int rt_write_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h, uint32_t stride);
/* Fraction of pixels whose alpha byte is >= 128 (the composite pass stores "reflective" there). */
double rt_alpha_fraction(const uint8_t *rgba, uint32_t w, uint32_t h, uint32_t stride);
/* Half float to float. */
float rt_half_to_float(uint16_t h);
/* Prints a GPU/thermal snapshot as a JSON object into buf (best effort; "{}" when nothing is known). */
void rt_thermal_snapshot(char *buf, size_t n);
