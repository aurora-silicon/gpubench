// SPDX-License-Identifier: GPL-2.0-only
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif
#ifdef __linux__
#include <dirent.h>
#endif

uint64_t rt_now_ns(void)
{
#ifdef __APPLE__
   static mach_timebase_info_data_t tb;
   if (!tb.denom) mach_timebase_info(&tb);
   return mach_absolute_time() * tb.numer / tb.denom;
#else
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

void rt_sleep_ms(unsigned ms)
{
   struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
   nanosleep(&ts, NULL);
}

uint32_t rt_crc32(uint32_t crc, const void *data, size_t n)
{
   static uint32_t table[256];
   if (!table[1])
      for (uint32_t i = 0; i < 256; i++) {
         uint32_t c = i;
         for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
         table[i] = c;
      }
   const uint8_t *p = data;
   crc = ~crc;
   for (size_t i = 0; i < n; i++) crc = table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
   return ~crc;
}

static void be32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }

static void chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n)
{
   uint8_t hdr[8];
   be32(hdr, n);
   memcpy(hdr + 4, type, 4);
   fwrite(hdr, 1, 8, f);
   if (n) fwrite(data, 1, n, f);
   uint32_t c = rt_crc32(0, type, 4);
   c = rt_crc32(c, data, n);
   uint8_t t[4];
   be32(t, c);
   fwrite(t, 1, 4, f);
}

int rt_write_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h, uint32_t stride)
{
   FILE *f = fopen(path, "wb");
   if (!f) return -1;
   static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
   fwrite(sig, 1, 8, f);
   uint8_t ihdr[13];
   be32(ihdr, w);
   be32(ihdr + 4, h);
   ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
   chunk(f, "IHDR", ihdr, 13);
   size_t row = (size_t)w * 3 + 1, raw_n = row * h;
   uint8_t *raw = malloc(raw_n);
   for (uint32_t y = 0; y < h; y++) {
      uint8_t *o = raw + y * row;
      const uint8_t *s = rgba + (size_t)y * stride;
      o[0] = 0;
      for (uint32_t x = 0; x < w; x++) {
         o[1 + x * 3] = s[x * 4]; o[2 + x * 3] = s[x * 4 + 1]; o[3 + x * 3] = s[x * 4 + 2];
      }
   }
   /* zlib stream with stored blocks of at most 65535 bytes */
   size_t nblk = (raw_n + 65534) / 65535;
   size_t z_n = 2 + raw_n + nblk * 5 + 4;
   uint8_t *z = malloc(z_n), *o = z;
   *o++ = 0x78; *o++ = 0x01;
   uint32_t a = 1, b = 0;
   for (size_t off = 0; off < raw_n;) {
      size_t n = raw_n - off > 65535 ? 65535 : raw_n - off;
      *o++ = off + n == raw_n ? 1 : 0;
      *o++ = n & 0xff; *o++ = n >> 8; *o++ = ~n & 0xff; *o++ = (~n >> 8) & 0xff;
      memcpy(o, raw + off, n);
      for (size_t i = 0; i < n; i++) { a = (a + raw[off + i]) % 65521; b = (b + a) % 65521; }
      o += n;
      off += n;
   }
   be32(o, (b << 16) | a);
   o += 4;
   chunk(f, "IDAT", z, (uint32_t)(o - z));
   chunk(f, "IEND", NULL, 0);
   free(raw);
   free(z);
   return fclose(f) ? -1 : 0;
}

double rt_alpha_fraction(const uint8_t *rgba, uint32_t w, uint32_t h, uint32_t stride)
{
   uint64_t c = 0;
   for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++) c += rgba[(size_t)y * stride + x * 4 + 3] >= 128;
   return (double)c / ((double)w * h);
}

float rt_half_to_float(uint16_t h)
{
   uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
   if (e == 0) {
      if (!m) f = s << 31;
      else {
         e = 127 - 15 + 1;
         while (!(m & 1024)) { m <<= 1; e--; }
         m &= 1023;
         f = (s << 31) | (e << 23) | (m << 13);
      }
   } else if (e == 31) f = (s << 31) | 0x7f800000 | (m << 13);
   else f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
   float r;
   memcpy(&r, &f, 4);
   return r;
}

void rt_thermal_snapshot(char *buf, size_t n)
{
   size_t l = 0;
   l += snprintf(buf + l, n - l, "{");
#ifdef __linux__
   int first = 1;
   DIR *d = opendir("/sys/class/thermal");
   if (d) {
      struct dirent *e;
      while ((e = readdir(d)) && l + 128 < n) {
         if (strncmp(e->d_name, "thermal_zone", 12)) continue;
         char p[300], type[64] = "", t[32] = "";
         snprintf(p, sizeof(p), "/sys/class/thermal/%s/type", e->d_name);
         FILE *f = fopen(p, "r");
         if (f) { if (!fgets(type, sizeof(type), f)) type[0] = 0; fclose(f); }
         snprintf(p, sizeof(p), "/sys/class/thermal/%s/temp", e->d_name);
         f = fopen(p, "r");
         if (f) { if (!fgets(t, sizeof(t), f)) t[0] = 0; fclose(f); }
         type[strcspn(type, "\n")] = 0;
         long mc = atol(t);
         if (!type[0] || !t[0]) continue;
         l += snprintf(buf + l, n - l, "%s\"%s\":%.1f", first ? "" : ",", type, mc / 1000.0);
         first = 0;
      }
      closedir(d);
   }
#endif
   snprintf(buf + l, n - l, "}");
}
