#ifndef LIDAR_GENERAL_EXT_H_
#define LIDAR_GENERAL_EXT_H_

#include "rayz_lidar_general.h"

struct RayzPointRawSingle {
  unsigned int ts_ns;      // relative timestamp (to ts_ms) in ns
  unsigned short encoder;  // 0-32000
  unsigned char laser;     // 0-255
  unsigned char mirror;    // 0-255
  unsigned char flags;     // x, x ,single/double threshold, is_roi
  short t0[3];             // 20ps 0-600m ~9mm  1ps=0.15mm
  unsigned char w0[3];     // 20ps 3cm 0-256ns 1ns
};

struct RayzPointRawDouble {
  unsigned int ts_ns;      // relative timestamp (to ts_ms) in ns
  unsigned short encoder;  // 0-32000
  unsigned char laser;     // 0-255
  unsigned char mirror;    // 0-255
  unsigned char flags;     // x, x ,single/double threshold, is_roi
  short t0[3];             // 20ps 0-600m ~9mm  1ps=0.15mm
  unsigned char w0[3];     // 20ps 3cm 0-256ns 1ns
  short t1[3];
  unsigned char w1[3];
};

#endif  // LIDAR_GENERAL_EXT_H_