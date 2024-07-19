#ifndef RAYZ_LIDAR_GENERAL_H_
#define RAYZ_LIDAR_GENERAL_H_
#pragma once

#ifdef _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport __attribute__((visibility("default")))
#endif

#ifdef __GNUC__
#define RAYZ_PACKED __attribute__((__packed__))
#else
#define RAYZ_PACKED __pragma(pack(push, 1)) __pragma(pack(pop))
#endif

#define RAYZ_SDK_VERSION "unknown_2407180939"

#define M_PI_ 3.14159265358979323846264338327950288
static const double kRangeResolution = 0.008;         // meter
static const double kAngleResolution = M_PI_ / 4096;  // rad
static const double kAngleOffsetHorizontal = 0;       // rad
static const double kAngleOffsetVertical = 0;         // rad

static const double kTimeResolution = 0.000000000020;  // s
static const double kLightSpeed = 299792458.0;

/*
0.01 degree = 0.000174532925199 rad
1/8192 rad = 0.006994113710093 degree
1/4096 rad = 0.013988227420186 degree
*/

typedef enum {
  kLidarInputTypeMemory = 0,
  kLidarInputTypeNetwork = 1,
  kLidarInputTypeFile = 2,
  kLidarInputTypeUnknown = 3
} LidarInputType;

typedef enum {
  kLidarStateNormal = 0,       // Normal work state
  kLidarStatePowerSaving = 1,  // Power-saving state
  kLidarStateStandBy = 2,      // Standby state
  kLidarStateInit = 3,         // Initialization state
  kLidarStateError = 4,        // Error state
  kLidarStateUnknown = 5       // Unknown state
} LidarState;

typedef enum {
  kLidarModeNormal = 0,      /**< Normal mode. */
  kLidarModePowerSaving = 1, /**< Power-saving mode. */
  kLidarModeStandby = 2      /**< Standby mode. */
} LidarMode;

typedef enum {
  TimeSyncNoSync = 0, /**< No sync signal mode. */
  TimeSyncPtp = 1,    /**< 1588v2.0 PTP sync mode. */
  TimeSyncNtp = 2,    /**< Reserved use. */
  TimeSyncPpsGps = 3, /**< pps+gps sync mode. */
  TimeSyncPps = 4,    /**< pps only sync mode. */
  TimeSyncUnknown = 5 /**< Unknown mode. */
} TimeSync;

typedef enum {
  RAYZ_RETURN_MODE_FIRST = 0,
  RAYZ_RETURN_MODE_STRONGEST = 1,
  RAYZ_RETURN_MODE_DUAL = 2,
  RAYZ_RETURN_MODE_TRIPLE = 3,
} RayzReturnMode;

// 4 bytes
typedef struct RAYZ_PACKED {
  unsigned short range : 16;
  union {
    struct {
      unsigned char ref : 8;   // reflectance or intensity
      unsigned char flag : 8;  // bit[3:0] - semantic, bit[7:4] - quality
    } attr;                    // attribute
    struct {
      unsigned char high : 8;  // high pluse width
      unsigned char low : 8;   // low pluse width
    } thr;                     // threshold
  };
  // ext: 4 bytes in ddr for compatibility
  // short h_angle;
  // short v_angle;
} RayzBlock;

// 12 bytes
typedef struct RAYZ_PACKED {
  unsigned short
      h_angle;  // azimuth, -PI/2 ~ PI/2 rad, unit is kRadAngleUnit rad
  unsigned short
      v_angle;  // inclination, -PI/2 ~ PI/2 rad, unit is kRadAngleUnit rad
  unsigned short range;      // unit is 8mm
  unsigned short ts_10usec;  // relative timestamp (to ts_usec) in 10usec
  unsigned char intensity;   // or reflectance;
  unsigned char vline;
  unsigned char pluse;
  unsigned char reserve;
} RayzCPoint;

// 16 bytes
typedef struct RAYZ_PACKED {
  float x;                   // front
  float y;                   // left
  float z;                   // top
  unsigned short ts_10usec;  // relative timestamp (to ts_usec) in 10usec
  unsigned char intensity;   // or reflectance;
  unsigned char reserve;
} RayzPoint;

// 24 bytes
typedef struct RAYZ_PACKED {
  float x;
  float y;
  float z;
  short h_angle;
  short v_angle;
  unsigned short range;      // 8mm
  unsigned short ts_10usec;  // relative timestamp (to ts_usec) in 10usec
  unsigned char intensity;   // or reflectance;
  unsigned char vline;
  unsigned char pluse;
} RayzPointHyb;

// 1019 bytes
typedef struct RAYZ_PACKED {
  int h_angle : 12;  // azimuth, -PI/2 ~ PI/2 rad, unit is kRadAngleUnit rad
  int v_angle : 12;  // inclination, -PI/2 ~ PI/2 rad, unit is kRadAngleUnit rad
  unsigned int range : 16;       // unit is 8mm
  unsigned char intensity : 8;   // or reflectance;
  unsigned char ts_100usec : 8;  // relative timestamp (to ts_usec) in 10usec
  float x;                       // front
  float y;                       // left
  float z;                       // top
  unsigned short hist[500];      // histogram
} RayzPointHist;

// 28 bytess
typedef struct RAYZ_PACKED {
  unsigned int id;
  unsigned char
      type;  // 0:unknown, 1:car, 2:pedestrian, 3:bicycle, 4:truck, 5:bus
  unsigned char confidence;   // 0~255, 0:unknown, 255:high
  unsigned short ts_100usec;  // relative timestamp (to ts_usec) in 100usec
  unsigned short center[3];   // x, y, z, unit 0.01m
  unsigned short size[3];     // length, width, height, unit 0.01m
  short speed;                // speed, unit 0.01m/s
  short euler[3];             // roll, pitch, yaw, unit 0.01 degree
} RayzObject3D;

// 16 bytes
typedef struct RAYZ_PACKED {
  unsigned short accel_ratio;
  short accel_x;  // accel_x/accel_ratio, unit g
  short accel_y;
  short accel_z;
  unsigned short gyro_ratio;
  short gyro_x;  // gyro_x/gyro_ratio/10, unit degree/s
  short gyro_y;
  short gyro_z;
} RayzLidarImu;

typedef struct RAYZ_PACKED {
  unsigned char lidar_mode;   // 0:normal, 1:power saving, 2:standby
  unsigned char lidar_state;  // 0:normal, 1:over temperature, 2:over voltage,
                              // 3:over current, 4:over load 5:over speed,
                              // 6:over frequency, 7:over distance,
                              // 8:over power, 9:over temperature
  unsigned char time_sync_mode;   // 0:no sync, 1:ptp, 2:ntp, 3:pps+gps,
                                  // 4:pps, 5:unknown
  unsigned char time_sync_state;  // 0:unknown, 1:sync, 2:unsync
  unsigned int rpm;
  // 8
  short temperature[4];
  // 16
  unsigned short frame_rate;
  unsigned char lidar_model[2];
  unsigned char lidar_ip[4];
  unsigned short lidar_port;
  unsigned char lidar_mac[6];
  // 32
  unsigned char firmware_version[4];
  unsigned char hardware_version[4];
  // 40
  unsigned char firmware_time[8];
  unsigned char hardware_time[8];
  // 56
  unsigned short accel_ratio;
  short accel_x;  // accel_x/accel_ratio, unit g
  short accel_y;
  short accel_z;
  // 64
  unsigned short gyro_ratio;
  short gyro_x;  // gyro_x/gyro_ratio/10, unit degree/s
  short gyro_y;
  short gyro_z;
  // 72
  unsigned int life_sec;
  unsigned int live_sec;
  unsigned int work_sec;
} RayzLidarStatus;

/*
get_hw_mode
get_sw_mode
get_lidar_status
get_lidar_config
*/
typedef struct RAYZ_PACKED {
  int code;      // 0~10: message with level, 11~100: Request, 101~: Response
                 // 200:success, 404:not found
                 // 400:bad request, 405:method not allowed,
                 // 500:internal error
  char key[64];  // "" means just message
                 // "get_lidar_status" , length should be 0
                 // "set_lidar_onoff" , value can be "on" or "off"
                 // "extend-id", value is key, id should be unique
  char value[0];
} RayzMessage;

// 32 bytes
typedef struct RAYZ_PACKED {
  unsigned short sob;     // set of books, crc16/modbus of "rayz": 0xCE 0xE8
  unsigned char id;       // source id
  unsigned char version;  // [7:4]-frame version, 0-10Hz, 1-20Hz
                          // [3:0]-protocol version, 4|5|6
  unsigned short seq;     // always starting from 1 on boot(for sensor data)
  unsigned short frag;    // fragment, always starting from 1, 0 means end
  double ts_usec;         // timestamp in microsecond, UTC.xxx_xxx, unit in usec
  union {
    unsigned int reserve;  // reserve for native use
    struct {
      unsigned short hline;  // horizontal block line, max 65536>>(flag[7:6]*2)
      unsigned char vline;   // vertical block line, max 256<<(flag[7:6]*2)
      unsigned char flag;
      // bit[2:0]: single return/dual return/triple return
      //           first[0], strongest[1]
      //           farthest[2], first-strongest[3],
      //           first-second[4], first-farthest[5]
      //           strongest-farthest[6], triple[7]
      // bit[3]: 0-single pluse, 1-big-small pluse
      // bit[5:4]: 0-ref mode, 1-ref_calib mode
      // bit[7:6]: reserve for split hline/vline bits
    } ext;  // should only work with RayzBlock
  };
  unsigned char topic;
  unsigned char type;
  /*
              sensor       data   | perception|  message  |  status  |
      type  |  0x00    |   0x10   | 0x30-0x3f | 0x40-0x4f | 0x50-0x5f|
      topic |  0x00    |   ....   |    0x30   |   0x40    |    0x50
       0x00 |  lidar   |   IMU    |  3D-box   |  default  |  default
       0x10 |  H260    |                                  |
       0x20 |  W100    |
       ...  |   ...
  */
  unsigned short unit;  // unit * number = length
  unsigned int number;  // unit * number = length
  unsigned int length;  // unit * number = length
#ifndef _MSC_VER
  union {
    unsigned char content[0];
    RayzCPoint point_sc[0];
    RayzPoint point_cc[0];
    RayzPointHist point_hist[0];
    RayzObject3D object3d[0];
    RayzLidarStatus status[0];
    RayzLidarImu imu[0];
    RayzMessage message[0];
    RayzBlock block[0];
  };
#endif
} RayzLidarPacket;

#endif  // RAYZ_LIDAR_GENERAL_H_