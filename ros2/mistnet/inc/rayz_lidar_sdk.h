#ifndef RAYZ_LIDAR_SDK_H_
#define RAYZ_LIDAR_SDK_H_

#ifdef RAYZ_SDK_INTERNAL
#include "rayz_lidar_general_intl.h"
#endif

#include "rayz_lidar_general.h"

extern "C" {

/*
 * @breif: get lidar version
 * @param: version: version string
 * @param: size: version string size
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_get_version(char *version, int size);

/*
 * @breif: create lidar driver
 * @param: model: lidar model, default "auto", candicate "auto", "legacy",
 * "osprey", "eagle"
 * @return: lidar handle
 */
DllExport int rayz_lidar_open(const char *url, const char *model);

/*
 * @breif: close lidar driver
 * @param: handle: lidar handle
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_close(int handle);

/*
 * @breif: start lidar driver
 * @param: handle: lidar handle
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_start(int handle);

/*
 * @breif: stop lidar driver
 * @param: handle: lidar handle
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_stop(int handle);

/*
 * @breif: set lidar config
 * @param: handle: lidar handle
 * @param: key: config key
 * @param: value: config value
 * @param: format: config format, such as "string", "json", "file"
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_set_config(int handle, const char *key,
                                    const char *value,
                                    char *format = (char *)"string");

/*
 * @breif: get lidar config
 * @param: handle: lidar handle
 * @param: key: config key
 * @param: value: config value
 * @param: size: config value max size
 * @param: format: config format, such as "string", "json", "file"
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_get_config(int handle, const char *key, char *value,
                                    int size, char *format = (char *)"string");

/*
 * @breif: get lidar type
 * @param: handle: lidar handle
 * @param: type: type string
 * @param: size: type string size
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_get_type(int handle, char *type, int size);

/*
 * @breif: get lidar serial number
 * @param: handle: lidar handle
 * @param: sn: serial number string
 * @param: size: serial number string size
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_get_sn(int handle, char *sn, int size);

/*
 * @breif: get data packet, should be called after rayz_lidar_start
 * @param: handle: lidar handle
 * @param: block_us(0): -1: forever; 0: noblock; >0: block us
 * @return: RayzLidarPacket *
 */
DllExport RayzLidarPacket *rayz_lidar_get_packet(int handle, int block_us);

typedef int (*RayzPacketCb)(int handle, RayzLidarPacket *packet, void *context);
/*
 * @breif: set frame packet callback
 * @param: handle: lidar handle
 * @param: callback: callback function
 * @param: context: callback context
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_set_callback(int handle, RayzPacketCb callback,
                                      void *context);

/*
 * @breif: set log callback, default log level is "info", should be called
 * before rayz_lidar_set_log_level if you want to change log level
 * @param: callback: log callback function
 * @param: level: 0 - 6, trace - off
 * @param: context: log callback context
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_set_log_callback(RayzPacketCb log_callback, int level,
                                          void *context);

/*
 * @breif: set log level
 * @param: level: log level, such as "trace", "debug", "info", "warn",
 * "error"
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_set_log_level(const char *level);

/*
 * @breif: set log file, default log level is "info", should be called before
 * rayz_lidar_set_log_level if you want to change log level
 * @param: path: log file path
 * @param: max_size: max log file size, default 1G
 * @param: max_files: max log file number, default 5
 * @param: daily: whether to split log file daily, default 0
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_set_log_file(const char *path, int max_size,
                                      int max_files, int daily);

/*
 * @breif: start record
 * @param: handle: lidar handle
 * @param: path: record file name, default ""
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_start_record(int handle, const char *path);

/*
 * @breif: pause/resume record
 * @param: handle: lidar handle
 * @param: pause: 1: pause, 0: resume
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_pause_record(int handle, int pause);

/*
 * @breif: stop record
 * @param: handle: lidar handle
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_stop_record(int handle);

typedef int (*RayzStreamCb)(int handle, char *data, int len, void *context);
/*
 * @breif: set frame stream out, call after rayz_lidar_start
 * @param: handle: lidar handle, -1 means all
 * @param: url: stream out url, [udp|tcp|ws]://ip:port
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_add_stream(int handle, const char *url,
                                    RayzStreamCb stream_callback = nullptr,
                                    void *context = nullptr);

/*
 * @breif: publish packet to stream out
 * @param: handle: lidar handle
 * @param: packet: packet to publish
 * @return: 0: success, -1: failed
 */
DllExport int rayz_lidar_pub_packet(int handle, RayzLidarPacket *packet);
}
#endif  // RAYZ_LIDAR_SDK_H_