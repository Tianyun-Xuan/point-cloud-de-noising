#include <../inc/rayz_lidar_sdk.h>
#include <../include/inference.h>

#include <chrono>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

RayzLidarPacket* fresh_frame = nullptr;

struct StartupConfig {
  std::string engine_file;
  std::string lidar_model;
  std::string input_channel;
  std::string output_channel;
  std::string output_type;
  bool debug;
  int protocol;
};

class MistNode : public rclcpp::Node {
 public:
  MistNode()
      : Node("mistnode",
             rclcpp::NodeOptions()
                 .allow_undeclared_parameters(true)
                 .automatically_declare_parameters_from_overrides(true)
                 .use_intra_process_comms(true)) {
    // load parameters
    this->get_parameter_or<std::string>("engine_file", config_.engine_file,
                                        "/home/rayz/code/engine.trt");
    this->get_parameter_or<std::string>("lidar_model", config_.lidar_model,
                                        "m2w");
    this->get_parameter_or<std::string>("input_channel", config_.input_channel,
                                        "udp://0.0.0.0:2368");
    this->get_parameter_or<std::string>(
        "output_channel", config_.output_channel, "ws://0.0.0.0:12369");
    this->get_parameter_or<std::string>("output_type", config_.output_type,
                                        "fresh");
    this->get_parameter_or<bool>("debug", config_.debug, false);
    this->get_parameter_or<int>("protocol", config_.protocol, 6);

    RCLCPP_INFO(this->get_logger(), "engine_file: %s",
                config_.engine_file.c_str());
    RCLCPP_INFO(this->get_logger(), "lidar_model: %s",
                config_.lidar_model.c_str());
    RCLCPP_INFO(this->get_logger(), "input_channel: %s",
                config_.input_channel.c_str());
    RCLCPP_INFO(this->get_logger(), "output_channel: %s",
                config_.output_channel.c_str());
    RCLCPP_INFO(this->get_logger(), "output_type: %s",
                config_.output_type.c_str());
    RCLCPP_INFO(this->get_logger(), "debug: %d", config_.debug);
    RCLCPP_INFO(this->get_logger(), "protocol: %d", config_.protocol);

    // init allocate fresh frame
    fresh_frame = (RayzLidarPacket*)calloc(
        1, sizeof(RayzCPoint) * 1 * 128 * 1200 + sizeof(RayzLidarPacket));

    // load model
    engine_ = std::make_shared<Inference>(config_.engine_file);
    engine_->printModelInfo();

    // rayz config
    if (config_.debug) rayz_lidar_set_log_level("debug");
    int lidar_handle = rayz_lidar_open(config_.input_channel.c_str(),
                                       config_.lidar_model.c_str());

    if (lidar_handle >= 0) {
      // rayz_lidar_set_config(lidar_handle, "rewind", "-1", (char*)"int");

      if (config_.output_type == "fresh") {
        rayz_lidar_set_callback(lidar_handle, data_callback_fresh_wrapper,
                                this);
      } else if (config_.output_type == "save") {
        rayz_lidar_set_callback(lidar_handle, data_callback_save_wrapper, this);
      } else if (config_.output_type == "mark") {
        rayz_lidar_set_callback(lidar_handle, data_callback_mark_wrapper, this);
      } else {
        rayz_lidar_set_callback(lidar_handle, data_callback_tradition_wrapper,
                                this);
      }
    }

    rayz_lidar_start(lidar_handle);
    rayz_lidar_add_stream(-1, config_.output_channel.c_str(), nullptr, nullptr);
  }

 private:
  // input paramters
  StartupConfig config_;
  std::shared_ptr<Inference> engine_ = nullptr;
  // 4 input channel * 128 height * 1200 width
  // distance pluse distance pluse
  std::vector<float> input_ = std::vector<float>(1 * 3 * 128 * 1200, 0.f);
  std::vector<float> output_ = std::vector<float>(1 * 3 * 128 * 1200, 0.f);
  std::vector<int> result_ = std::vector<int>(1 * 128 * 1200, 0);

  // tradition
  bool init_flag = false;
  std::vector<float> pre_frame = std::vector<float>(1 * 3 * 128 * 1200, 0.f);

 private:
  // create a fresh new frame, and only keep the useful points
  static int data_callback_fresh_wrapper(int handle, RayzLidarPacket* frame,
                                         void* context) {
    return (reinterpret_cast<MistNode*>(context))
        ->data_callback_fresh(handle, frame, context);
  }

  // keep the old frame, and mark the mist points
  static int data_callback_mark_wrapper(int handle, RayzLidarPacket* frame,
                                        void* context) {
    return (reinterpret_cast<MistNode*>(context))
        ->data_callback_mark(handle, frame, context);
  }

  // save the txt file for each frame
  static int data_callback_save_wrapper(int handle, RayzLidarPacket* frame,
                                        void* context) {
    return (reinterpret_cast<MistNode*>(context))
        ->data_callback_save(handle, frame, context);
  }

  // save the txt file for each frame
  static int data_callback_tradition_wrapper(int handle, RayzLidarPacket* frame,
                                             void* context) {
    return (reinterpret_cast<MistNode*>(context))
        ->data_callback_tradition(handle, frame, context);
  }

  int data_callback_save(int handle, RayzLidarPacket* frame, void* context) {
    if (frame->type == 0) {
      RayzCPoint* points = (RayzCPoint*)frame->content;

      // v 0.0.6
      if (this->config_.protocol == 6 && frame->number == 304200) {
        int number = frame->number / 2;
        std::fstream txtout;
        txtout.open("savetxt/" + std::to_string(frame->seq) + ".txt",
                    std::ios::out);

        // CPoint -> depth-map
        for (int i = 0; i < number; i++) {
          for (int j = 0; j < 2; j++) {
            int index = i * 2 + j;
            RayzCPoint point = points[index];
            if (short(point.range) != 0) {
              // vangle hangle row col range pluse index echo
              double azimuth = (short)point.h_angle * 0.016 * M_PI_ / 180.0;
              double inclination = (short)point.v_angle / 128.0 * M_PI_ / 180.0;
              double radius = short(point.range) * kRangeResolution;

              double t = radius * cos(inclination);
              double x = t * sin(azimuth);
              double y = t * cos(azimuth);
              double z = radius * sin(inclination);

              txtout << x << " " << y << " " << z << " " << short(point.vline)
                     << " " << short(point.ts_10usec) << " "
                     << short(point.range) << " " << short(point.pluse) << " "
                     << j << std::endl;
            }
          }
        }
      }
    }
    return 0;
  }

  int data_callback_fresh(int handle, RayzLidarPacket* frame, void* context) {
    // only process frame type 0 point cloud
    if (frame->type == 0) {
      std::cout << "Frame type: " << frame->type << std::endl;
      std::cout << "Frame number: " << frame->number << std::endl;
      // read RayzCPoint in frame
      RayzCPoint* points = (RayzCPoint*)frame->content;

      std::fill(input_.begin(), input_.end(), 0.0f);
      std::fill(output_.begin(), output_.end(), 0.0f);
      std::fill(result_.begin(), result_.end(), 0);

      // echo
      if (frame->number == 304200) {
        const int number = 153600;

        // CPoint -> depth-map
        for (int i = 0; i < number; i++) {
          int index = i * 2;
          RayzCPoint& point = points[index];
          if (point.range > 0) {
            // vangle hangle row col range pluse index echo
            double azimuth = (short)point.h_angle * 0.016 * M_PI_ / 180.0;
            double inclination = (short)point.v_angle / 128.0 * M_PI_ / 180.0;
            double radius = short(point.range) * kRangeResolution;

            double t = radius * cos(inclination);
            double x = t * sin(azimuth);
            double y = t * cos(azimuth);
            double z = radius * sin(inclination);

            int x_index = short(point.vline) * 1200 + short(point.ts_10usec);

            input_[x_index] = x;
            input_[x_index + 128 * 1200] = y;
            input_[x_index + 128 * 1200 * 2] = z;
          }
        }

        // infer
        if (engine_->infer(input_, output_) != 0) {
          std::cerr << "Inference failed" << std::endl;
          return -1;
        }

        // argmax
        argmax(output_, result_);

        int point_count = 0;
        // recreate fresh frame
        if (fresh_frame == nullptr) {
          fresh_frame = (RayzLidarPacket*)calloc(
              1, sizeof(RayzCPoint) * 2 * 128 * 1200 + sizeof(RayzLidarPacket));
        }
        memcpy(fresh_frame, frame, sizeof(RayzLidarPacket));
        for (int i = 0; i < number; i++) {
          for (int j = 0; j < 2; j++) {
            int index = i * 2 + j;
            RayzCPoint& point = points[index];

            if (point.range != 0) {
              int pixel_index =
                  short(point.vline) * 1200 + short(point.ts_10usec);
              int flag = j == 0 ? 1 : 2;

              if (result_[pixel_index] == 0) {
                memcpy(&fresh_frame->point_sc[point_count], &point,
                       sizeof(RayzCPoint));
                point_count++;
              }
            }
          }
        }

        std::cout << "Useful points: " << point_count << "/" << frame->number
                  << std::endl;

        fresh_frame->number = point_count;
        fresh_frame->length = point_count * sizeof(RayzCPoint);

        // publish
        rayz_lidar_pub_packet(handle, fresh_frame);
        return 1;
      } else {
        // log error
        std::cerr << "Invalid frame number: " << frame->number << std::endl;
        return -1;
      }
    } else {
      // publish
      rayz_lidar_pub_packet(handle, frame);
      return 1;
    }

    return -1;
  }

  int data_callback_mark(int handle, RayzLidarPacket* frame, void* context) {
    // only process frame type 0 point cloud
    if (frame->type == 0) {
      // read RayzCPoint in frame
      RayzCPoint* points = (RayzCPoint*)frame->content;

      std::fill(input_.begin(), input_.end(), 0.0f);
      std::fill(output_.begin(), output_.end(), 0.0f);
      std::fill(result_.begin(), result_.end(), 0);

      // echo
      if (frame->number == 304200) {
        const int number = frame->number / 2;

        // CPoint -> depth-map
        for (int i = 0; i < number; i++) {
          int index = i * 2;
          RayzCPoint& point = points[index];
          if (point.range > 0) {
            // vangle hangle row col range pluse index echo
            double azimuth = (short)point.h_angle * 0.016 * M_PI_ / 180.0;
            double inclination = (short)point.v_angle / 128.0 * M_PI_ / 180.0;
            double radius = short(point.range) * kRangeResolution;

            double t = radius * cos(inclination);
            double x = t * sin(azimuth);
            double y = t * cos(azimuth);
            double z = radius * sin(inclination);

            int x_index = short(point.vline) * 1200 + short(point.ts_10usec);

            input_[x_index] = x;
            input_[x_index + 128 * 1200] = y;
            input_[x_index + 128 * 1200 * 2] = z;
          }
        }

        // infer
        if (engine_->infer(input_, output_) != 0) {
          std::cerr << "Inference failed" << std::endl;
          return -1;
        }

        // argmax
        argmax(output_, result_);

        int point_count = 0;
        // recreate fresh frame
        if (fresh_frame == nullptr) {
          fresh_frame = (RayzLidarPacket*)calloc(
              1, sizeof(RayzCPoint) * 2 * 128 * 1200 + sizeof(RayzLidarPacket));
        }
        memcpy(fresh_frame, frame, sizeof(RayzLidarPacket));
        for (int i = 0; i < number; i++) {
          for (int j = 0; j < 2; j++) {
            int index = i * 2 + j;
            RayzCPoint& point = points[index];

            // only keep points with range >0
            if (point.range > 0) {
              if (j == 0) {
                int pixel_index =
                    short(point.vline) * 1200 + short(point.ts_10usec);

                if ((result_[pixel_index]) == 1) {
                  point.intensity = 255;
                } else {
                  point.intensity = 100;
                }
              } else {
                point.intensity = 180;
              }

              memcpy(&fresh_frame->point_sc[point_count], &point,
                     sizeof(RayzCPoint));
              point_count++;
            }
          }
        }

        std::cout << "Totally points: " << point_count << "/" << frame->number
                  << std::endl;

        fresh_frame->number = point_count;
        fresh_frame->length = point_count * sizeof(RayzCPoint);

        // publish
        rayz_lidar_pub_packet(handle, fresh_frame);
        return 1;
      } else {
        // log error
        std::cerr << "Invalid frame number: " << frame->number << std::endl;
        return -1;
      }
    } else {
      // publish
      rayz_lidar_pub_packet(handle, frame);
      return 1;
    }

    return -1;
  }

  int data_callback_tradition(int handle, RayzLidarPacket* frame,
                              void* context) {
    // only process frame type 0 point cloud
    if (frame->type == 0) {
      std::cout << "Frame type: " << frame->type << std::endl;
      std::cout << "Frame number: " << frame->number << std::endl;
      // read RayzCPoint in frame
      RayzCPoint* points = (RayzCPoint*)frame->content;

      std::fill(input_.begin(), input_.end(), 0.0f);
      std::fill(output_.begin(), output_.end(), 0.0f);
      std::fill(result_.begin(), result_.end(), 0);

      // echo
      if (frame->number == 304200) {
        const int number = 153600;

        // CPoint -> depth-map
        for (int i = 0; i < number; i++) {
          int index = i * 2;
          RayzCPoint& point = points[index];
          if (point.range > 0) {
            // // vangle hangle row col range pluse index echo
            // double azimuth = (short)point.h_angle * 0.016 * M_PI_ / 180.0;
            // double inclination = (short)point.v_angle / 128.0 * M_PI_ /
            // 180.0; double radius = short(point.range) * kRangeResolution;

            // double t = radius * cos(inclination);
            // double x = t * sin(azimuth);
            // double y = t * cos(azimuth);
            // double z = radius * sin(inclination);

            int x_index = short(point.vline) * 1200 + short(point.ts_10usec);

            input_[x_index] = short(point.range);
            input_[x_index + 128 * 1200] = short(point.pluse);
            // input_[x_index + 128 * 1200 * 2] = z;
          }
        }

        // tradition
        if (!init_flag) {
          pre_frame = input_;
          init_flag = true;
        } else {
          // calculate the difference and mark the mist points
          double mean = 0.0;
          double std = 0.0;
          for (int i = 0; i < 128 * 1200; i++) {
            auto diff_range = input_[i] - pre_frame[i];
            auto diff_pluse = input_[i + 128 * 1200] - pre_frame[i + 128 * 1200];

            auto diff_distance = std::fabs(diff_range) * kRangeResolution;
            output_[i] = diff_distance;
            mean += diff_distance;
          }

          mean /= 128 * 1200;
          for (int i = 0; i < 128 * 1200; i++) {
            std += (output_[i] - mean) * (output_[i] - mean);
          }
          std = std::sqrt(std / (128 * 1200));

          for (int i = 0; i < 128 * 1200; i++) {
            if (output_[i] > mean + 3 * std) {
              result_[i] = 1;
            }
          }
        }

        int point_count = 0;
        // recreate fresh frame
        if (fresh_frame == nullptr) {
          fresh_frame = (RayzLidarPacket*)calloc(
              1, sizeof(RayzCPoint) * 2 * 128 * 1200 + sizeof(RayzLidarPacket));
        }
        memcpy(fresh_frame, frame, sizeof(RayzLidarPacket));
        for (int i = 0; i < number; i++) {
          for (int j = 0; j < 2; j++) {
            int index = i * 2 + j;
            RayzCPoint& point = points[index];

            // only keep points with range >0
            if (point.range > 0) {
              if (j == 0) {
                int pixel_index =
                    short(point.vline) * 1200 + short(point.ts_10usec);

                if ((result_[pixel_index]) == 1) {
                  point.intensity = 255;
                } else {
                  point.intensity = 100;
                }
              } else {
                point.intensity = 180;
              }

              memcpy(&fresh_frame->point_sc[point_count], &point,
                     sizeof(RayzCPoint));
              point_count++;
            }
          }
        }

        std::cout << "Useful points: " << point_count << "/" << frame->number
                  << std::endl;

        fresh_frame->number = point_count;
        fresh_frame->length = point_count * sizeof(RayzCPoint);

        // publish
        rayz_lidar_pub_packet(handle, fresh_frame);
        return 1;
      } else {
        // log error
        std::cerr << "Invalid frame number: " << frame->number << std::endl;
        return -1;
      }
    } else {
      // publish
      rayz_lidar_pub_packet(handle, frame);
      return 1;
    }

    return -1;
  }
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MistNode>());
  rclcpp::shutdown();
  return 0;
}
