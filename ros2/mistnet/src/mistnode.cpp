#include <../inc/rayz_lidar_sdk.h>
#include <../include/mistnet/inference.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

RayzLidarPacket* fresh_frame;

class MistNode : public rclcpp::Node {
 private:
  static int data_callback_wrapper(int handle, RayzLidarPacket* frame,
                                   void* context) {
    return (reinterpret_cast<MistNode*>(context))
        ->data_callback(handle, frame, context);
  }

  int data_callback(int handle, RayzLidarPacket* frame, void* context) {
    // only process frame type 0 point cloud
    if (frame->type == 0) {
      // read RayzCPoint in frame
      RayzCPoint* points = (RayzCPoint*)frame->content;

      std::fill(input_.begin(), input_.end(), 0.0f);
      std::fill(output_.begin(), output_.end(), 0.0f);
      std::fill(result_.begin(), result_.end(), 0);

      // echo
      if (frame->number % 3 == 0) {
        int number = frame->number / 3;

        // CPoint -> depth-map
        for (int i = 0; i < number; i++) {
          // get first two echos
          for (int j = 0; j < 2; j++) {
            int index = i * 3 + j;
            RayzCPoint point = points[index];
            if (point.range != 0) {
              int pixel_index =
                  short(point.vline) * 1200 + short(point.reserve);
              int distance_id = 2 * j * 128 * 1200 + pixel_index;
              int pluse_id = (2 * j + 1) * 128 * 1200 + pixel_index;

              input_[distance_id] = short(point.range);
              input_[pluse_id] = short(point.pluse);
            }
          }
        }

        // infer
        if (engine_.infer(input_, output_) != 0) {
          std::cerr << "Inference failed" << std::endl;
          return -1;
        }

        // argmax
        argmax(output_, result_);

        // two bits are used to code the echo
        // 00: no echo
        // 01: first echo
        // 10: second echo
        // 11: both echo

        // // remove noise
        // std::vector<RayzCPoint> filtered_points;
        // for (int i = 0; i < useful_points.size(); i++) {
        //   auto& point = useful_points[i];
        //   int echo = useful_echo[i];
        //   int row = short(point.vline);
        //   int col = short(point.reserve);

        //   if (result[row * 1200 + col] & echo != 0) {
        //     filtered_points.emplace_back(point);
        //   }
        // }

        int point_count = 0;
        // recreate fresh frame
        memcpy(fresh_frame, frame, sizeof(RayzLidarPacket));
        for (int i = 0; i < number; i++) {
          // get first two echos
          for (int j = 0; j < 2; j++) {
            int index = i * 3 + j;
            RayzCPoint point = points[index];
            if (point.range != 0) {
              int pixel_index =
                  short(point.vline) * 1200 + short(point.reserve);
              int flag = 2 ^ (index % 3);
              if (result_[pixel_index] & flag != 0) {
                // memcpy(&fresh_frame->content[point_count], &point,
                //        sizeof(RayzCPoint));
                point_count++;
              }
            }
          }
        }

        fresh_frame->number = point_count;
        fresh_frame->length = point_count * sizeof(RayzCPoint);

        // // publish
        // rayz_lidar_pub_packet(handle, fresh_frame);
        return 1;
      }
    }

    return -1;
  }

 public:
  MistNode() : Node("mistnode") {
    fresh_frame = (RayzLidarPacket*)calloc(
        1, sizeof(RayzCPoint) * 2 * 128 * 1200 + sizeof(RayzLidarPacket));

    // load model
    engine_.printModelInfo();

    // rayz config
    rayz_lidar_set_log_level("debug");
    int lidar_handle = rayz_lidar_open("/home/rayz/code/data/5.pcap", "m2w");

    if (lidar_handle >= 0) {
      rayz_lidar_set_config(lidar_handle, "rewind", "-1", (char*)"int");
      rayz_lidar_set_callback(lidar_handle, data_callback_wrapper, this);
      rayz_lidar_start(lidar_handle);
      rayz_lidar_add_stream(lidar_handle, "ws://192.168.0.3:2368");
    }
  }

 private:
  Inference engine_ = Inference("/home/rayz/code/engine.trt");
  // 4 input channel * 128 height * 1200 width
  // distance pluse distance pluse
  std::vector<float> input_ = std::vector<float>(1 * 4 * 128 * 1200, 0.f);
  std::vector<float> output_ = std::vector<float>(1 * 4 * 128 * 1200, 0.f);
  std::vector<int> result_ = std::vector<int>(1 * 128 * 1200, 0);
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MistNode>());
  rclcpp::shutdown();
  return 0;
}
