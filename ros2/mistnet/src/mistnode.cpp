#include <../include/mistnet/inference.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class MistNode : public rclcpp::Node {
 public:
  MistNode() : Node("mist_node") {
    // Create a publisher on the "mistnet" topic
    publisher_ = this->create_publisher<std_msgs::msg::String>("mistnet", 10);

    // Create a timer that calls the publishMessage function every 500ms
    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(500),
                                std::bind(&MistNode::publishMessage, this));
  }

 private:
  void publishMessage() {
    // Create a message to send
    auto message = std_msgs::msg::String();
    message.data = "Hello, world!";

    // Publish the message
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MistNode>());
  rclcpp::shutdown();
  return 0;
}
