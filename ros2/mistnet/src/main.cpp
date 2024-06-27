#include <../include/mistnet/inference.h>

#include <chrono>
#include <filesystem>

void argmax(const std::vector<float> &output, std::vector<int> &result) {
  // 4 * 128 * 1200 argmax dim=0 -> 128 * 1200
  result.resize(128 * 1200);
  std::vector<float> result_float(128 * 1200);

  for (int i = 0; i < 128 * 1200; i++) {
    int max_idx = 0;
    float max_val = output[i];
    for (int j = 1; j < 4; j++) {
      if (output[i + j * 128 * 1200] > max_val) {
        max_val = output[i + j * 128 * 1200];
        max_idx = j;
      }
    }
    result_float[i] = max_idx;
  }

  // convert float to int
  result.assign(result_float.begin(), result_float.end());
}

float accuracy(const std::vector<int> &result, const std::vector<int> &label) {
  int correct = 0;
  for (int i = 0; i < result.size(); i++) {
    if (result[i] == label[i]) {
      correct++;
    }
  }

  return static_cast<float>(correct) / result.size();
}

int load_txt_input(const std::string &input_file, std::vector<float> &input,
                   std::vector<float> &label) {
  // txt files are save as shape (5, 128 *1200)
  // first 4 rows are the input data
  // last row is the label

  input.resize(4 * 128 * 1200);
  label.resize(128 * 1200);

  std::ifstream file(input_file);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << input_file << std::endl;
    return -1;
  }

  for (int i = 0; i < 4 * 128 * 1200; i++) {
    file >> input[i];
  }

  for (int i = 0; i < 128 * 1200; i++) {
    file >> label[i];
  }

  return 0;
}

void local_test(const std::string &engineFile, const std::string &val_dir) {
  // load valfiles
  std::vector<std::string> valfiles;
  for (const auto &entry : std::filesystem::directory_iterator(val_dir)) {
    valfiles.push_back(entry.path());
    std::cout << entry.path() << std::endl;
  }
  std::cout << "Total testing " << valfiles.size() << " files" << std::endl;

  // load engine
  Inference inference(engineFile);
  inference.printModelInfo();

  std::vector<float> input(1 * 4 * 128 * 1200, 1.0f);
  std::vector<float> output(1 * 4 * 128 * 1200);
  std::vector<float> label(1 * 4 * 128 * 1200);
  std::vector<int> result(1 * 128 * 1200);

  const auto &test_size = valfiles.size();

  std::vector<float> list_accuracy(test_size);
  std::vector<int> list_timecost(test_size);

  for (int i = 0; i < test_size; i++) {
    const auto &valfile = valfiles[i];

    if (load_txt_input(valfile, input, label) != 0) continue;

    auto start = std::chrono::high_resolution_clock::now();
    if (inference.infer(input, output) != 0) {
      std::cerr << "Inference failed" << std::endl;
      return;
    }
    argmax(output, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Inference time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms" << std::endl;

    // convert float to int
    std::vector<int> label_int(label.begin(), label.end());
    list_accuracy[i] = accuracy(result, label_int);
    std::cout << "Accuracy: " << list_accuracy[i] << std::endl;
  }
}

// int main(int argc, char **argv) {
//   const std::string engineFile =
//       argc == 3 ? argv[1] : "/home/rayz/code/engine.trt";
//   const std::string val_dir = argc == 3 ? argv[2] :
//   "/home/rayz/code/data/c/";

//   local_test(engineFile, val_dir);

//   return 0;
// }