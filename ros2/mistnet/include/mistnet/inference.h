#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <vector>

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << msg << std::endl;
    }
  }
};

class Inference {
 public:
  // module for loading engine and running inference

  Inference(const std::string& engineFile);

  ~Inference();

  int infer(const std::vector<float>& input, std::vector<float>& output);

  // print debug info of the model
  void printModelInfo();

 private:
  std::vector<char> loadEngine(const std::string& engineFile);

  void cudaCheck(cudaError_t status) {
    if (status != 0) {
      std::cerr << "CUDA failure: " << status << std::endl;
      exit(-1);
    }
  }

  std::vector<char> engineData;
  Logger gLogger;
  nvinfer1::IRuntime* runtime;
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;
  int inputIndex;
  int outputIndex;
  int batchSize;
  int inputSize;
  int outputSize;
  void* buffers[2];
  cudaStream_t stream;
};

void local_test(const std::string& engineFile, const std::string& val_dir);
int load_txt_input(const std::string& input_file, std::vector<float>& input,
                   std::vector<float>& label);

void argmax(const std::vector<float>& output, std::vector<int>& result);
float accuracy(const std::vector<int>& output,
               const std::vector<int>& label);