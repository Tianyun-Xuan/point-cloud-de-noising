#include <../include/mistnet/inference.h>

// Constructor
Inference::Inference(const std::string &engineFile) {
  // Load the TensorRT engine
  engineData = loadEngine(engineFile);

  // Create a runtime
  runtime = nvinfer1::createInferRuntime(gLogger);
  if (!runtime) {
    std::cerr << "Failed to create TensorRT runtime" << std::endl;
    exit(-1);
  }

  // Deserialize the engine
  engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(),
                                          nullptr);
  if (!engine) {
    std::cerr << "Failed to create TensorRT engine" << std::endl;
    exit(-1);
  }

  // Create an execution context
  context = engine->createExecutionContext();
  if (!context) {
    std::cerr << "Failed to create TensorRT execution context" << std::endl;
    exit(-1);
  }

  // Set the input dimensions
  nvinfer1::Dims inputDims = engine->getBindingDimensions(0);
  inputDims.d[0] = 1;  // Batch size
  context->setBindingDimensions(0, inputDims);

  // Allocate memory for inputs and outputs
  inputIndex = engine->getBindingIndex("input");
  outputIndex = engine->getBindingIndex("output");
  inputSize = 1 * 4 * 128 * 1200 * sizeof(float);
  outputSize = 1 * 4 * 128 * 1200 * sizeof(float);
  if (cudaMalloc(&buffers[inputIndex], inputSize) != 0) {
    std::cerr << "Failed to malloc input buffer" << std::endl;
    exit(-1);
  }
  if (cudaMalloc(&buffers[outputIndex], outputSize) != 0) {
    std::cerr << "Failed to malloc output buffer" << std::endl;
    exit(-1);
  }

  // Create CUDA stream
  if (cudaStreamCreate(&stream) != 0) {
    std::cerr << "Failed to create CUDA stream" << std::endl;
    exit(-1);
  }
}

Inference::~Inference() {
  // Clean up
  cudaStreamDestroy(stream);
  cudaFree(buffers[inputIndex]);
  cudaFree(buffers[outputIndex]);
  context->destroy();
  engine->destroy();
  runtime->destroy();
}

// Function to load the TensorRT engine
std::vector<char> Inference::loadEngine(const std::string &engineFile) {
  std::ifstream file(engineFile, std::ios::binary);
  if (!file) {
    std::cerr << "Error opening engine file: " << engineFile << std::endl;
    exit(-1);
  }
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  file.close();
  return buffer;
}

int Inference::infer(const std::vector<float> &input,
                     std::vector<float> &output) {
  // check input & output size
  if (input.size() != 1 * 4 * 128 * 1200) {
    std::cerr << "Input size mismatch" << std::endl;
    return -1;
  }

  if (output.size() != 1 * 4 * 128 * 1200) {
    std::cerr << "Output size mismatch" << std::endl;
    return -1;
  }

  // Copy input data to device
  cudaCheck(cudaMemcpyAsync(buffers[inputIndex], input.data(), inputSize,
                            cudaMemcpyHostToDevice, stream));

  // Run inference
  if (!context->enqueueV2(buffers, stream, nullptr)) {
    std::cerr << "Failed to execute TensorRT inference" << std::endl;
    return -1;
  }

  // Copy output data to host
  cudaCheck(cudaMemcpyAsync(output.data(), buffers[outputIndex], outputSize,
                            cudaMemcpyDeviceToHost, stream));

  // Synchronize stream
  cudaCheck(cudaStreamSynchronize(stream));

  return 0;
}

void Inference::printModelInfo() {
  // print model input&output size and name
  std::cout << "Number of inputs: " << engine->getNbBindings() << std::endl;
  for (int i = 0; i < engine->getNbBindings(); ++i) {
    nvinfer1::Dims dims = engine->getBindingDimensions(i);
    std::cout << "Bindings " << i << " name: " << engine->getBindingName(i)
              << std::endl;
    std::cout << " shape: ";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j] << " ";
    }
    std::cout << std::endl;
  }
}