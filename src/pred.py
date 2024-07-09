import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import time
import os


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.nptype = {
    trt.bool: np.bool_,
    trt.int8: np.int8,
    trt.int32: np.int32,
    trt.float16: np.float16,
    trt.float32: np.float32,
}

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding)) * 1
        dtype = trt.nptype[engine.get_tensor_dtype(binding)]
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    stream.synchronize()
    return [out[0] for out in outputs]


def prepare_input(data):
    np.copyto(inputs[0][0], data.ravel())



# 替换为你的 TensorRT 引擎文件路径
engine_file_path = "/home/mist/models/0703/engine.trt"
engine = load_engine(engine_file_path)
inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()


# use test data
test_dir = "data/verify/"
test_files = os.listdir(test_dir)

precisions = []
recalls = []

for file in test_files:
    data = np.load(test_dir + file).reshape(5, 128, 1200).astype(np.float32)

    input_data = torch.tensor(data[:4,:,:], dtype=torch.float32)  # 前4个维度作为输入
    labels = torch.tensor(data[4,:,:], dtype=torch.long)  # 最后1个维度作为标签

    prepare_input(input_data)
    start_time = time.time()
    output = do_inference(context, bindings, inputs, outputs, stream)
    end_time = time.time()
    predictions = np.argmax(np.array(output).reshape(1, 4, 128,1200), axis=1).reshape(128,1200)
    labels = np.array(labels).reshape(128,1200)

    
    # for i in range(128):
    #     for j in range(1200):
    #         label_flag = label[i][j].item()
    #         infer_flag = output[i][j].item()
    #         if label_flag not in bias_dict:
    #             bias_dict[label_flag] = 
    #         else:
    #             bias_dict[label_flag].append(infer_flag)
    # for key in bias_dict:
    #     bias_dict[key] = np.unique(bias_dict[key])

    
    # precison and recall
    precision = np.zeros(4)
    recall = np.zeros(4)

    for i in range(4):
        precision[i] = np.sum(np.logical_and(
            labels == i, predictions == i)) / np.sum(predictions == i)
        recall[i] = np.sum(np.logical_and(
            labels == i, predictions == i)) / np.sum(labels == i)
    
    precisions.append(precision)
    recalls.append(recall)

    print(f"inference time: {end_time - start_time}, precision: {precision}, recall: {recall}")

precisions = np.array(precisions)
recalls = np.array(recalls)
print("precision: ", np.mean(precisions, axis=0))
print("recall: ", np.mean(recalls, axis=0))