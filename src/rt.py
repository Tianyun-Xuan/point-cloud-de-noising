import torch
import tensorrt as trt
from mistnet import MistNet, test
from dataset import create_dataloader, NPYDataset
from torch.quantization import quantize_dynamic
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import torch.nn.functional as F

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def check_onnx(model_path):
    # Load the ONNX model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))


def infer_onnx(model_path, test_loader):
    print(ort.get_available_providers())
    session = ort.InferenceSession(model_path)

    precisions = []
    recalls = []

    for batch in tqdm(test_loader, desc="Testing", unit="batch"):
        data, labels = batch
        input_data = np.array(data).reshape(1, 4, 128, 1200).astype(np.float32)
        labels = np.array(labels).reshape(128, 1200).astype(np.float32)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        output = outputs[0]
        print(output.shape)
        predictions = np.argmax(output, axis=1).reshape(128, 1200)

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

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    print("precision: ", np.mean(precisions, axis=0))
    print("recall: ", np.mean(recalls, axis=0))


def infer_single_frame(model_path, frame):
    print(ort.get_available_providers())
    session = ort.InferenceSession(model_path)

    data = frame[:4, :, :]
    labels = frame[4, :, :]

    input_data = np.array(data).reshape(1, 4, 128, 1200).astype(np.float32)
    output_data = np.array(labels).reshape(128, 1200).astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    output = outputs[0]

    predictions = np.argmax(output, axis=1).reshape(128, 1200)

    # precison and recall
    precision = np.zeros(4)
    recall = np.zeros(4)

    for i in range(4):
        precision[i] = np.sum(np.logical_and(
            labels == i, predictions == i)) / np.sum(predictions == i)
        recall[i] = np.sum(np.logical_and(
            labels == i, predictions == i)) / np.sum(labels == i)

    print("precision: ", precision)
    print("recall: ", recall)


def build_engine(onnx_file_path, engine_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    engine = builder.build_serialized_network(network, config)

    with open(engine_file_path, "wb") as f:
        f.write(engine)
    return engine


if __name__ == "__main__":
    # # engine = build_engine("model.onnx", "model_fp16.trt")

    # # if engine is not None:
    # #     print("Engine was built successfully!")
    # # else:
    # #     print("Engine was not built successfully!")

    # test environment
    print(torch.__version__)
    # 判断是否支持MPS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # load model

    model = MistNet(4,4)
    model.load_state_dict(torch.load('/home/mist/models/0719/model.pth'))

    model.eval().to(device)

    # ### tired but seems not work
    # # # quantization to int8 using post training dynamic quantization
    # # qconfig_spec = {
    # #     torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
    # #     torch.nn.Conv2d: torch.quantization.default_dynamic_qconfig,
    # #     torch.nn.ReLU: torch.quantization.default_dynamic_qconfig,
    # #     torch.nn.MaxPool2d: torch.quantization.default_dynamic_qconfig,
    # #     torch.nn.BatchNorm2d: torch.quantization.default_dynamic_qconfig,
    # #     torch.nn.Dropout: torch.quantization.default_dynamic_qconfig,
    # # }
    # # model_int8 = quantize_dynamic(model, qconfig_spec, dtype=torch.qint8)

    test_loader = create_dataloader('data/verify/', 1)
    # test(model, test_loader, device=device)

    dummy_input = torch.randn(1, 4, 128, 1200).to(device)

    # onnx model float32
    input_names = ["input"]
    output_names = ["output"]
    onnx_path = "model.onnx"

    torch.onnx.export(model, dummy_input, onnx_path, verbose=True,
                      input_names=input_names, output_names=output_names, opset_version = 17)

    infer_onnx(onnx_path, test_loader)

    # infer_single_frame("models/0701/model.onnx", np.load("data/8/npy/27628.npy").reshape(5,128,1200))
    # infer_single_frame("models/0703/model.onnx", np.load("data/8/npy/27628.npy").reshape(5,128,1200))
    # infer_single_frame("models/0704/model.onnx", np.load("data/8/npy/27628.npy").reshape(5,128,1200))
