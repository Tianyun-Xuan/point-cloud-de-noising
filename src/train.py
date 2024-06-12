from mistnet import MistNet, train
from dataset import create_dataloader, NPYDataset
import torch


# def quantized_test():
#     model = MistNet()
#     model.eval()
#     fuse_model(model)
#     model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
#     torch.quantization.prepare(model, inplace=True)
#     torch.quantization.convert(model, inplace=True)
#     model.load_state_dict(torch.load('model_int8.pth'))
#     # test_loader = create_dataloader('data/5/train', 1)

#     model.eval()

#     # scriped_model = torch.jit.script(model)

#     # test(model, test_loader, device='cpu')

#     dummy_input = torch.randn(1, 6, 128, 1200)

#     # onnx model int8
#     torch.onnx.export(
#     model,
#     dummy_input,
#     "quantized_model.onnx",
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output'],
#     # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
#     # keep_initializers_as_inputs=True,
#     verbose=False)


if __name__ == "__main__":
    num_classes, height, width = 4, 128, 1200
    model = MistNet(num_classes=num_classes, in_channels=4)
    optimizer = model.configure_optimizers()
    batch_size = 4
    # 创建训练、验证和测试数据加载器
    train_loader = create_dataloader('data/train', batch_size)
    val_loader = create_dataloader('data/train', batch_size)
    test_loader = create_dataloader('data/train', batch_size)

    print(torch.__version__)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(device)

    # 训练模型
    train(model, train_loader, val_loader, optimizer, epochs=10, device=device)

    # save model
    torch.save(model.state_dict(), 'model_class4_cpu.pth')
    print("Model saved.")
