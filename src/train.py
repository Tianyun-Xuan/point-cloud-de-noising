from mistnet import MistNet, train, test, fuse_model
from dataset import create_dataloader, NPYDataset
import torch


def quantized_test():
    model = MistNet()
    model.eval()
    fuse_model(model)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(torch.load('model_int8.pth'))
    # test_loader = create_dataloader('data/5/train', 1)

    model.eval()

    # scriped_model = torch.jit.script(model)
    
    # test(model, test_loader, device='cpu')

    dummy_input = torch.randn(1, 6, 128, 1200)

    # onnx model int8
    torch.onnx.export(
    model,
    dummy_input,
    "quantized_model.onnx",
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    # keep_initializers_as_inputs=True,
    verbose=False)
    

if __name__ == "__main__":
    num_classes, height, width = 8, 128, 1200
    model = MistNet(num_classes)
    optimizer = model.configure_optimizers()
    batch_size = 4
    # 创建训练、验证和测试数据加载器
    train_loader = create_dataloader('data/6/train', batch_size)
    val_loader = create_dataloader('data/6/train', batch_size)
    test_loader = create_dataloader('data/6/train', batch_size)

    print(torch.__version__)
    # 判断是否支持MPS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # 训练模型
    train(model, train_loader, val_loader, optimizer, epochs=1, device=device)

    # save model
    torch.save(model.state_dict(), 'model_int8.pth')
    print("Model saved.")