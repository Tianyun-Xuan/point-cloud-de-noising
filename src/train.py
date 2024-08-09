from mistnet import MistNet, train
from dataset import create_dataloader, NPYDataset
import torch

if __name__ == "__main__":
    num_classes, height, width = 4, 128, 1200
    model = MistNet(num_classes=num_classes, in_channels=4)
    optimizer = model.configure_optimizers()
    batch_size = 4
    # 创建训练、验证和测试数据加载器
    train_loader = create_dataloader('data/mist/npy', batch_size)
    val_loader = create_dataloader('data/mist/npy', batch_size)
    test_loader = create_dataloader('data/mist/npy', batch_size)

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
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved.")
