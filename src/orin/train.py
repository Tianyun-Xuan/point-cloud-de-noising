from mistnet import MistNet, train, test
from dataset import create_dataloader, NPYDataset
import torch

if __name__ == "__main__":
    num_classes, height, width = 2, 128, 1024
    model = MistNet(num_classes)  # .to('cuda')
    optimizer = model.configure_optimizers()

    batch_size = 4

    # 创建训练、验证和测试数据加载器
    train_loader = create_dataloader('data/train', batch_size)
    val_loader = create_dataloader('data/val', batch_size)
    test_loader = create_dataloader('data/test', batch_size)

    print(torch.__version__)
    # 判断是否支持MPS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)

    # 训练模型
    train(model, train_loader, val_loader, optimizer, epochs=10, device=device)

    # 测试模型
    test(model, test_loader, device=device)
    
    # save model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved.")
    
