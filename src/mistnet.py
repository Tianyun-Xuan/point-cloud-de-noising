from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AveragePrecision
from lilanet import LiLaBlock, EvenBlock, BasicConv2d
from dataset import create_dataloader
import numpy as np
import torch.quantization


class MistNet(nn.Module):
    def __init__(self, num_classes=8, in_channels=6):
        super(MistNet, self).__init__()
        self.lila1 = EvenBlock(in_channels, 96, modified=True)
        self.lila2 = EvenBlock(96, 128, modified=True)
        self.lila3 = EvenBlock(128, 256, modified=True)
        # self.lila4 = EvenBlock(256, 256, modified=True)
        # self.dropout = nn.Dropout2d()
        self.lila5 = EvenBlock(256, 128, modified=True)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

        self.accuracy = Accuracy(num_classes=num_classes,
                                 average='weighted', task='multiclass')
        self.average_precision = AveragePrecision(
            num_classes=num_classes, average='weighted', task='multiclass')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.test_results = []

    def forward(self, x):

        x = self.lila1(x)
        x = self.lila2(x)
        x = self.lila3(x)
        # x = self.lila4(x)
        # x = self.dropout(x)
        x = self.lila5(x)
        x = self.classifier(x)

        return x

    def shared_step(self, batch):
        data, labels = batch
        logits = self(data)
        loss = F.cross_entropy(logits, labels)
        return loss, logits, labels

    def training_step(self, batch):
        loss, logits, labels = self.shared_step(batch)
        acc = self.accuracy(logits, labels)
        return loss, acc

    def validation_step(self, batch):
        loss, logits, labels = self.shared_step(batch)
        acc = self.accuracy(logits, labels)
        return loss, acc

    def test_step(self, batch):
        loss, logits, labels = self.shared_step(batch)
        probabilities = F.softmax(logits, dim=1)
        probabilities = torch.clamp(probabilities, min=1e-6, max=1-1e-6)

        acc = self.accuracy(logits, labels)
        ap = self.average_precision(probabilities, labels)
        predictions = torch.argmax(logits, dim=1)
        self.test_results.append(
            {'predictions': predictions, 'labels': labels})
        return loss, acc, ap

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), betas=(0.9, 0.999), eps=1e-8)
        return optimizer

    def on_test_epoch_end(self):
        predictions = torch.cat([x['predictions'] for x in self.test_results])
        labels = torch.cat([x['labels'] for x in self.test_results])
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        np.save("test_predictions.npy", predictions)
        np.save("test_labels.npy", labels)
        self.test_results = []


# def fuse_model(model):
#     for module_name, module in model.named_children():
#         if isinstance(module, BasicConv2d):
#             torch.quantization.fuse_modules(module, ['conv', 'bn', 'relu'], inplace=True)
#         else:
#             fuse_model(module)

def train(model, train_loader, val_loader, optimizer, epochs=10, device='cpu'):
    model.eval()
    # fuse_model(model)
    model.train()
    model.to(device)
    print("Totally dataset size: ", len(train_loader.dataset))
    print("Training device: ", device)

    # # 融合模型中的层
    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # torch.quantization.prepare(model, inplace=True)

    for epoch in range(epochs):
        train_losses = []
        train_accs = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch"):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss, acc = model.training_step((data, labels))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(acc.item())
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Train Acc: {avg_train_acc}")

        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", unit="batch"):
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                loss, acc = model.validation_step((data, labels))
                val_losses.append(loss.item())
                val_accs.append(acc.item())
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)
        print(
            f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss}, Val Acc: {avg_val_acc}")

        # # Save checkpoint every epoch
        # checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
        # torch.save(model.state_dict(), checkpoint_path)
        # print(f"Checkpoint saved: {checkpoint_path}")

    # # 量化
    # model.eval().to("cpu")
    # torch.quantization.convert(model, inplace=True)
    # torch.save(model.state_dict(), 'model_class4.pth')


def test(model, test_loader, device='cpu'):
    model.eval()
    model.to(device)
    test_losses = []
    test_accs = []
    test_aps = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            loss, acc, ap = model.test_step((data, labels))
            test_losses.append(loss.item())
            test_accs.append(acc.item())
            test_aps.append(ap.item())
    avg_test_loss = np.mean(test_losses)
    avg_test_acc = np.mean(test_accs)
    avg_test_ap = np.mean(test_aps)
    print(
        f"Test Loss: {avg_test_loss}, Test Acc: {avg_test_acc}, Test AP: {avg_test_ap}")
    model.on_test_epoch_end()


if __name__ == "__main__":
    num_classes, height, width = 2, 128, 1024
    model = MistNet(num_classes)  # .to('cuda')
    optimizer = model.configure_optimizers()

    batch_size = 4

    # 创建训练、验证和测试数据加载器
    train_loader = create_dataloader('data/train', batch_size)
    val_loader = create_dataloader('data/val', batch_size)
    test_loader = create_dataloader('data/test', batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练模型
    train(model, train_loader, val_loader, optimizer, epochs=2, device=device)

    # 测试模型
    test(model, test_loader, device=device)

    # 模型推理测试
    inp = torch.randn(1, 6, height, width).to(device)  # 6通道输入
    out = model(inp)
    assert out.size() == torch.Size([1, num_classes, height, width])
    print("Pass size check.")
