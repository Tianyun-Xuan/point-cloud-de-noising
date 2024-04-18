import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# Correctly use torchmetrics for metrics
from torchmetrics import Accuracy, AveragePrecision

from .lilanet import LiLaBlock  # Ensure this relative import works in your project structure
import numpy as np


class MistNet(pl.LightningModule):
    def __init__(self, num_classes=8):
        super().__init__()
        self.lila1 = LiLaBlock(6, 96, modified=True)
        self.lila2 = LiLaBlock(96, 128, modified=True)
        self.lila3 = LiLaBlock(128, 256, modified=True)
        self.lila4 = LiLaBlock(256, 256, modified=True)
        self.dropout = nn.Dropout2d()
        self.lila5 = LiLaBlock(256, 128, modified=True)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

        # Initialize metrics with the appropriate task
        self.accuracy = Accuracy(num_classes=num_classes,
                                 average='macro', task='multiclass')
        self.average_precision = AveragePrecision(
            num_classes=num_classes, task='multiclass')

        self.save_hyperparameters()  # Save hyperparameters

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.test_results = []  # 用于存储测试结果的列表

    def forward(self, distance_1, reflectivity_1, distance_2, reflectivity_2, distance_3, reflectivity_3):
        x = torch.cat([distance_1, reflectivity_1, distance_2,
                      reflectivity_2, distance_3, reflectivity_3], 1)
        x = self.lila1(x)
        x = self.lila2(x)
        x = self.lila3(x)
        x = self.lila4(x)
        x = self.dropout(x)
        x = self.lila5(x)
        x = self.classifier(x)
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Assuming the batch only contains input data for prediction
        distance_1, reflectivity_1, distance_2, reflectivity_2, distance_3, reflectivity_3 = batch
        # Compute logits using the forward method
        logits = self(distance_1, reflectivity_1, distance_2,
                      reflectivity_2, distance_3, reflectivity_3)
        predictions = torch.argmax(logits, dim=1)  # Convert logits to predictions
        return predictions

    def shared_step(self, batch):
        distance_1, reflectivity_1, distance_2, reflectivity_2, distance_3, reflectivity_3, labels = batch
        logits = self(distance_1, reflectivity_1, distance_2,
                      reflectivity_2, distance_3, reflectivity_3)
        loss = F.cross_entropy(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # Compute and log additional metrics as needed
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        # Compute and log additional metrics as needed

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.shared_step(batch)
        # Log test metrics
        acc = self.accuracy(logits, labels)
        ap = self.average_precision(logits, labels)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_ap", ap, on_step=False, on_epoch=True)

        # 将需要的结果添加到列表中
        predictions = torch.argmax(logits, dim=1)
        self.test_results.append({'predictions': predictions, 'labels': labels})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), eps=1e-8)
        return optimizer

    def on_test_epoch_end(self):
        # 此时 self.test_results 包含了整个测试集的推断结果
        # 你可以在这里进行分析或将结果保存到文件

        # 例如，保存到一个文件中
        # 假设 self.test_results 是一个列表，每个元素都是一个字典，包含 'predictions' 和 'labels'
        predictions = torch.cat([x['predictions'] for x in self.test_results])
        labels = torch.cat([x['labels'] for x in self.test_results])
        # 确保在 CPU 上并转换为 numpy 以便保存
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        # 使用 numpy 保存
        np.save("test_predictions.npy", predictions)
        np.save("test_labels.npy", labels)

        # 清空结果列表以释放内存
        self.test_results = []


if __name__ == "__main__":
    num_classes, height, width = 3, 64, 512
    model = MistNet(num_classes)
    inp = torch.randn(6, 1, height, width)
    out = model(inp, inp)
    assert out.size() == torch.Size([5, num_classes, height, width])
    print("Pass size check.")
