from mistnet import MistNet, train
from dataset import create_dataloader, NPYDataset
import torch
import numpy as np
import tqdm as tqdm


def infer_torch(model_path, test_loader):

    # load model
    model = MistNet(4, 4)
    model.load_state_dict(torch.load('model.pth'))
    model.eval().to("mps")

    precisions = []
    recalls = []

    for batch in tqdm.tqdm(test_loader):
        data, labels = batch
        input_data = np.array(data).reshape(1, 4, 128, 1200).astype(np.float32)
        labels = np.array(labels).reshape(128, 1200).astype(np.float32)
        input_data = torch.tensor(input_data, dtype=torch.float32).to("mps")
        output = model(input_data)
        output = output.cpu().detach().numpy()
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


infer_torch('model.pth', create_dataloader('data/5/npy', 1))
