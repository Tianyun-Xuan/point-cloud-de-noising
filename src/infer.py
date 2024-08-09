from mistnet import MistNet, train
from dataset import create_dataloader, NPYDataset
import torch
import numpy as np
import tqdm as tqdm
import os


def infer_torch(model_path, test_loader):

    # load model
    model = MistNet(4, 4)
    model.load_state_dict(torch.load('model.pth'))
    model.eval().to("mps")

    precisions = []
    recalls = []

    # frame_count = 0
    for batch in tqdm.tqdm(test_loader):
        data, labels = batch
        input_data = np.array(data).reshape(1, 4, 128, 1200).astype(np.float32)
        labels = np.array(labels).reshape(128, 1200).astype(np.float32)
        input_data = torch.tensor(input_data, dtype=torch.float32).to("mps")
        output = model(input_data)
        output = output.cpu().detach().numpy()
        predictions = np.argmax(output, axis=1).reshape(128, 1200)

        # # save id as 00001.npy
        # np.save(os.path.join('data/pred',
        #         '{}.npy'.format(str(frame_count).zfill(5))), predictions)
        # frame_count += 1

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
    print("Label 0 precision range : [{}, {}]".format(
        np.min(precisions[:, 0]), np.max(precisions[:, 0])))
    print("Label 1 precision range : [{}, {}]".format(
        np.min(precisions[:, 1]), np.max(precisions[:, 1])))
    print("Label 2 precision range : [{}, {}]".format(
        np.min(precisions[:, 2]), np.max(precisions[:, 2])))
    print("Label 3 precision range : [{}, {}]".format(
        np.min(precisions[:, 3]), np.max(precisions[:, 3])))
    print("recall: ", np.mean(recalls, axis=0))
    print("Label 0 recall range : [{}, {}]".format(
        np.min(recalls[:, 0]), np.max(recalls[:, 0])))
    print("Label 1 recall range : [{}, {}]".format(
        np.min(recalls[:, 1]), np.max(recalls[:, 1])))
    print("Label 2 recall range : [{}, {}]".format(
        np.min(recalls[:, 2]), np.max(recalls[:, 2])))
    print("Label 3 recall range : [{}, {}]".format(
        np.min(recalls[:, 3]), np.max(recalls[:, 3])))


infer_torch('model.pth', create_dataloader('data/mist/npy', 1))


# def infer_torch(model_path, test_dir):

#     # load model
#     model = MistNet(4, 4)
#     model.load_state_dict(torch.load('models/model.pth'))
#     model.eval().to("mps")

#     precisions = []
#     recalls = []
#     test_loader = os.listdir(test_dir)
#     test_loader.sort()

#     for file in tqdm.tqdm(test_loader):
#         batch = np.load(os.path.join(test_dir, file)).reshape(1, 5, 128, 1200)
#         data = batch[0, :4]
#         data[0, 0, :] = np.clip(data[0, 0, :], 0, 1023)
#         data[2, 0, :] = np.clip(data[2, 0, :], 0, 1023)
#         labels = batch[0, 4]
#         input_data = np.array(data).reshape(1, 4, 128, 1200).astype(np.float32)
#         labels = np.array(labels).reshape(128, 1200).astype(np.float32)
#         input_data = torch.tensor(input_data, dtype=torch.float32).to("mps")
#         output = model(input_data)
#         output = output.cpu().detach().numpy()
#         predictions = np.argmax(output, axis=1).reshape(128, 1200)

#         np.save(os.path.join('data/pred', file), predictions)

#         # precison and recall
#         precision = np.zeros(4)
#         recall = np.zeros(4)

#         for i in range(4):
#             precision[i] = np.sum(np.logical_and(
#                 labels == i, predictions == i)) / np.sum(predictions == i)
#             recall[i] = np.sum(np.logical_and(
#                 labels == i, predictions == i)) / np.sum(labels == i)

#         precisions.append(precision)
#         recalls.append(recall)

#     precisions = np.array(precisions)
#     recalls = np.array(recalls)
#     print("precision: ", np.mean(precisions, axis=0))
#     print("Label 0 precision range : [{}, {}]".format(
#         np.min(precisions[:, 0]), np.max(precisions[:, 0])))
#     print("Label 1 precision range : [{}, {}]".format(
#         np.min(precisions[:, 1]), np.max(precisions[:, 1])))
#     print("Label 2 precision range : [{}, {}]".format(
#         np.min(precisions[:, 2]), np.max(precisions[:, 2])))
#     print("Label 3 precision range : [{}, {}]".format(
#         np.min(precisions[:, 3]), np.max(precisions[:, 3])))
#     print("recall: ", np.mean(recalls, axis=0))
#     print("Label 0 recall range : [{}, {}]".format(
#         np.min(recalls[:, 0]), np.max(recalls[:, 0])))
#     print("Label 1 recall range : [{}, {}]".format(
#         np.min(recalls[:, 1]), np.max(recalls[:, 1])))
#     print("Label 2 recall range : [{}, {}]".format(
#         np.min(recalls[:, 2]), np.max(recalls[:, 2])))
#     print("Label 3 recall range : [{}, {}]".format(
#         np.min(recalls[:, 3]), np.max(recalls[:, 3])))


# infer_torch('model.pth', "data/mist/npy")
