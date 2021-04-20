import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys


class ClimbingDataset(Dataset):

    # Print everything
    np.set_printoptions(threshold=np.inf)

    # video level data structures
    label_dict = {}
    video_list = []

    # frame level data structures
    frame_list = []
    frame_label_list = []

    def __init__(self, frame_dataset, label_dataset):

        # "climb_frame_dataset.pkl"
        # "climb_label_dataset.pkl"
        with open(frame_dataset, 'rb') as pickle_file:
            self.frame_list = pickle.load(pickle_file)

        with open(label_dataset, 'rb') as pickle_file:
            self.frame_label_list = pickle.load(pickle_file)

        #print(len(self.frame_list), len(self.frame_label_list))

    def __len__(self):
        return len(self.frame_label_list)
        # return len(self.label_dict)

    def __getitem__(self, idx):
        # print(self.frame_label_list[idx])
        return torch.from_numpy(self.frame_list[idx]), torch.tensor(self.frame_label_list[idx], dtype=torch.LongTensor)

    # Create weighed sampler to deal with class imbalance
    def get_weights(self):
        unique, counts = np.unique(self.frame_label_list, return_counts=True)
        num_samples = sum(counts)
        class_weights = [num_samples / counts[i] for i in range(len(counts))]

        # Get balanced sample between climbing vs non-climbing
        # for i in range(len(class_weights)):
        #     if i != 1:
        #         class_weights[i] = (num_samples - counts[1]) / num_samples

        weights = [class_weights[self.frame_label_list[i]] for i in range(int(num_samples))]

        return weights


class ClimbingVideoDataset(ClimbingDataset):

    def __getitem__(self, idx):
        #print(self.frame_list[idx][0])
        #print(np.stack(self.frame_list[idx],axis=0))

        data = torch.from_numpy(np.stack(self.frame_list[idx]))
        #print(data.size())
        #x, y, z = data.size()[0], data.size()[1], data.size()[2]
        #data = data.reshape(z, x, y)
        # print(data[0].numpy())
        # sys.exit()

        label = torch.tensor(np.stack(self.frame_label_list[idx], axis=0), dtype=torch.int32).squeeze(0).squeeze(0)

        return data, label

