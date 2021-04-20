import torch
from model import Net, CNNLSTM_Net
from torch.utils.data import WeightedRandomSampler, DataLoader
from climbing_dataset import ClimbingVideoDataset
import numpy as np
import torch.nn.functional as F
import sys
import pickle
import cv2
from preprocessing import save_video
import time

# Print everything
np.set_printoptions(threshold=np.inf)

def train(model, device, train_loader, optimizer, epoch, num_samples):
    log_interval=10
    dry_run=False
    criterion= torch.nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.unsqueeze(2).type(torch.FloatTensor).to(device), target.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output = model(data)

        print(output.size())
        print(target[0].size())

        # loss = F.nll_loss(output, target)
        loss = criterion(output, target[0])
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break

        # if (batch_idx * len(data)) > num_samples:
        #     break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    vid_len = 0
    with torch.no_grad():
        for data, target in test_loader:
            vid_len=target.size()[1]
            data, target = data.unsqueeze(2).type(torch.FloatTensor).to(device), target.type(torch.LongTensor).to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            #print(pred_prob.size())
            pred = output[0].argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #ßprint(pred.numpy())
            correct = pred.eq(target.view_as(pred)).sum().item()
            print("Video length: ", vid_len, " Correct predictions: ", correct, " Percent: ",
                  str(float(correct / vid_len)))


def run_model():
    # Training settings for Colab
    batch_size = 1
    test_batch_size = 1
    epochs = 10
    lr = .0005
    gamma = .7
    no_cuda = False
    dry_run = False
    seed = 1
    log_interval = 10
    save_model = True


    # CUDA and param setup
    # torch.manual_seed(seed)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Construct dataset from dataloader
    climb_dataset = ClimbingVideoDataset(frame_dataset="Processed dataset/climb_video_dataset.pkl", label_dataset="Processed dataset/climb_video_label_dataset.pkl")
    train_loader = DataLoader(climb_dataset, **train_kwargs)

    climb_test_dataset = ClimbingVideoDataset(frame_dataset="Processed dataset/climb_video_dataset_val.pkl", label_dataset="Processed dataset/climb_video_label_dataset_val.pkl")
    test_loader = DataLoader(climb_test_dataset, **test_kwargs)
    print("Loaded training and validation dataset")

    # Load pretrained CNN parameters
    device = torch.device('cpu')
    pretrained_model = Net()
    pretrained_model.load_state_dict(torch.load("pc_cnn.pt", map_location=device))
    pretrained_model.to(device)

    model = CNNLSTM_Net()
    model.preinitialize_cnn(pretrained_model)
    print("Preinitialized CNN-LSTM")

    #print(pretrained_model.fc2.bias.eq(model.fc_out.bias))
    #sys.exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, 60000)
        test(model, device, test_loader)
        optimizer.step()

if __name__ == '__main__':
    run_model()
