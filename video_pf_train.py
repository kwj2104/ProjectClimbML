import torch
from model import Net, CNNLSTM_Net, CNNBiLSTM_Net, CNNBiLSTM_pf_Net
from torch.utils.data import WeightedRandomSampler, DataLoader
from climbing_dataset import ClimbingPFVideoDataset
import numpy as np
import torch.nn.functional as F
import sys
import pickle
from preprocessing import save_video
import time

# import cv2

#  Spell
import spell.metrics as metrics

# Print everything
np.set_printoptions(threshold=np.inf)


def train(model, device, train_loader, optimizer, epoch, num_samples):
    log_interval = 10
    dry_run = False

    # Add weights to loss function to deal with class imbalance between send/fail
    # weights = torch.tensor([6.0, .5]).to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.unsqueeze(2).type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
        #print(data.size(), target.size())
        # if (data.size()[1] != target.size()[1]):
        #     print("bad sample???")
        #     continue

        # print(target)

        optimizer.zero_grad()
        output = model(data)
        # print(output.size())
        # print(target.size())

        loss = criterion(output[0], target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break

        # sys.exit()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    vid_len = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    print("Validation...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.unsqueeze(2).type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            #print(pred_prob.size())
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            loss = criterion(output[0], target)
            print(loss.item())
            # correct = pred.eq(target.view_as(pred)).sum().item()
            # print("Video length: ", vid_len, " Correct predictions: ", correct, " Percent: ",
            #       str(float(correct / vid_len)))
            # metrics.send_metric("Val_correctpct", float(correct / vid_len))

    sys.exit()

def run_model():

    dataset_path = "/mnt/"

    # dataset_path = "Processed_dataset/"
    model_path = "Pretrained_models/"

    # Training settings for Colab
    batch_size = 1
    test_batch_size = 1
    epochs = 10
    lr = .0003
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
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Construct dataset from dataloader
    climb_dataset = ClimbingPFVideoDataset(frame_dataset= dataset_path + "climb_video_dataset.pkl", label_dataset=dataset_path + "climb_video_pf_label_dataset.pkl")
    train_loader = DataLoader(climb_dataset, **train_kwargs)

    climb_test_dataset = ClimbingPFVideoDataset(frame_dataset= dataset_path + "climb_video_dataset_val.pkl", label_dataset=dataset_path + "climb_video_pf_label_dataset_val.pkl")
    test_loader = DataLoader(climb_test_dataset, **test_kwargs)
    print("Loaded training and validation dataset")

    # Load pretrained CNN parameters
    # device = torch.device('cuda')
    pretrained_model = Net()
    pretrained_model.load_state_dict(torch.load(model_path + "pc_cnn.pt", map_location=device))
    pretrained_model.to(device)

    model = CNNBiLSTM_pf_Net().to(device)
    # model.preinitialize_cnn(pretrained_model)
    print("Preinitialized CNN-LSTM")

    #print(pretrained_model.fc2.bias.eq(model.fc_out.bias))
    #sys.exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Training...")
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        # train(model, device, train_loader, optimizer, epoch, 60000)
        test(model, device, test_loader)
        optimizer.step()

    if save_model:
        torch.save(model.state_dict(), "pc_bilstm_pf.pt")

if __name__ == '__main__':
    run_model()
