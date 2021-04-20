import torch
from torch import nn
import torch.nn.functional as F
import sys
import cv2
import time

class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        output = x

        return output


class CNNLSTM_Net(nn.Module):

    def __init__(self):

        super(CNNLSTM_Net, self).__init__()

        # CNN (need to pre-initialize)
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 3)

        # LSTM
        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=1)

    def preinitialize_cnn(self, cnn_model):

        pretrained_params = []
        for layer in cnn_model.parameters():
            pretrained_params.append(layer)

        self.conv1.weight = pretrained_params[0]
        self.conv1.bias = pretrained_params[1]

        self.conv2.weight = pretrained_params[2]
        self.conv2.bias = pretrained_params[3]

        self.conv3.weight = pretrained_params[4]
        self.conv3.bias = pretrained_params[5]

        self.conv4.weight = pretrained_params[6]
        self.conv4.bias = pretrained_params[7]

        self.conv5.weight = pretrained_params[8]
        self.conv5.bias = pretrained_params[9]

        self.fc1.weight = pretrained_params[10]
        self.fc1.bias = pretrained_params[11]

        self.fc_out.weight = pretrained_params[12]
        self.fc_out.bias = pretrained_params[13]

        #sys.exit()

    def forward(self, frames):

        #Reshape to iterate over frames
        size = frames.size()
        #print(size)
        b, f, c, x, y = size[0], size[1], size[2], size[3], size[4]
        frames = frames.reshape(f, b, c, x, y)

        # initialize hidden state and final output
        hidden = (torch.zeros(1, f, 128), torch.zeros(1, f, 128))
        out = []

        cnn_output_list = []

        # Send every frame through the CNN first
        for i in range(frames.size()[0]):
            # f = frames[i].squeeze(0).squeeze(0).numpy()
            # print(f)
            # cv2.imshow("Final output", f)
            # # print(frames[i].reshape(x, y, 1).numpy().shape)
            # # time.sleep(.05)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            #print(frames[i].size())

            x = self.conv1(frames[i])
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            # print(x)
            # sys.exit()

            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            x = self.conv3(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            x = self.conv4(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            x = self.conv5(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

            x = torch.flatten(x, 1)

            x = self.fc1(x)
            x = F.relu(x)

            cnn_output_list.append(x)

        cnn_cat = torch.cat(cnn_output_list).unsqueeze(0)
        #print("CNN_output: ", cnn_cat.size())

        # Run frames through LSTM
        rnn_output, hidden = self.rnn(cnn_cat, hidden)
        #print("RNN_output: ", rnn_output.size())
        rnn_output = rnn_output.reshape(f, 1, 128)

        # TEST with no LSTM - REMOVE
        #rnn_output = cnn_cat.reshape(f, 1, 128)

        # send every RNN output in the video through final FC layer
        for i in range(f):
            out.append(self.fc_out(rnn_output[i]))

        return torch.cat(out)
