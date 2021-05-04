import torch
import cv2
import time
import sys
import numpy as np

from preprocessing import save_video
from model import Net, CNNLSTM_Net, CNNBiLSTM_Net


def test_model_video(model, filename="raw_videos/AndyLiu/Moonboard_Benchmarks_Zen_Master_(7c__V9).mp4"):
    model.eval()

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (0,20)
    fontScale = .5
    fontColor = (0, 0, 0)
    lineType = 2

    frames = save_video(filename)
    text = ""
    data = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0).unsqueeze(2).type(torch.FloatTensor)
    #print(data.size())
    output = model(data)
    #print(output.size())
    #sys.exit()

    for i, f in enumerate(frames):

        #data = torch.from_numpy(f).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)

        #sys.exit()
        pred = output[i].argmax(dim=0, keepdim=True)

        if pred.item() == 0:
            text = "Falling"
        elif pred.item() == 1:
            text = "Climbing"
        else:
            text = "Other"

        cv2.putText(f, text,
            position,
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.imshow("Final output", f)
        #print(f.shape)
        time.sleep(.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue

    for f in frames:

        data = torch.from_numpy(f).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)

        sys.exit()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        if pred.item() == 0:
            text = "Falling"
        elif pred.item() == 1:
            text = "Climbing"
        else:
            text = "Other"

        cv2.putText(f, text,
            position,
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.imshow("Final output", f)
        #print(f.shape)
        time.sleep(.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue

def test_model(model, filename="raw_videos/AndyLiu/Moonboard_Benchmarks_Zen_Master_(7c__V9).mp4"):
    model.eval()

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (0,20)
    fontScale = .5
    fontColor = (0, 0, 0)
    lineType = 2

    frames = save_video(filename)
    text = ""

    for f in frames:

        data = torch.from_numpy(f).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)

        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        if pred.item() == 0:
            text = "Falling"
        elif pred.item() == 1:
            text = "Climbing"
        else:
            text = "Other"

        cv2.putText(f, text,
            position,
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.imshow("Final output", f)
        #print(f.shape)
        time.sleep(.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue

if __name__ == '__main__':
    file = "raw_videos/AndyLiu/Moonboard_Benchmarks_Zero_Hour_Hard_(7a+__V7).mp4"
    device = torch.device('cpu')
    model = Net()
    model.load_state_dict(torch.load("pc_cnn.pt", map_location=device))
    model.to(device)

    model2 = CNNLSTM_Net()
    model2.load_state_dict(torch.load("pc_cnnlstm_2.pt", map_location=device))
    model2.to(device)

    model3 = CNNBiLSTM_Net()
    model3.load_state_dict(torch.load("pc_cnnbilstm_2.pt", map_location=device))
    model3.to(device)

    #test_model(model, filename=file)
    #test_model_video(model2, filename=file)
    test_model_video(model3, filename=file)