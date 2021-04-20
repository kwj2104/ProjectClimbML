import argparse
import sys
import cv2
import time
from torch.utils.data import WeightedRandomSampler, DataLoader

from climbing_dataset import ClimbingDataset

# import wandb
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from model import Net

def train(args, model, device, train_loader, optimizer, epoch, num_samples):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)

        data, target = data.unsqueeze(1).type(torch.FloatTensor).to(device), target.type(torch.LongTensor).to(device)
        # print(data.size(), target.size())
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        print("loss: ", loss.item())
        #
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     # wandb.log({"Epoch": epoch, "Loss": loss.item()})
        #     if args.dry_run:
        #         break

        # if batch_idx * len(data)> num_samples:
        #     break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_data(train_loader):

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (0,20)
    fontScale = .5
    fontColor = (0, 0, 0)
    lineType = 2

    for data, label in train_loader:

        data = data.numpy()[0]
        print(data)
        print(data.shape)

        if label.item() == 0:
            text = "Falling"
        elif label.item() == 1:
            text = "Climbing"
        else:
            text = "Other"

        cv2.putText(data, text,
            position,
            font,
            fontScale,
            fontColor,
            lineType)

        cv2.imshow("Final output", data)
        time.sleep(3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='ProjectClimb')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # wandb setup
    # Hyperparameters
    # config_defaults = {
    #     'epoch': args.epochs,
    #     'learning_rate': args.lr,
    #     # optimizer': 'adam',
    #     'batch_size': args.batch_size,
    #     'gamma': args.gamma,
    # }
    # wandb.init(project="PC test", config=config_defaults)
    # wandb.run.name = wandb.run.id
    # wandb.run.save()
    # config = wandb.config

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # construct dataset
    climb_dataset = ClimbingDataset(frame_dataset="climb_frame_dataset.pkl", label_dataset="climb_label_dataset.pkl")
    sampler = WeightedRandomSampler(torch.DoubleTensor(climb_dataset.get_weights()), len(climb_dataset), replacement=False)
    train_loader = DataLoader(climb_dataset, sampler=sampler, **train_kwargs)

    climb_test_dataset = ClimbingDataset(frame_dataset="climb_frame_dataset_val.pkl", label_dataset="climb_label_dataset_val.pkl")
    val_sampler = WeightedRandomSampler(torch.DoubleTensor(climb_test_dataset.get_weights()), len(climb_test_dataset), replacement=False)
    test_loader = DataLoader(climb_test_dataset, sampler=val_sampler, **test_kwargs)

    # test_data(test_loader)
    # sys.exit()

    print("Dataset loaded")
    print("Dataset count: ", len(climb_dataset))
    print("Dataset val count: ", len(climb_test_dataset), len(test_loader))

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, epoch, 10000)
        train(args, model, device, test_samples, optimizer, epoch, 10000)
        # test(model, device, test_loader)
        optimizer.step()

    if args.save_model:
        torch.save(model.state_dict(), "pc_cnn.pt")

if __name__ == '__main__':
    main()
