from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from models import *
from utils import plot_diag_weight

def train(args, model, device, train_loader, optimizer, epoch):
    
    step_loss = []
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        step_loss.append(loss.item())
        
    return step_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 MLPs Vanilla and Monarch')
    parser.add_argument('--monarch', action='store_true',  default=False,
                        help='Whether to train with monarch matrices or not')
    
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    
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
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size,
                    'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 Data
    dataset_train = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    dataset_test = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    #Initialize the model
    if args.monarch:
        model = CIFAR10_Monarch_MLP(3072).to(device)
        
        # # Plot the diagonal matrix
        # fc1= model._modules['fc1']
        # plot_diag_weight(fc1)
    else:
        model = CIFAR10_MLP_Vanilla(3072).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #args.lr

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    training_loss = []
    for epoch in range(1, args.epochs + 1):
        step_loss = train(args, model, device, train_loader, optimizer, epoch)
        training_loss += step_loss
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        if args.monarch: torch.save(model.state_dict(), "plots/CIFAR10_mlp_monarch.pt")
        else: torch.save(model.state_dict(), "plots/CIFAR10_mlp_vanilla.pt")
        
    #Save step loss
    loss_name = 'loss/CIFAR10_MLP_Monarch.npy' if args.monarch else 'loss/CIFAR10_MLP_Vanilla.npy'
    np.save(loss_name, np.asarray(training_loss))

if __name__ == '__main__':
    main()