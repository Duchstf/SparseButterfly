from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary
from models import *
import warmup_scheduler
import numpy as np

def train(args, model, device, train_loader, optimizer, epoch):

    citerion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    step_loss = []
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = citerion(output, target)
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

def test(args, model, device, test_loader):

    citerion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += citerion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 MLPs Vanilla and Monarch')

    parser.add_argument('--monarch', action='store_true',  default=False, help='Whether to train with monarch matrices or not')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 300)')
    parser.add_argument('--seed', type=int, default=420, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--hidden-c', type=int, default=512)
    parser.add_argument('--hidden-s', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--drop-p', type=int, default=0.)
    parser.add_argument('--off-act', action='store_true', help='Disable activation function')
    parser.add_argument('--is-cls-token', action='store_true', help='Introduce a class token.')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--warmup-epoch', type=int, default=5)
    parser.add_argument('--autoaugment', action='store_true')
    parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad")
    parser.add_argument('--cutmix-beta', type=float, default=1.0)
    parser.add_argument('--cutmix-prob', type=float, default=0.)

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False, help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda: device = torch.device("cuda")
    elif use_mps: device = torch.device("mps")
    else: device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 Data
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    #Initialize the model
    if args.monarch:
        model = None
    else:
        model = MLPMixer(in_channels=3,
        img_size=32, 
        patch_size=4, 
        hidden_size=128, 
        hidden_s=512, 
        hidden_c=64, 
        num_layers=8, 
        num_classes=10, 
        drop_p=0.,
        off_act=False,
        is_cls_token=True)

        print(torchsummary.summary(model, (3,32,32)))
    
    # Set optimizer and scheduler
    optimizer =optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    if args.scheduler=='step': base_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=args.gamma)
    elif args.scheduler=='cosine': base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else: raise ValueError(f"No such scheduler: {scheduler}")

    if args.warmup_epoch: scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)
    else: scheduler = base_scheduler
    
    # Record training loss
    training_loss = []
    test_loss_list = []
    for epoch in range(1, args.epochs + 1):
        step_loss = train(args, model, device, train_loader, optimizer, epoch)
        training_loss += step_loss
        test_loss = test(args, model, device, test_loader)
        test_loss_list.append(test_loss)
        scheduler.step()

    if args.save_model:
        if args.monarch: torch.save(model.state_dict(), "plots/CIFAR10_mlp_monarch.pt")
        else: torch.save(model.state_dict(), "plots/CIFAR10_mlp_vanilla.pt")
        
    #Save step loss
    loss_name = 'loss/CIFAR10_MLP_Monarch.npy' if args.monarch else 'loss/CIFAR10_MLP_Vanilla.npy'
    np.save(loss_name, np.asarray(training_loss))

    loss_name = 'loss/CIFAR10_MLP_Monarch_test.npy' if args.monarch else 'loss/CIFAR10_MLP_Vanilla_test.npy'
    np.save(loss_name, np.asarray(test_loss_list))

if __name__ == '__main__':
    main()