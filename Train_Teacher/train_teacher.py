import os
import torch
import argparse
import resnet_teacher
import wresnet_cifar

from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import SVHN
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='SVHN', choices=['MNIST', 'SVHN', 'cifar10', 'cifar100'])
parser.add_argument('--data', type=str, default='/home/yaof/data/SVHN/')
parser.add_argument('--output_dir', type=str, default='/home/yaof/trained_models/')
parser.add_argument('--teacher_name', type=str, default='Res18', help='the name of the teacher network')
parser.add_argument('--use_gpu', type=bool, default=True, help='set to False to debug on cpu, using LeNets')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

args.device = 'cuda:0' if args.use_gpu else 'cpu'
acc = 0
acc_best = 0

if args.dataset == 'MNIST':
    data_train = MNIST(args.data,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    data_test = MNIST(args.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ]))

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    net = LeNet5().to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # arg.data: the path of the dataset
    data_train = CIFAR10(args.data, transform=transform_train, download=False)
    data_test = CIFAR10(args.data, train=False, transform=transform_test,
                        download=True)

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

    args.data = os.path.join(args.data, args.teacher_name)
    args.output_dir = os.path.join(args.output_dir, args.teacher_name)
    if args.teacher_name == 'Res50':
        net = resnet_teacher.ResNet50().to(args.device)
    if args.teacher_name == 'Res34':
        net = resnet_teacher.ResNet34().to(args.device)

    if args.teacher_name == 'WRN50-2':
        net = wresnet_cifar.WideResNet(depth=50, num_classes=10, widen_factor=2, dropRate=0.0).to(args.device)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

if args.dataset == 'SVHN':
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    data_train = SVHN(args.data, download=True, transform=transform_train)

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    data_test = SVHN(args.data, split='test', download=True, transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=512, num_workers=8)

    args.data = os.path.join(args.data, args.teacher_name)
    args.output_dir = os.path.join(args.output_dir, args.teacher_name)
    if args.teacher_name == 'Res50':
        net = resnet_teacher.ResNet50().to(args.device)
    if args.teacher_name == 'Res34':
        net = resnet_teacher.ResNet34().to(args.device)
    if args.teacher_name == 'Res18':
        net = resnet_teacher.ResNet18().to(args.device)

    if args.teacher_name == 'WRN50-2':
        net = wresnet_cifar.WideResNet(depth=50, num_classes=10, widen_factor=2, dropRate=0.0).to(args.device)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

if args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR100(args.data,
                          transform=transform_train)
    data_test = CIFAR100(args.data,
                         train=False,
                         transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=128, num_workers=0)

    if args.teacher_name == 'Res34':
        net = resnet_teacher.ResNet34(num_class=100).to(args.device)
    if args.teacher_name == 'Res18':
        net = resnet_teacher.ResNet18(num_class=100).to(args.device)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 40:
        lr = 0.1
    elif epoch < 80:
        lr = 0.04
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    if args.dataset != 'MNIST':
        adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).to(args.device), Variable(labels).to(args.device)

        optimizer.zero_grad()

        output, _, _, _, _ = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()


def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).to(args.device), Variable(labels).to(args.device)
            output, _, _, _, _ = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
        torch.save(net.state_dict(), args.output_dir + 'best_teacher')
    print('Test Avg. Loss: %f, Accuracy: %f, Best Accuracy: %f'
          % (avg_loss.data.item(), acc, acc_best))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    if args.dataset == 'MNIST':
        epoch = 10
    else:
        epoch = 150
    if True:
        for e in range(1, epoch):
            train_and_test(e)
    else:
        model_path = "/home/yaof/trained_models/res34_nasty_teacher_cifar100"
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint)
        test()

    torch.save(net.state_dict(), args.output_dir + 'teacher')
    # torch.save(net, args.output_dir + 'teacher')


if __name__ == '__main__':
    main()