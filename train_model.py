import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from resnet.models.resnet import Resnet110
from resnet.loaders import get_train_loader, get_test_loader


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Change an optimizer's learning rate.
    Parameters
    ----------
    optimizer: torch.optim.Optimizer
    lr: float
    Returns
    -------
    optimizer: torch.optim.Optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


n_epochs = 500

lr = 0.1
lr_dec_mul = 0.1
lr_dec_moments = [250, 375]

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    train_loader = get_train_loader()
    train_size = len(train_loader.dataset)
    test_loader = get_test_loader()
    test_size = len(test_loader.dataset)

    writer = SummaryWriter('./log')

    model = Resnet110(10)

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


    def train():
        model.train()
        total_loss = 0
        correct = 0
        for inputs, targets in train_loader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()

        return total_loss / train_size, correct / train_size


    def test():
        model.eval()
        total_loss = 0
        correct = 0
        for inputs, targets in test_loader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()

        return total_loss / test_size, correct / test_size

    #for iter in range()

    for epoch in range(n_epochs):
        if epoch in lr_dec_moments:
            lr *= lr_dec_mul
            set_lr(optimizer, lr)

        loss, acc = train()
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/acc', acc, epoch)
        loss, acc = test()
        writer.add_scalar('test/loss', loss, epoch)
        writer.add_scalar('test/acc', acc, epoch)

    writer.export_scalars_to_json('./log.json')
