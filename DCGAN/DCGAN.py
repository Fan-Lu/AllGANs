'''DCGAN on MNIST'''

import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse, os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='DCGAN')
parser.add_argument('--data_dir', metavar='DIR', help='path to data', default='../Data/MNIST_data/')
parser.add_argument('--image_size', default=64, type=int, help='size of image')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--num_epoch', default=20, type=int, help='batch size')
parser.add_argument('--use_cuda', default=True, type=int, help='batch size')
parser.add_argument('--model_save', metavar='DIR', help='path to store results', default='model_saved/')
parser.add_argument('--result_save', metavar='DIR', help='path to store results', default='result_saved/')

args = parser.parse_args()


# Training dataset
transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

mnist_data = datasets.MNIST(root=args.data_dir,
                         train=True,
                         transform=transform,
                         download=True)

trainloader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=args.batch_size,
                                          drop_last=False, shuffle=True)


class Disciminator(nn.Module):
    def __init__(self):
        super(Disciminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 13 * 13, 64 * 13 * 13)
        self.fc2 = nn.Linear(64 * 13 * 13, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = self.maxp1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = self.maxp1(x)

        x = x.view(-1, 64*13*13)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.fc2(x)
        return x


class Generator(nn.Module):
    def __init__(self, noise_dim=96):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 1024)
        self.fc2 = nn.Linear(1024, 8*8*128)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(8*8*128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)

        x = x.view(-1, 128, 8, 8)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.bn3(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.bn4(x)
        x = F.tanh(self.conv3(x))

        return x


D = Disciminator().cuda()
D_Optimizer = torch.optim.Adam(D.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
G = Generator().cuda()
G_Optimizer = torch.optim.Adam(G.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
BCE = nn.BCEWithLogitsLoss()

if __name__ == '__main__':

    for epoch in range(args.num_epoch):
        for index, data in enumerate(trainloader, 0):
            #   1. Train D on real+fake
            inputs, label = data
            D.zero_grad()

            real_image = Variable(inputs).cuda()
            real_logits = D(2 * (real_image - 0.5))
            real_loss = BCE(real_logits, Variable(torch.ones(real_logits.size())).cuda())
            real_loss.backward()

            fake_input = Variable(torch.randn(args.batch_size, 96)).cuda()
            fake_image = G(fake_input)
            fake_logits = D(fake_image)
            fake_loss = BCE(fake_logits, Variable(torch.zeros(fake_logits.size())).cuda())
            total_error = real_loss + fake_loss
            fake_loss.backward()
            D_Optimizer.step()

            #   2. Train G on D's response (but do not train D on these labels)
            G.zero_grad()

            fake_input = Variable(torch.randn(args.batch_size, 96)).cuda()
            fake_image = G(fake_input)
            g_fake_logits = D(fake_image)
            g_fake_loss = BCE(g_fake_logits, Variable(torch.ones(g_fake_logits.size())).cuda())
            g_fake_loss.backward()
            G_Optimizer.step()

            if (index % 100 == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, index, total_error.data[0], g_fake_loss.data[0]))
        print('Saving results ============>')
        fake_input = Variable(torch.randn(2, 96)).cuda()
        fake_image = G(fake_input)
        imgs_numpy = fake_image.data[0].cpu().numpy()
        plt.imshow((imgs_numpy / 2 + 0.5).reshape(64, 64), cmap='gray', aspect='equal')
        if not os.path.exists(args.result_save):
            os.mkdir(args.result_save)
        save_fn = args.result_save + 'MNIST_DCGAN_G_epoch_{:d}'.format(epoch) + '.png'
        plt.savefig(save_fn)

        if not os.path.exists(args.model_save):
            os.mkdir(args.model_save)
        torch.save(D.state_dict(), '%s/D_epoch_%d.pth' % (args.model_save, epoch))
        torch.save(G.state_dict(), '%s/G_epoch_%d.pth' % (args.model_save, epoch))
