import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, os
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools

parase = argparse.ArgumentParser(description='StackGAN')
parase.add_argument('--batch_size', default=128, type=int, help='batchsize')
parase.add_argument('--data_dir', default='../Data/MNIST_data/', help='path to data')
parase.add_argument('--image_size', default=64, help='image size')


args = parase.parse_args()


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


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def get_digits(digits, labels):
    for i in range(args.batch_size):
        digits[i][labels[i]] = 1

    return digits


# G(z)
class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(110, d*8, 4, 1, 0)
        self.deconv1_bn = nn.InstanceNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.InstanceNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.InstanceNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.InstanceNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        # 128*1*64*64

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, noise, x_digits):
        # noise: bs*100*1*1, x_digits: bs*10
        x = torch.cat((noise, x_digits), dim=1)
        x = x.view(-1, 110, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv1_bn = nn.InstanceNorm2d(d)

        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.InstanceNorm2d(d*2)

        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.InstanceNorm2d(d*4)

        self.conv4 = nn.Conv2d(d*4, d*2, 4, 2, 1)
        self.conv4_bn = nn.InstanceNorm2d(d*2)

        # bs*1024*4*4
        self.conv5 = nn.Conv2d(d*2, 11, 4, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.leaky_relu(x)

        x = self.conv5(x)
        # bsx11x1x1
        x = x.squeeze()

        score = F.sigmoid(x[:, 0])
        digits = x[:, 1:]
        return score, digits

G = Generator().cuda()
D = Discriminator().cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
optimizer_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
optimizer_info = torch.optim.Adam(itertools.chain(G.parameters(), D.parameters()), lr=1e-3, betas=(0.9, 0.999))
criterion_info = nn.CrossEntropyLoss()

noise = torch.FloatTensor(args.batch_size, 100)
fixed_noise = Variable(torch.FloatTensor(2, 100).normal_(0, 1))

true_label = Variable(torch.ones(args.batch_size)).cuda()
false_label = Variable(torch.zeros(args.batch_size)).cuda()
x_digits = torch.zeros(args.batch_size, 10)

if __name__ == '__main__':
    for epoch in range(20):
        for index, data in enumerate(trainloader):
            real_img, real_labels = data
            real_img = Variable(real_img).cuda()
            #   1. Update Discriminator
            #   1A. Train D with real image
            D.zero_grad()
            real_score_D, _ = D(real_img)
            real_loss_D = criterion(real_score_D, true_label)
            real_loss_D.backward(retain_graph=True)

            #   1B. Train D with fake image
            x_digits_D = Variable(get_digits(x_digits, real_labels)).cuda()
            input_D = Variable(noise.normal_(0, 1)).cuda()
            fake_imge_D = G(input_D, x_digits_D)
            fake_score_D, _ = D(fake_imge_D)
            fake_loss_D = criterion(fake_score_D, false_label)
            fake_loss_D.backward(retain_graph=True)
            total_error = real_loss_D + fake_loss_D
            optimizer_D.step()

            #   2. Update Generator
            G.zero_grad()
            x_digits_G = Variable(get_digits(x_digits, real_labels)).cuda()
            input_G = Variable(noise.normal_(0, 1)).cuda()
            fake_imge_G = G(input_G, x_digits_G)
            fake_score_G, fake_digits_G = D(fake_imge_G)
            fake_loss_G = criterion(fake_score_G, true_label)
            fake_loss_G.backward(retain_graph=True)
            optimizer_G.step()

            #   info loss
            info_loss = criterion_info(fake_digits_G, torch.max(x_digits_D, 1)[1])
            info_loss.backward(retain_graph=True)
            optimizer_info.step()

            if (index % 10 == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}, Info: {:.4}'.format(epoch, index, total_error.data[0], fake_loss_G.data[0], info_loss.data[0]))
        print('Saving results ============>')
        real = torch.FloatTensor(torch.zeros(2))
        x_digits_D = Variable(get_digits(x_digits, real)).cuda()
        fake_image = G(fixed_noise, x_digits_D)
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


