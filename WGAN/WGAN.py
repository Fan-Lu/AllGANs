from torchvision import transforms, datasets
import torch
import argparse
import torch.optim as optim
from torch.autograd import Variable

import models.mlp as mlp

import matplotlib.pyplot as plt
import os

paraser = argparse.ArgumentParser(description='WGAN')
paraser.add_argument('--batch_size', default=128, type=int, help='batch size')
paraser.add_argument('--num_epoch', default=20, type=int, help='num epoch')
paraser.add_argument('--lrD', default=0.00005, type=int, help='D learning rate')
paraser.add_argument('--lrG', default=0.00005, type=int, help='G learning rate')
paraser.add_argument('--image_size', default=64, type=int, help='G learning rate')
paraser.add_argument('--data_dir', metavar='DIR', help='path to data', default='../Data/MNIST_data/')
paraser.add_argument('--model_save', metavar='DIR', help='path to store results', default='model_saved/')
paraser.add_argument('--result_save', metavar='DIR', help='path to store results', default='result_saved/')

args = paraser.parse_args()


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


D = mlp.MLP_D().cuda()
G = mlp.MLP_G().cuda()

optimizerD = optim.RMSprop(D.parameters(), lr=args.lrD)
optimizerG = optim.RMSprop(D.parameters(), lr=args.lrG)


one = torch.FloatTensor([1]).cuda()
mone = one * -1


fixed_noise = torch.FloatTensor(2, 100).normal_(0, 1).cuda()

if __name__ == '__main__':
    for epoch in range(args.num_epoch):
        data_iter = iter(trainloader)
        i = 0
        while i < len(trainloader):
            j = 0
            while j < 25 and i < len(trainloader):
                j += 1

                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01)

                inputs, labels = data_iter.next()
                i += 1
                real_image = Variable(inputs).cuda()
                #1. Train with D

                for p in D.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                #1A. Train on real
                D.zero_grad()
                real_loss_D = D(real_image)
                real_loss_D.backward(one)

                #1B. Train on fake
                fake_input = Variable(torch.randn(args.batch_size, 100)).cuda()
                fake_image = G(fake_input)
                fake_loss_D = D(fake_image)
                fake_loss_D.backward(mone)
                loss_D = real_loss_D - fake_loss_D
                optimizerD.step()


            #2. Train with G
            for p in G.parameters():
                p.requires_grad = False  # save time

            G.zero_grad()
            fake_input_G = Variable(torch.randn(args.batch_size, 100)).cuda()
            fake_image_G = G(fake_input_G)
            fake_loss_G = D(fake_image_G)
            fake_loss_G.backward(one)
            optimizerG.step()

            if (i % 10 == 0):
                print('Epoch: {}, Iter: {}, loss_D: {:.4}, loss_G:{:.4}'.format(epoch, i, loss_D.data[0], fake_loss_G.data[0]))

        print('Saving results ============>')
        fake_input = Variable(fixed_noise)
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












