# this is the model from 4/1, in case you want to load it back

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1) # changed things
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)
        # self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        # self.conv4_bn = nn.BatchNorm2d(d*8)
        # self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))
        # x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # x = F.sigmoid(self.conv5(x))
        return x

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*4, 4, 1, 0) # changed things
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 3, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1) # 1 to 3, 4 to 3
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        # self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1) # 1 to 3

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = torch.tanh(self.deconv5(x)) # deprecated
        return x

D = discriminator(32)
G = generator(32)
D.load_state_dict(torch.load('MNIST_ASGAN_results/discriminator_param.pkl'))
G.load_state_dict(torch.load('MNIST_ASGAN_results/generator_param.pkl'))


def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    # z_ = Variable(z_.cuda(), volatile=True)
    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ti = test_images[k].cpu().data.numpy().transpose(1,2,0) # all 3 channels!
        ax[i, j].imshow(ti, cmap='gray')
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


show_result(20, show=True, save=False)