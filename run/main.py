import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from spikingjelly import visualizing
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import time
import math

if sys.platform != 'win32':
    import readline
import numpy as np

torch.backends.cudnn.benchmark = True
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


class Net(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0, mode='latency', warmup=False):
        super().__init__()
        self.T = T
        self.deafult_T = 8
        self.surrogate_function = surrogate.ATan_binary()
        self.mode=mode
        self.warmup=warmup

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        )

        self.conv = nn.Sequential(
            # neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), monitor_state=True),
            neuron.OneSpikeIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), monitor_state=True),

            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            # neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), monitor_state=True),
            neuron.OneSpikeIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), monitor_state=True),
            # nn.MaxPool2d(2, 2)  # 14 * 14
            layer.FirstSpikePool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)  # 14 * 14
        )
        if self.mode=='potential' or (self.mode=='latency' and self.warmup):
            Readout_v_threshold = math.inf
        else:
            Readout_v_threshold = v_threshold
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 100, bias=False),
            # neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), monitor_state=True),
            neuron.OneSpikeIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), monitor_state=True),
            nn.Linear(100, 10, bias=False),
            neuron.IFNode(v_threshold=Readout_v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), monitor_state=True)
            # neuron.OneSpikeIFNode(v_threshold=Readout_v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan(), monitor_state=True)
            # layer.SynapseFilter(tau=tau, learnable=False)
        )

        for module in self.static_conv:
            if hasattr(module, 'weight'):
                # with torch.no_grad():
                module.weight = nn.Parameter(module.weight*(self.deafult_T/self.T))
                # module.weight = nn.Parameter(torch.mul(module.weight,(self.deafult_T/self.T)))
        for module in self.conv:
            if hasattr(module, 'weight'):
                module.weight = nn.Parameter(module.weight * (self.deafult_T/self.T))
        for module in self.fc:
            if hasattr(module, 'weight'):
                module.weight = nn.Parameter(module.weight * (self.deafult_T/self.T))


    def calc_reg_loss(self, reg_loss_list, step):
        reg_loss_list_tmp = reg_loss_list

        reg_loss_list_tmp[0][step] = self.conv[0].spike.sum()/(np.prod(self.conv[0].spike.shape))
        reg_loss_list_tmp[1][step] = self.conv[2].spike.sum()/(np.prod(self.conv[2].spike.shape))
        reg_loss_list_tmp[2][step] = self.fc[2].spike.sum()/(np.prod(self.fc[2].spike.shape))
        reg_loss_list_tmp[3][step] = self.fc[4].spike.sum()/(np.prod(self.fc[4].spike.shape))
        return reg_loss_list_tmp


    def forward(self, x):
        reg_loss_list = [torch.zeros((self.T)).to(x.device)]*4

        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x))
        out_potential_counter = self.fc[4].v
        latency_score = self.surrogate_function(out_spikes_counter-0.5)

        # reg_loss_list = self.calc_reg_loss(reg_loss_list, 0)
        reg_loss_list[0][0] = (self.conv[0].spike.sum() / (np.prod(self.conv[0].spike.shape)))
        reg_loss_list[1][0] = (self.conv[2].spike.sum() / (np.prod(self.conv[2].spike.shape)))
        reg_loss_list[2][0] = (self.fc[2].spike.sum() / (np.prod(self.fc[2].spike.shape)))
        reg_loss_list[3][0] = (self.fc[4].spike.sum() / (np.prod(self.fc[4].spike.shape)))

        for t in range(1, self.T):
            out_spikes_counter = out_spikes_counter+self.fc(self.conv(x))
            out_potential_counter = out_potential_counter + self.fc[4].v
            latency_score = latency_score + self.surrogate_function(out_spikes_counter-0.5)

            # reg_loss_list = self.calc_reg_loss(reg_loss_list, t)
            reg_loss_list[0][t] = (self.conv[0].spike.sum() / (np.prod(self.conv[0].spike.shape)))
            reg_loss_list[1][t] = (self.conv[2].spike.sum() / (np.prod(self.conv[2].spike.shape)))
            reg_loss_list[2][t] = (self.fc[2].spike.sum() / (np.prod(self.fc[2].spike.shape)))
            reg_loss_list[3][t] = (self.fc[4].spike.sum() / (np.prod(self.fc[4].spike.shape)))
            # reg_loss_list = torch.stack(reg_loss_list, dim=0)

        reg_loss=0
        for item in reg_loss_list :
            reg_loss += item.max()


        if self.mode=='latency' and not(self.warmup):
            return latency_score / self.T, reg_loss
        elif self.mode=='potential' or (self.mode=='latency' and self.warmup):
            return out_potential_counter / self.T, reg_loss
        elif self.mode == 'frequency':
            return out_spikes_counter / self.T, reg_loss



def plot_spikes(net, data_loader, device, nb_plt, batch_size):
    for img, label in data_loader:
        img = img.to(device)
        net(img)
        break
    batch_idx = np.random.choice(batch_size, nb_plt, replace=False)
    for module in net.conv:
        if isinstance(module, neuron.OneSpikeIFNode) or isinstance(module, neuron.IFNode):
            spike_array = np.squeeze(np.asarray(module.monitor['s']))[:,:,0,14,14]
            s_t_array = spike_array.T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
            visualizing.plot_1d_spikes(spikes=s_t_array, title='Spikes of Neurons', xlabel='Simulating Step',
                                       ylabel='Neuron Index', int_x_ticks=True, int_y_ticks=True,
                                       plot_firing_rate=True, firing_rate_map_title='Firing Rate', dpi=200)
    for module in net.fc:
        if isinstance(module, neuron.OneSpikeIFNode) or isinstance(module, neuron.IFNode):
            spike_array = np.squeeze(np.asarray(module.monitor['s']))[:, :, 0]
            s_t_array = spike_array.T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
            visualizing.plot_1d_spikes(spikes=s_t_array, title='Spikes of Neurons', xlabel='Simulating Step',
                                       ylabel='Neuron Index', int_x_ticks=True, int_y_ticks=True,
                                       plot_firing_rate=True, firing_rate_map_title='Firing Rate', dpi=200)
    plt.show()


def main():
    '''
    * :ref:`API in English <conv_fashion_mnist.main-en>`

    .. _conv_fashion_mnist.main-cn:

    :return: None

    使用卷积-全连接的网络结构，进行Fashion MNIST识别。这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。会将训练过
    程中测试集正确率最高的网络保存在 ``tensorboard`` 日志文件的同级目录下。这个目录的位置，是在运行 ``main()``
    函数时由用户输入的。

    训练100个epoch，训练batch和测试集上的正确率如下：

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
        :width: 100%

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
        :width: 100%

    * :ref:`中文API <conv_fashion_mnist.main-cn>`

    .. _conv_fashion_mnist.main-en:

    The network with Conv-FC structure for classifying Fashion MNIST. This function initials the network, starts training
    and shows accuracy on test dataset. The net with the max accuracy on test dataset will be saved in
    the root directory for saving ``tensorboard`` logs, which is inputted by user when running the ``main()``  function.

    After 100 epochs, the accuracy on train batch and test dataset is as followed:

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
        :width: 100%

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
        :width: 100%
    '''
    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    # dataset_dir = input('输入保存Fashion MNIST数据集的位置，例如“./”\n input root directory for saving Fashion MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    # learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“8”\n input simulating steps, e.g., "8": '))
    # tau = float(input('输入LIF神经元的时间常数tau，例如“2.0”\n input membrane time constant, tau, for LIF neurons, e.g., "2.0": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”\n input training epochs, e.g., "100": '))
    warmup_epochs = int(input('input warmup epochs, e.g., "50": '))
    # log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')

    # device = "cpu"
    dataset_dir = "./dataset/"
    # batch_size = 2
    learning_rate = 1e-3
    # T = 8
    tau = 2.0
    # train_epoch = 10
    # warmup_epochs = 5
    log_dir = "./log/"

    print("batch_size = ", batch_size)
    print("learning_rate = ", learning_rate)
    print("T = ", T)
    print("tau = ", tau)
    print("train_epoch = ", train_epoch)
    print("warmup_epochs = ", warmup_epochs)

    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    # 初始化网络
    net = Net(tau=tau, T=T, mode='latency', warmup=(warmup_epochs>0)).to(device)
    # net = torch.load(log_dir + '/net_max_acc.pt')
    plot_spikes(net, test_data_loader, device, nb_plt=batch_size, batch_size=batch_size)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    log_softmax_fn = torch.nn.LogSoftmax(dim=1)
    loss_fn = torch.nn.NLLLoss()
    train_times = 0
    max_test_accuracy = 0
    for epoch in range(train_epoch):
        if epoch<warmup_epochs :
            net.warmup=True
        else :
            net.warmup=False
            net.fc[-1].v_threshold=1.0
        net.train()
        t_start = time.perf_counter()
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            optimizer.zero_grad()

            out_spikes_counter_frequency, reg_loss = net(img)

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            print("loss = ", loss, ", reg_loss = ", reg_loss)
            loss += reg_loss
            # log_p_y = log_softmax_fn(out_spikes_counter_frequency)
            # loss = loss_fn(log_p_y, label)
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            if train_times % 256 == 0:
                writer.add_scalar('train_accuracy', accuracy, train_times)
            train_times += 1
        t_train = time.perf_counter() - t_start
        net.eval()
        t_start = time.perf_counter()
        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.to(device)
                out_spikes_counter_frequency, reg_loss = net(img)

                correct_sum += (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            test_accuracy = correct_sum / test_sum
            t_test = time.perf_counter() - t_start
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            if max_test_accuracy < test_accuracy:
                max_test_accuracy = test_accuracy
                print('saving net...')
                torch.save(net, log_dir + '/net_max_acc.pt')
                print('saved')

        print(
            'epoch={}, t_train={}, t_test={}, device={}, dataset_dir={}, batch_size={}, learning_rate={}, T={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
                epoch, t_train, t_test, device, dataset_dir, batch_size, learning_rate, T, log_dir, max_test_accuracy,
                train_times))

    plot_spikes(net, test_data_loader, device, nb_plt=batch_size, batch_size=batch_size)



if __name__ == '__main__':
    main()




