# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
#sys.path.insert(0,'../../ki_nc/ki-nc/lava-dl')
import lava.lib.dl.slayer as slayer

#from optimizer import Nadam


def augment(event):
    #same as in: https://arxiv.org/pdf/2008.01151.pdf
    x_shift = 8 #20  
    y_shift = 8 #20
    theta = 10 #90
    xjitter = np.random.randint(2*x_shift) - x_shift
    yjitter = np.random.randint(2*y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event

def downsample_events(event, factor):
    event.x = event.x // factor
    event.y = event.y // factor
    return event

class DVSPlugsDataset(Dataset):
    """DVS Plugs dataset class

    Parameters
    ----------
    path : str, optional
        path of dataset root, by default '/home/datasets/dvs_gesture_bs2'
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 1
    sample_length : int, optional
        length of sample data, by default 300
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default None.
    random_shift: bool, optional
        shift input sequence randomly in time. By default True.
    data_format: str, optional
        data format of the input data, either 'bs2' or 'npy'. By default 'bs2'.
    ds_factor: int, optional
        factor to downsample event input. By default 1.
    """
    def __init__(
        self, path='/home/neumeier/eleanor/dvs_port_bs2',
        train=True,
        sampling_time=1, sample_length=300,
        transform=None, random_shift=True, data_format='bs2', ds_factor=1, uniform_polarity=True,
        lava=False
    ):
        super(DVSPlugsDataset, self).__init__()
        self.path = path
        if train:
            dataParams = np.loadtxt(self.path + '/old_labels/train.txt').astype('int')
        else:
            dataParams = np.loadtxt(self.path + '/old_labels/test.txt').astype('int')

        self.samples = dataParams[:, 0]
        self.labels = dataParams[:, 1]
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length/sampling_time)
        self.transform = transform
        self.random_shift = random_shift
        self.data_format = data_format
        self.ds_factor = ds_factor
        self.uni_pol = uniform_polarity
        self.lava = lava

    def __getitem__(self, i):
        i = 10 * i
        label = self.labels[i]
        # dataset in .bs2-format
        if self.data_format == 'bs2':
            filename = self.path + '/' + str(self.samples[i]) + '.bs2'
            event = slayer.io.read_2d_spikes(filename)
        # dataset in .npy-format
        elif self.data_format == 'npy':
            filename = self.path + '/' + str(self.samples[i]) + '.npy'
            event = slayer.io.read_np_spikes(filename, time_unit=1e-3)
        else:
            print('No correct data format!!! -> Only bs2 and npy valid')

        if self.transform is not None:
            event = self.transform(event)
        # downsample event input
        event = downsample_events(event, self.ds_factor)
        h_inp = int(128 / self.ds_factor)
        w_inp = int(128 / self.ds_factor)


        if self.random_shift:
            spike = event.fill_tensor(np.zeros((2, h_inp, w_inp, self.num_time_bins)), sampling_time=self.sampling_time,
                                      random_shift=True)
        else:
            spike = event.fill_tensor(np.zeros((2, h_inp, w_inp, self.num_time_bins)), sampling_time=self.sampling_time)

        if self.uni_pol:
            spike = np.logical_or(spike[0,:,:,:],spike[1,:,:,:])#either polarity is active then its one
            #print(spike.shape)
            spike = np.expand_dims(spike, axis=0)
            #print(spike.shape)

        if self.lava:
            # convert to WHC format
            spike = np.moveaxis(spike, 0, 2)
            spike = np.moveaxis(spike, 0, 1)
            spike = spike.astype(np.int32)
            label = label.astype(np.int32)
        else:
            spike = torch.from_numpy(spike)

        return spike, label

    def __len__(self):
        return len(self.samples)


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': False,
        }
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.05), }
        #neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
            #slayer.block.cuba.Input(neuron_params=neuron_params_drop),
            slayer.block.cuba.Pool(neuron_params=neuron_params_drop, kernel_size=4, stride=4),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=2, out_features=16, kernel_size=5,
                                   padding=2, delay=False, weight_scale=1, weight_norm=True),
            slayer.block.cuba.Pool(neuron_params=neuron_params_drop, kernel_size=2, stride=2),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=16, out_features=32, kernel_size=3,
                                   padding=1, delay=False, weight_scale=1, weight_norm=True),
            slayer.block.cuba.Pool(neuron_params=neuron_params_drop, kernel_size=2, stride=2),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(neuron_params=neuron_params_drop, in_neurons=(8*8*32), out_neurons=512,
                                    delay=False, weight_scale=1, weight_norm=True),
            slayer.block.cuba.Dense(neuron_params=neuron_params_drop, in_neurons=512, out_neurons=4,
                                    weight_scale=1, weight_norm=True),
        ])

    def forward(self, spike):
        # print(spike.size())
        count = []
        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)
        ).to(spike.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename, add_input_layer=False, input_dims=[2, 128, 128]):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        if add_input_layer:
            input_layer = layer.create_group(f'{0}')
            input_layer.create_dataset('shape', data=np.array(input_dims))
            input_layer.create_dataset('type', (1, ), 'S10', ['input'.encode('ascii', 'ignore')])
        for i, b in enumerate(self.blocks):
            if add_input_layer:
                b.export_hdf5(layer.create_group(f'{i+1}'))
            else:
                b.export_hdf5(layer.create_group(f'{i}'))
        # add simulation key for nxsdk
        sim = h.create_group('simulation')
        sim.create_dataset('Ts', data=1)
        sim.create_dataset('tSample', data=300)


class LavaNetwork(torch.nn.Module):
    def __init__(self):
        super(LavaNetwork, self).__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': False,
        }
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.05), }
        #neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
            #slayer.block.cuba.Input(neuron_params=neuron_params_drop),
            #slayer.block.cuba.Pool(neuron_params=neuron_params_drop, kernel_size=4, stride=4),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=1, out_features=16, kernel_size=5,
                                   padding=2, delay=False, weight_norm=False),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=16, out_features=16, kernel_size=2,
                                   padding=0, stride=2, delay=False, weight_norm=False),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=16, out_features=32, kernel_size=3,
                                   padding=1, delay=False, weight_norm=False),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=32, out_features=32, kernel_size=2,
                                   padding=0, stride=2, delay=False, weight_norm=False),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(neuron_params=neuron_params_drop, in_neurons=(8*8*32), out_neurons=512,
                                    delay=False, weight_norm=False),
            slayer.block.cuba.Dense(neuron_params=neuron_params_drop, in_neurons=512, out_neurons=4,
                                    weight_scale=1, weight_norm=False),
        ])

    def forward(self, spike):
        # print(spike.size())
        count = []
        for block in self.blocks:
            #print(block)
            spike = block(spike)
            #print(spike)
            count.append(torch.mean(spike).item())
            #print(count)
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)
        ).to(spike.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename, add_input_layer=False, input_dims=[2, 32, 32]):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        if add_input_layer:
            input_layer = layer.create_group(f'{0}')
            input_layer.create_dataset('shape', data=np.array(input_dims))
            input_layer.create_dataset('type', (1, ), 'S10', ['input'.encode('ascii', 'ignore')])
        for i, b in enumerate(self.blocks):
            if add_input_layer:
                b.export_hdf5(layer.create_group(f'{i+1}'))
            else:
                b.export_hdf5(layer.create_group(f'{i}'))
        # add simulation key for nxsdk
        sim = h.create_group('simulation')
        sim.create_dataset('Ts', data=1)
        sim.create_dataset('tSample', data=300)

class SmallLavaNetwork(torch.nn.Module):
    def __init__(self):
        super(SmallLavaNetwork, self).__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': False,
        }
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.05), }
        #neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
            #slayer.block.cuba.Input(neuron_params=neuron_params_drop),
            #slayer.block.cuba.Pool(neuron_params=neuron_params_drop, kernel_size=4, stride=4),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=1, out_features=16, kernel_size=5,
                                   padding=0, stride=2, delay=False, weight_norm=False),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=16, out_features=32, kernel_size=5,
                                   padding=0, stride=1, delay=False, weight_norm=False),
            slayer.block.cuba.Conv(neuron_params=neuron_params_drop, in_features=32, out_features=32, kernel_size=3,
                                   padding=0, stride=1, delay=False, weight_norm=False),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(neuron_params=neuron_params_drop, in_neurons=(8*8*32), out_neurons=128,
                                    delay=False, weight_norm=False),
            slayer.block.cuba.Dense(neuron_params=neuron_params_drop, in_neurons=128, out_neurons=4,
                                    weight_scale=1, weight_norm=False),
        ])

    def forward(self, spike):
        # print(spike.size())
        count = []
        for block in self.blocks:
            #print(block)
            spike = block(spike)
            #print(spike)
            count.append(torch.mean(spike).item())
            #print(count)
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)
        ).to(spike.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
        ]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename, add_input_layer=False, input_dims=[2, 32, 32]):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        if add_input_layer:
            input_layer = layer.create_group(f'{0}')
            input_layer.create_dataset('shape', data=np.array(input_dims))
            input_layer.create_dataset('type', (1, ), 'S10', ['input'.encode('ascii', 'ignore')])
        for i, b in enumerate(self.blocks):
            if add_input_layer:
                b.export_hdf5(layer.create_group(f'{i+1}'))
            else:
                b.export_hdf5(layer.create_group(f'{i}'))
        # add simulation key for nxsdk
        sim = h.create_group('simulation')
        sim.create_dataset('Ts', data=1)
        sim.create_dataset('tSample', data=300)
                    

if __name__ == '__main__':
    trained_folder = 'Trained_new_2'
    os.makedirs(trained_folder, exist_ok=True)

    #torch.manual_seed(0)

    device = torch.device('cpu')
    # device = torch.device('cuda')

    # test conv
    #test_conv = nn.Conv2d(2, 10, kernel_size=5, padding=2, stride=1, bias=False).to(device)


    # one GPU
    net = SmallLavaNetwork().to(device)

    # two GPUs in parallel
    # print(torch.cuda.device_count())
    # net = Network_Decolle().to(device)
    # net.forward(torch.rand(1, 2, 128, 128, 1).to(device))
    # net = nn.DataParallel(net, device_ids=[0, 1])
    # net.to(device)
    print('Init done!')

    #optimizer = Nadam(net.parameters(), lr=0.003, amsgrad=True)
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #steps = [50, 200, 500]

    training_set = DVSPlugsDataset(path='slayer_training/dvs_port_ext_bs2',
                                     train=True, transform=augment, random_shift=True, ds_factor=4)
    testing_set = DVSPlugsDataset(path='slayer_training/dvs_port_ext_bs2',
                                    train=False, transform=augment, random_shift=False, ds_factor=4)

    input, label = training_set[0]
    input = input.unsqueeze(dim=0).to(device)
    print(input.shape)
    #out_test = test_conv(input[:,:,:,:,0])
    #print(out_test.shape)
    output, _ = net(input)
    print(output.shape)

    train_loader = DataLoader(dataset=training_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    error = slayer.loss.SpikeRate(
           true_rate=0.2, false_rate=0.03, reduction='sum').to(device)
    #error = slayer.loss.SpikeMax(mode='logsoftmax').to(device)

    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
            net, error, optimizer, stats,
            classifier=slayer.classifier.Rate.predict, count_log=True
        )
    #torch.autograd.set_detect_anomaly(True)

    epochs = 100 #1000
    print("start training")
    for epoch in range(epochs):
        #if epoch in steps:
            #assistant.reduce_lr(factor=10 / 3)
        for i, (inp, label) in enumerate(train_loader):  # training loop
            output, count = assistant.train(inp, label)
            #stats.print(epoch, iter=i, dataloader=train_loader)

        for i, (inp, label) in enumerate(test_loader):  # testing loop
            output, count = assistant.test(inp, label)
            #stats.print(epoch, iter=i, dataloader=test_loader)

        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), trained_folder + '/network.pt')
            #net.export_hdf5(trained_folder + '/network.net', add_input_layer=True, input_dims=[2, 128, 128])
        stats.update()
        stats.save(trained_folder + '/')
        stats.plot(path=trained_folder + '/')
        print("Epoch " + str(epoch) + " finished")
        net.grad_flow(trained_folder + '/')
        
    #net.export_hdf5(trained_folder + '/network.net', add_input_layer=True, input_dims=[2, 32, 32])
        
