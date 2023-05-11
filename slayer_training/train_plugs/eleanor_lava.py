import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import logging
from typing import Dict, Tuple
import sys

from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.proc.io.encoder import Compression
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore

from lava.lib.dl import netx
from eleanor_train import DVSPlugsDataset, LavaNetwork, SmallLavaNetwork


class CustomRunConfig(Loihi2SimCfg):
    def select(self, proc, proc_models):
        # customize run config to always use float model for io.sink.RingBuffer
        if isinstance(proc, io.sink.RingBuffer):
            return io.sink.PyReceiveModelFloat
        else:
            return super().select(proc, proc_models)

class OutputProcess(AbstractProcess):
    """Process to gather spikes from 12 output LIF neurons and interpret the
    highest spiking rate as the classifier output"""

    def __init__(self, num_classes: int = 4,
                 num_samples: int = 1,
                 len_per_sample: int = 100):
        super().__init__()
        shape = (num_classes,)
        self.num_samples = Var(shape=(1,), init=num_samples)
        self.spikes_in = InPort(shape=shape)
        self.label_in = InPort(shape=(1,))
        self.spikes_accum = Var(shape=shape)  # Accumulated spikes for classification
        self.num_steps_per_image = Var(shape=(1,), init=len_per_sample)
        self.pred_labels = Var(shape=(num_samples,))
        self.gt_labels = Var(shape=(num_samples,))

@implements(proc=OutputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputProcessModel(PyLoihiProcessModel):
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    num_samples: int = LavaPyType(int, int, precision=32)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
    num_steps_per_image: int = LavaPyType(int, int, precision=32)
    pred_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    gt_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.current_img_id = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        #print(f'ts: {self.time_step}')
        #sys.stdout.flush()
        if (self.time_step) % self.num_steps_per_image == 0: # and self.time_step > 1:
            #print("Out post mgmt")
            #sys.stdout.flush()
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above
        returns True.
        """
        #gt_label = self.label_in.recv()
        pred_label = np.argmax(self.spikes_accum)
        #print(self.gt_labels[self.current_img_id])
        #sys.stdout.flush()
        #self.gt_labels[self.current_img_id] = gt_label
        self.pred_labels[self.current_img_id] = pred_label
        self.current_img_id += 1
        self.spikes_accum = np.zeros_like(self.spikes_accum)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        print(f'ts: {self.time_step}')
        sys.stdout.flush()
        spk_in = self.spikes_in.recv()
        gt_label = self.label_in.recv()
        #print(gt_label)
        #sys.stdout.flush()
        self.gt_labels[self.current_img_id] = gt_label
        self.spikes_accum = self.spikes_accum + spk_in
        print(self.spikes_accum)
        sys.stdout.flush()


if __name__ == '__main__':
    # parameters
    num_samples = 16  # len(testing_set)
    steps_per_sample = 128 #1024
    num_steps = num_samples * steps_per_sample #+ 1

    # check if Loihi2 is available
    from lava.utils.system import Loihi2
    Loihi2.preferred_partition = 'kp'
    loihi2_is_available = Loihi2.is_loihi2_available
    if loihi2_is_available:
        from lava.proc import embedded_io as eio
        print(f'Running on {Loihi2.partition}')
    else:
        print("Loihi2 compiler is not available in this system. "
              "This tutorial will execute on CPU backend.")

    # instantiate network
    #net = SmallLavaNetwork()
    #output, count = net(torch.rand(1, 1, 32, 32, 1))
    #net.load_state_dict(torch.load('Trained_ext_small_lava_merged_pol/network.pt', map_location=torch.device('cpu')))
    #net.export_hdf5('Trained_ext_small_lava_merged_pol/network_small.net')

    # import trained network
    net = netx.hdf5.Network(net_config='Trained_ext_small_lava_merged_pol/network_small.net', input_shape=(32, 32, 1),
                            reset_interval=steps_per_sample, reset_offset=1)
    print(net)
    readout_offset = len(net.layers)

    # dataset
    training_set = DVSPlugsDataset(path='../dvs_port_ext_bs2',
                                   train=True, ds_factor=4, lava=True)
    testing_set = DVSPlugsDataset(path='../dvs_port_ext_bs2',
                                  train=False, ds_factor=4, lava=True)
    inp, gt = testing_set[0]

    dataloader = io.dataloader.SpikeDataloader(
        dataset=testing_set,
        interval=steps_per_sample)

    # connect in- and output
    out_logger = io.sink.RingBuffer(shape=(4,), buffer=num_steps)
    out_proc = OutputProcess(
        num_classes=4,
        num_samples=num_samples,
        len_per_sample=steps_per_sample)

    # customize run config
    loihi2sim_exception_map = {
        io.sink.RingBuffer: io.sink.PyReceiveModelFloat,}
    #run_config = CustomRunConfig(select_tag='fixed_pt')
    if loihi2_is_available:
        # connect with embedded processor adapters
        in_adapter = eio.spike.PyToN3ConvAdapter(shape=dataloader.s_out.shape, num_message_bits=0, compression=Compression.DENSE)
        dataloader.s_out.connect(in_adapter.inp)
        in_adapter.out.connect(net.in_layer.inp)
        out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape, num_message_bits=0)
        net.out.connect(out_adapter.inp)
        out_adapter.out.connect(out_proc.spikes_in)
        run_config = Loihi2HwCfg()
    else:
        dataloader.s_out.connect(net.inp)
        net.out.connect(out_proc.spikes_in)
        run_config = Loihi2SimCfg(select_tag='fixed_pt', exception_proc_model_map=loihi2sim_exception_map)

    #net.out.connect(out_logger.a_in) #(net.out_layer.neuron.v)
    dataloader.ground_truth.connect(out_proc.label_in)

    net._log_config.level = logging.INFO

    # run network
    net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)

    # gather results
    #current1 = net.layers[0].neuron.u.get()
    #print(f"\nMean: {np.mean(current1)}\n"
    #      f"Min : {np.min(current1)}\n"
    #      f"Max : {np.max(current1)}")
    #results = out_logger.data.get()
    #print(results)
    ground_truth = out_proc.gt_labels.get().astype(np.int32)
    predictions = out_proc.pred_labels.get().astype(np.int32)
    print(ground_truth)

    # stop execution
    net.stop()

    accuracy = np.sum(ground_truth == predictions) / ground_truth.size * 100

    print(f"\nGround truth: {ground_truth}\n"
          f"Predictions : {predictions}\n"
          f"Accuracy    : {accuracy}")



