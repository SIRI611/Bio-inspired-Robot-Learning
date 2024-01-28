import copy
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class FastRangeAdaptor:

    def __init__(self, max_output=0.5, min_output=-0.5, factor=1, bias=0, t=0):
        self.max_output = max_output
        self.min_output = min_output
        self.factor = factor
        self.bias = bias
        self.max_input = None
        self.min_input = None
        self.output = None
        self.t = t
        self.input = None

    def step_dynamics(self, dt, current_input):
        self.t += dt
        self.input = current_input
        if self.max_input is None:
            self.max_input = current_input
        if self.min_input is None:
            self.min_input = current_input
        if_updated = False
        if self.max_input < current_input:
            self.max_input = current_input
            if_updated = True
        elif self.min_input > current_input:
            self.min_input = current_input
            if_updated = True
        if if_updated:
            self.factor = (self.max_output - self.min_output) / (self.max_input - self.min_input)
            self.bias = ((self.max_output + self.min_output) - (self.max_input + self.min_input)) / 2
        self.output = (current_input - self.bias) * self.factor

    def update(self):
        pass

    def update_parameters(self, parameters):
        parameters = np.array(parameters)
        self.factor = parameters[:, 0]
        self.bias = parameters[:, 1]

    def init_recording(self):
        self.trace = {
            't': deque(),
            'factor': deque(),
            'bias': deque(),
            'input_value': deque(),
            'output': deque()
        }

    def recording(self):
        for key in self.trace:
            self.trace[key].append(copy.deepcopy(getattr(self, key)))

    def plot(self, neuron_id, down_sample_rate=1):
        figure_dict = dict()
        figure_dict[str(neuron_id)] = plt.figure()
        index = np.s_[::down_sample_rate, neuron_id]

        plt.plot(np.array(self.trace['t'])[::down_sample_rate],
                 np.vstack((np.array(self.trace['factor'])[index],
                            np.array(self.trace['bias'])[index],
                            np.array(self.trace['input_value'])[index],
                            np.array(self.trace['output'])[index])).T)

        return figure_dict
