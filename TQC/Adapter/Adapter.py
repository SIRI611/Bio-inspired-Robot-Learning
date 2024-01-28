import copy
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class Adapter:

    def __init__(self, last_input=None, current_input=None, factor=1,
                 if_inertia=True, decay_rate=0.1, current_diff=0, last_diff=0):
        self.last_input = last_input
        self.current_input = current_input
        self.factor = factor
        # assert 0<atten_coeff<=1, "Attenuation coefficient should smaller or equal to 1 and larger than 0"
        # self.atten_coeff = atten_coeff #Attenuation coefficient
        self.if_inertia = if_inertia
        self.current_diff = current_diff
        self.last_diff = last_diff
        self.decay_rate = decay_rate

    def step_dynamics(self, dt, current_input):
        self.t += dt
        if self.last_input is None:
            self.last_input = current_input
            self.current_input = current_input
        else:
            self.current_input = current_input
        self.current_diff = (self.current_input - self.last_input) / dt * self.factor \
                            + self.if_inertia * self.last_diff * np.exp(-dt * self.decay_rate)

    def update(self):
        self.last_input = self.current_input
        self.last_diff = self.current_diff

    def update_parameters(self, parameters):
        parameters = np.array(parameters)
        self.factor = parameters[:, 0]
        self.decay_rate = parameters[:, 1]
        self.if_inertia = parameters[:, 2]

    def init_recording(self):
        self.trace = {
            't': deque(),
            'factor': deque(),
            'decay_rate': deque(),
            'if_inertia': deque()
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
                            np.array(self.trace['decay_rate'])[index],
                            np.array(self.trace['if_inertia'])[index])).T)

        return figure_dict
