import copy

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import time
import tempfile

# from tracelogger import TraceLogger
# from tracereader import TraceReader


class RangeAdapter:

    def __init__(self, targe_max_output=1, targe_min_output=-1, factor=1, bias=0,
                 update_rate=0.00005, t=0, name='RangeAdapter'):
        self.targe_max_output = targe_max_output
        self.targe_min_output = targe_min_output
        self.targe_output_bias = (self.targe_max_output + self.targe_min_output) / 2
        self.targe_output_range = (self.targe_max_output - self.targe_min_output)
        self.current_factor = factor
        self.current_bias = bias
        self.last_factor = factor
        self.last_bias = bias
        self.current_output = None
        self.last_output = self.current_output
        self.current_input = None
        self.last_input = self.current_input
        self.t = t
        self.update_rate = update_rate
        self.name = name

    def step_dynamics(self, dt, input_value):
        self.t += dt
        self.current_input = np.array(input_value)
        # factor1 = (np.log10(self.current_factor) - (-2)) * (2 - np.log10(self.current_factor)) / 4
        factor1 = 1
        self.current_bias += (self.current_input - self.last_bias) * self.last_factor \
                             * self.update_rate * dt * factor1
        self.current_factor += (self.targe_output_range * 0.5 - self.last_factor \
                                * np.abs(self.current_input - self.current_bias)) \
                               * dt * self.update_rate * factor1
        self.current_factor[self.current_factor < np.finfo(self.current_factor.dtype).tiny] \
            = np.finfo(self.current_factor.dtype).tiny
        self.current_output = (input_value - self.last_bias) * self.current_factor + self.targe_output_bias
        return self.current_output

    def step_dynamics(self, dt, input_value, factor_rate=1e-1):
        self.t += dt
        self.current_input = np.array(input_value)
        factor1 = (np.log10(self.current_factor) - np.log10(np.finfo(float).tiny)+1) \
                  * (np.log10(np.finfo(float).max) - np.log10(self.current_factor)+1) / \
                  ((np.log10(np.finfo(float).max) - np.log10(np.finfo(float).tiny))/2)**2
        self.current_bias += (self.current_input - self.last_bias)  \
                             * self.update_rate * dt * factor1
        self.current_factor += (self.targe_output_range * 0.25 - self.last_factor \
                                * np.abs(self.current_input - self.current_bias)) \
                               * dt * self.update_rate * factor1*factor_rate
        if np.isscalar(self.current_factor):
            if self.current_factor < np.finfo(self.current_factor.dtype).tiny:
                self.current_factor = np.finfo(self.current_factor.dtype).tiny
        else:
            self.current_factor[self.current_factor < np.finfo(self.current_factor.dtype).tiny] \
                = np.finfo(self.current_factor.dtype).tiny
        self.current_output = (input_value - self.last_bias) * self.current_factor + self.targe_output_bias
        return self.current_output

    def update(self):
        self.last_factor = self.current_factor
        self.last_bias = self.current_bias
        self.last_input = self.current_input
        self.last_output = self.current_output

    def update_parameters(self, parameters):
        parameters = np.array(parameters)
        self.factor = parameters[:, 0]
        self.bias = parameters[:, 1]

    # def init_recording(self, name_list=[], log_path='', log_name=''):
    #     if not name_list:
    #         name_list = ['t',
    #                      'current_factor',
    #                      'current_bias',
    #                      'current_input',
    #                      'current_output']
    #     self.name_list = name_list
    #     self.recording_state = True
    #     if not log_path:
    #         log_path = tempfile.gettempdir()
    #     if not log_name:
    #         log_name = self.name + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    #     self.log_file_path = os.path.join(log_path, log_name + '.pkl')
    #     self.trace_logger = TraceLogger(name_list, self.log_file_path)

    # def recording(self, finish=False):
    #     temp_dict = {}
    #     for item in self.name_list:
    #         # exec("temp = self.%s" % (key))
    #         temp_dict[item] = getattr(self, item)
    #     self.trace_logger.append(temp_dict)

    # def memory_maintenance(self):
    #     self.trace_logger.memory_maintenance()

    # def save_recording(self, append=True):
    #     self.trace_logger.save_trace(append=append)

    # def clear_record_cache(self):
    #     self.trace_logger.clear_cache()

    # def retrieve_record(self):
    #     self.trace_reader = TraceReader(self.log_file_path)
    #     self.trace = self.trace_reader.get_trace()
    #     return self.trace

    def plot(self, path=None, save_plots=False, start_time_rate=0, down_sample_rate=10,
             neuron_id=0, name_str=None, figure_handle=None):
        self.retrieve_record()
        if name_str is None:
            name_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        figure_dict = dict()
        if figure_handle is None:
            figure_dict[str(neuron_id)] = plt.figure()
        else:
            figure_dict[str(neuron_id)] = figure_handle
        ax_list =  figure_dict[str(neuron_id)].get_axes()
        if len(ax_list) == 0:
            ax = figure_dict[str(neuron_id)].add_subplot()
        else:
            ax = ax_list[0]
            ax.clear()
        start_index = int(start_time_rate * len(self.trace['t']))
        index = np.s_[start_index::down_sample_rate, neuron_id]
        t_list = (np.array(self.trace['t'])[start_index::down_sample_rate])
        line_current_factor, = ax.plot(t_list, np.array(self.trace['current_factor'])[index], label='factor')
        line_bias_factor, = ax.plot(t_list, np.array(self.trace['current_bias'])[index], label='bias')
        line_input_factor, = ax.plot(t_list, np.array(self.trace['current_input'])[index], label='input')
        line_output_factor, = ax.plot(t_list, np.array(self.trace['current_output'])[index], label='output')
        plt.legend()
        # [line_current_factor, line_bias_factor, line_input_factor, line_output_factor])

        # ['factor', 'bias', 'input', 'output'])
        if save_plots == True:
            if not os.path.exists(path):
                os.makedirs(path)
            pp = PdfPages(path + "RangeAdapter" + name_str + '.pdf')
            for key in figure_dict:
                figure_dict[key].savefig(pp, format='pdf')
            pp.close()
        return figure_dict


# class RangeAdaptor(RangeAdapter):
#     pass


if __name__ == "__main__":
    dt = 20
    start_time = 0
    stop_time = 100000
    t = np.linspace(start_time, stop_time, num=int(stop_time / dt + 1))
    # input_value = np.log(t+1) * (np.sin(t / 1000)) + np.random.randn(3, t.shape[0])
    input_value = np.sin(t / 10000 + 1) * 3 * (np.sin(t / 1000)) + np.random.randn(3, t.shape[0]) + np.sin(
        t / 20000) * 2
    input_value *= 1000
    range_adapter = RangeAdapter()
    range_adapter.init_recording()
    for step_index in range(len(t)):
        range_adapter.step_dynamics(dt, input_value[:, step_index])
        range_adapter.recording()
        range_adapter.update()

    range_adapter.save_recording()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    figure=plt.figure()
    figure.add_subplot()
    figure = range_adapter.plot(figure_handle=figure)['0']
    plt.grid()
    plt.show()
