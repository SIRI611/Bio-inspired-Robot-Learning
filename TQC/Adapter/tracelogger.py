from collections import deque
import warnings
import dill
import sys
import tempfile
import time
import os
import copy


class TraceLogger:
    def __init__(self, keys, log_file_path='', element_number_allowed=2 ** 20):
        self.trace = dict.fromkeys(keys)
        for key in self.trace.keys():
            self.trace[key] = deque()
        if log_file_path:
            if log_file_path.endswith('.pkl'):
                self.log_file_path = log_file_path
            else:
                self.log_file_path = os.path.join(log_file_path, 'TraceLogger.pkl')
        else:
            self.log_file_path = tempfile.mkstemp(
                suffix=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()), prefix='TraceLogger')
        if not os.path.exists(os.path.dirname(self.log_file_path)):
            os.makedirs(os.path.dirname(self.log_file_path))
        self.element_number_allowed = element_number_allowed
        self.index_of_dumped = 0
        self.keys = keys
        self.counter = 0
        print('self.log_file_path: ', self.log_file_path)

    def append(self, data_dict):
        for key, value in data_dict.items():
            self.counter += 1
            self.trace[key].append(copy.deepcopy(value))
            if key not in self.trace:
                warnings.warn('The key %s to log does not exist in the trace' % key)

    def save_trace(self, log_file_path='', append=True):
        if not log_file_path:
            path = self.log_file_path
        else:
            path = log_file_path
        if append:
            with open(path, 'ab') as fTraces:
                dill.dump(self.trace, fTraces, protocol=dill.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as fTraces:
                dill.dump(self.trace, fTraces, protocol=dill.HIGHEST_PROTOCOL)

    def clear_cache(self):
        for key in self.trace:
            self.trace[key].clear()
        self.counter = 0

    def memory_maintenance(self, log_file_path='', force_save=False):
        oversized = self.counter > self.element_number_allowed
        last_counter = self.counter
        if oversized or force_save:
            self.save_trace(log_file_path=log_file_path)
            self.clear_cache()
        return [oversized, last_counter]


if __name__ == "__main__":
    import random

    keys = ['a', 'b', 'c']
    log_file_path = "F:\\recording\\recording\\Test\\" + \
                    time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pkl'
    aTL = TraceLogger(keys, log_file_path=log_file_path, element_number_allowed=2 ** 12)
    for i1 in range(10000):
        step_dict = dict.fromkeys(keys, i1)
        aTL.append(step_dict)
        aTL.memory_maintenance()
    aTL.save_trace()
