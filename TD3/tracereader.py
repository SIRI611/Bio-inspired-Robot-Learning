import os
import warnings
from collections import deque

import numpy as np

from filereader import fileload


class TraceReader:
    def __init__(self, log_file_path=''):
        self.log_file_path = log_file_path
        self.fileloader = fileload(self.log_file_path)
        self.trace = {}

    def append(self, data_dict):
        if not self.trace:
            self.trace = data_dict
        else:
            for key, value in data_dict.items():
                if not key in self.trace:
                    warnings.warn('The key %s to log does not exist in the trace' % key)
                self.trace[key].extend(value)

    def get_trace(self):
        for asegment in self.fileloader:
            self.append(asegment)
            # print(type(asegment))
        return self.trace

    def reconstruct_trace(self):
        self.reconstructed_trace = dict()
        for key in self.trace.keys():
            self.reconstructed_trace[key] = np.stack(np.array(self.trace[key]))
        return self.reconstructed_trace

if __name__ == "__main__":
    from pprint import pprint
    log_file_path = "test_trace_logger/" + '2024-01-26_18-45-31' + '.pkl'
    aTR = TraceReader(log_file_path=log_file_path)
    trace = aTR.get_trace()
    pprint(len(trace))
    # for key, value in trace.items():
    #     print(key, ':')
    #     pprint(value)
