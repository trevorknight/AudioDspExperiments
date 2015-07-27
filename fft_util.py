""" Convenience methods for converting a signal into the frequency domain with windowed

Usage.
fft_util = FftUtils(1024)
freq_domain_data = fft_util.analysis(signal)
reconstructed_signal = fft_util.resynthesis(freq_domain_data)
"""
from math import ceil
import numpy as np


class FftUtil:
    def __init__(self, window_length):
        if window_length <= 0 or (window_length & (window_length - 1)) is not 0:
            raise ValueError('Window length should be a positive power of 2')
        self.__window_length = window_length
        self.__half_window = window_length // 2

    def analysis(self, data):
        new_data_length = ceil(data.size / self.__half_window) * self.__half_window
        data.resize(new_data_length, refcheck=False)
        window = np.hanning(self.__window_length)
        output = np.array([np.fft.fft(window * data[i:i+self.__window_length])
                               for i in range(0, len(data)-self.__half_window, self.__half_window)])
        return output

    def resynthesis(self, freq_domain_data):
        output = np.zeros((freq_domain_data.shape[0] + 1) * self.__half_window)
        for n, i in enumerate(range(0, len(output)-self.__window_length, self.__half_window)):
            output[i:i+self.__window_length] += np.real(np.fft.ifft(freq_domain_data[n]))
        return output