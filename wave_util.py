""" Convenience methods for reading/writing all of a wave file

Usage.
wave_data, wave_params = read_wave_data(in_filename)
write_wave_data(out_filename, wave_data, wave_params)

"""

import numpy as np
import wave

def read_wave_data(filename):
    wave_read = wave.open(filename, 'rb')
    data = wave_read.readframes(-1)
    data = np.fromstring(data, 'Int16')
    params = wave_read.getparams()
    wave_read.close()
    return (data, params)


def write_wave_data(filename, data, wave_params):
    wave_write = wave.open(filename, 'wb')
    wave_write.setparams(wave_params)
    data = data.astype('Int16')
    wave_write.writeframes(data)
    wave_write.close()


