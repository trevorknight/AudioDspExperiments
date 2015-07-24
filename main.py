import numpy as np
import matplotlib.pyplot as plt
import wave
from math import ceil

IN_FILENAME = 'in_audio.wav'
OUT_FILENAME = 'out_audio.wav'
WINDOW_LENGTH = 1024
HALF_WINDOW = WINDOW_LENGTH // 2
KAISER_ALPHA = 6


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


def show_total_spectrum(data):
    sp = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()


def compute_spectrogram_data(data):
    new_data_length = ceil(data.size / HALF_WINDOW) * HALF_WINDOW
    data.resize(new_data_length, refcheck=False)
    window = np.hanning(WINDOW_LENGTH)
    number_windows = (data.size / HALF_WINDOW) - 1
    spectrogram_data = np.zeros((number_windows, HALF_WINDOW + 1))
    window_i = 0
    for i in range(0, data.size - WINDOW_LENGTH, HALF_WINDOW):
        print(i)
        print(window_i)
        windowed_data = window * data[i:i+WINDOW_LENGTH]
        fft_result = abs(np.fft.fft(windowed_data)[:HALF_WINDOW + 1]) ** 2
        spectrogram_data[window_i, :] = fft_result
        window_i += 1
    return spectrogram_data


def plot_spectrogram(spectrogram_data, sample_rate):
    nyq_freq = sample_rate / 2
    fig, ax = plt.subplots()
    # ax.set_yscale('symlog', linthreshy=1)
    x, y = np.mgrid[:spectrogram_data.shape[0], 0:nyq_freq:nyq_freq/spectrogram_data.shape[1]]
    ax.pcolormesh(x, y, spectrogram_data)
    ax.axis('tight')
    plt.show()


def analysis(data):
    new_data_length = ceil(data.size / HALF_WINDOW) * HALF_WINDOW
    data.resize(new_data_length, refcheck=False)
    window = np.hanning(WINDOW_LENGTH)
    number_windows = (data.size / HALF_WINDOW) - 1
    output = np.array([np.fft.fft(window * data[i:i+WINDOW_LENGTH]) for i in range(0, len(data)-WINDOW_LENGTH, HALF_WINDOW)])
    #     np.zeros((number_windows, WINDOW_LENGTH))
    # window_i = 0
    # for i in range(0, data.size - WINDOW_LENGTH, HALF_WINDOW):
    #     windowed_data = window * data[i:i+WINDOW_LENGTH]
    #     output[window_i, :] = np.fft.fft(windowed_data)
    #     window_i += 1
    return output


def resynthesis(complex_data):
    output = np.zeros((complex_data.shape[0] + 1) * HALF_WINDOW)
    for n, i in enumerate(range(0, len(output)-WINDOW_LENGTH, HALF_WINDOW)):
        output[i:i+WINDOW_LENGTH] += np.real(np.fft.ifft(complex_data[n]))
    return output


def plot_windows():
    # window = np.kaiser(WINDOW_LENGTH, KAISER_ALPHA)
    window = np.hanning(WINDOW_LENGTH)
    output = np.zeros(WINDOW_LENGTH*10)
    for i in range(0, output.size - WINDOW_LENGTH, HALF_WINDOW//2):
        output[i:i+WINDOW_LENGTH] += window
    plt.plot(output)
    plt.show()


def test_analysis_resynthesis():
    wave_data, wave_params = read_wave_data(IN_FILENAME)
    complex_data = analysis(wave_data)
    output = resynthesis(complex_data)
    write_wave_data(OUT_FILENAME, output, wave_params)
    print(np.linalg.norm(wave_data - output))


def main():
    wave_data, wave_params = read_wave_data(IN_FILENAME)
    spectrogram_data = compute_spectrogram_data(wave_data)
    plot_spectrogram(spectrogram_data, wave_params.framerate)
    reconverted_audio = resynthesis(spectrogram_data)
    write_wave_data(OUT_FILENAME, spectrogram_data, wave_params)


if __name__ == '__main__':
    #test_analysis_resynthesis()
    plot_windows()