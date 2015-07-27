from fft_util import FftUtil
from wave_util import read_wave_data, write_wave_data


import numpy as np
import matplotlib.pyplot as plt

IN_FILENAME = 'in_audio.wav'
OUT_FILENAME = 'out_audio.wav'



def show_total_spectrum(data):
    sp = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()


def reduce_to_spectrogram_data(freq_domain_data):
    window_size = freq_domain_data.shape[1]
    return np.copy(abs(freq_domain_data[:, :window_size//2+1]) ** 2)


def expand_from_spectrogram_data(spectrogram_data):
    half_window = spectrogram_data.shape[1]-1
    output = np.copy(spectrogram_data)
    in_shape = spectrogram_data.shape
    output = np.append(output, np.zeros((in_shape[0], half_window-1)), axis=1)
    output[:, half_window+1:] = output[:, half_window-1:0:-1]
    output = output ** 0.5
    return output


def lim_griffen_reconstruction(spectrogram_data, wave_params):
    magnitude_data = expand_from_spectrogram_data(spectrogram_data)
    half_window = spectrogram_data.shape[1]-1
    fft_util = FftUtil(half_window * 2)
    length_of_audio = (magnitude_data.shape[0] + 1) * half_window
    output_signal = np.random.rand(length_of_audio)
    for i in range(101):
        output_freq_domain = fft_util.analysis(output_signal)
        output_signal = fft_util.resynthesis(magnitude_data *
                                             np.exp(np.angle(output_freq_domain) * 1j))
        if i % 10 == 0:
            output_filename = "iter_" + str(i) + ".wav"
            write_wave_data(output_filename, output_signal, wave_params)
    return (output_freq_domain, output_signal)


def plot_spectrogram(spectrogram_data, sample_rate, title=""):
    nyq_freq = sample_rate / 2
    fig, ax = plt.subplots()
    #ax.set_yscale('symlog', linthreshy=0.1)
    x, y = np.mgrid[:spectrogram_data.shape[0], 0:nyq_freq:nyq_freq/spectrogram_data.shape[1]]
    ax.pcolormesh(x, y, spectrogram_data)
    ax.axis('tight')
    plt.title(title)
    plt.show()
    return spectrogram_data


def test_window_params_with_plot(window_length):
    output = np.zeros(window_length*10)
    window = np.hanning(window_length)
    for i in range(0, output.size - window_length, window_length//2):
        output[i:i+window_length] += window
    plt.plot(output)
    plt.show()


def test_analysis_resynthesis():
    fft_util = FftUtil(1024)
    wave_data, wave_params = read_wave_data(IN_FILENAME)
    complex_data = fft_util.analysis(wave_data)
    output = fft_util.resynthesis(complex_data)
    write_wave_data(OUT_FILENAME, output, wave_params)


def plot_comparison(array1, array2):
    if (len(array1) != len(array2)):
        print("WARNING: arrays for plot comparison not same length")
        new_data_shape = max(array1.shape, array2.shape)
        array1.resize(new_data_shape)
        array2.resize(new_data_shape)
    plt.plot(array1)
    plt.plot(array2)
    plt.show()


def main():
    wave_data, wave_params = read_wave_data(IN_FILENAME)

    fft_util = FftUtil(1024)

    freq_domain_data = fft_util.analysis(wave_data)
    plot_spectrogram(freq_domain_data, wave_params.framerate, 'Unaltered freq domain data')
    
    spectrogram_data = reduce_to_spectrogram_data(freq_domain_data)
    plot_spectrogram(spectrogram_data, wave_params.framerate, "Spectrogram data")
        
    expanded_spectrogram_data = expand_from_spectrogram_data(spectrogram_data)
    plot_spectrogram(expanded_spectrogram_data, wave_params.framerate, "Expanded spectrogram data")

    lg_freq_domain, lg_signal = lim_griffen_reconstruction(spectrogram_data, wave_params)
    plot_spectrogram(lg_freq_domain, wave_params.framerate, "Lim Griffen reconstruction")

    reconverted_audio = fft_util.resynthesis(freq_domain_data)
    reconverted_spectrogram_data = fft_util.resynthesis(expanded_spectrogram_data)

    write_wave_data(OUT_FILENAME, reconverted_spectrogram_data, wave_params)
    print("L2-norm of original audio and resynthesis of unmodified data: ", 
          np.linalg.norm(wave_data - reconverted_audio))
    print("L2-norm of original audio and resynthesis of spectrogram_data data: ", 
          np.linalg.norm(wave_data - reconverted_spectrogram_data))
    print("L2-norm of original audio and lim griffen reconstruction: ",
          np.linalg.norm(wave_data - lg_signal))

    plot_comparison(wave_data, lg_signal)

if __name__ == '__main__':
    main()
