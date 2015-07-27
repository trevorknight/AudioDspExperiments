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


def compute_spectrogram_data(freq_domain_data):
    return np.copy(abs(freq_domain_data[:, :HALF_WINDOW+1]) ** 2)


def expand_spectrogram_data(spectrogram_data):
    output = np.copy(spectrogram_data)
    in_shape = spectrogram_data.shape
    output = np.append(output, np.zeros((in_shape[0], HALF_WINDOW-1)), axis=1)
    output[:, HALF_WINDOW+1:] = output[:, HALF_WINDOW-1:0:-1]
    output = output ** 0.5
    return output


def lim_griffen_reconstruction(spectrogram_data, wave_params):
    length_of_audio = (spectrogram_data.shape[0] + 1) * HALF_WINDOW
    output_signal = np.random.rand(length_of_audio)
    for i in range(10):
        output_freq_domain = analysis(output_signal)
        output_signal = resynthesis(spectrogram_data * 
                                    np.exp(np.angle(output_freq_domain) * 1j))
        output_filename = "iter_" + str(i) + ".wav"
        write_wave_data(output_filename, output_signal, wave_params)


def plot_spectrogram(spectrogram_data, sample_rate):
    nyq_freq = sample_rate / 2
    fig, ax = plt.subplots()
    # ax.set_yscale('symlog', linthreshy=1)
    x, y = np.mgrid[:spectrogram_data.shape[0], 0:nyq_freq:nyq_freq/spectrogram_data.shape[1]]
    ax.pcolormesh(x, y, spectrogram_data)
    ax.axis('tight')
    plt.show()
    return spectrogram_data


def analysis(data):
    new_data_length = ceil(data.size / HALF_WINDOW) * HALF_WINDOW
    data.resize(new_data_length, refcheck=False)
    window = np.hanning(WINDOW_LENGTH)
    output = np.array([np.fft.fft(window * data[i:i+WINDOW_LENGTH]) 
                           for i in range(0, len(data)-HALF_WINDOW, HALF_WINDOW)])
    return output


def resynthesis(freq_domain_data):
    output = np.zeros((freq_domain_data.shape[0] + 1) * HALF_WINDOW)
    for n, i in enumerate(range(0, len(output)-WINDOW_LENGTH, HALF_WINDOW)):
        output[i:i+WINDOW_LENGTH] += np.real(np.fft.ifft(freq_domain_data[n]))
    return output


def test_window_params_with_plot():
    # window = np.kaiser(WINDOW_LENGTH, KAISER_ALPHA)
    window = np.hanning(WINDOW_LENGTH)
    output = np.zeros(WINDOW_LENGTH*10)
    for i in range(0, output.size - WINDOW_LENGTH, HALF_WINDOW):
        output[i:i+WINDOW_LENGTH] += window
    plt.plot(output)
    plt.show()


def test_analysis_resynthesis():
    wave_data, wave_params = read_wave_data(IN_FILENAME)
    complex_data = analysis(wave_data)
    output = resynthesis(complex_data)
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

    freq_domain_data = analysis(wave_data)
    plot_spectrogram(freq_domain_data, wave_params.framerate)
    
    spectrogram_data = compute_spectrogram_data(freq_domain_data)
    plot_spectrogram(spectrogram_data, wave_params.framerate)
        
    reverse_spectrogram_data = expand_spectrogram_data(spectrogram_data)
    plot_spectrogram(reverse_spectrogram_data, wave_params.framerate)

    lim_griffen_reconstruction(reverse_spectrogram_data, wave_params)
    
    reconverted_audio = resynthesis(freq_domain_data)
    reconverted_spectrogram_data = resynthesis(reverse_spectrogram_data)

    write_wave_data(OUT_FILENAME, reconverted_spectrogram_data, wave_params)
    print("L2-norm of original audio and resynthesis of unmodified data: ", 
          np.linalg.norm(wave_data - reconverted_audio))
    print("L2-norm of original audio and resynthesis of spectrogram_data data: ", 
          np.linalg.norm(wave_data - reconverted_spectrogram_data))
    
    plot_comparison(wave_data, reconverted_spectrogram_data)

if __name__ == '__main__':
    main()
