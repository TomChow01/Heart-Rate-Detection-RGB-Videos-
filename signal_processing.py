import cv2
import numpy as np
import time
from scipy import signal

# Signal Processing class
class Signal_processing():
    def __init__(self):
        self.a = 1
        
    def extract_color(self, ROIs):
        '''
        extract average value of green color from ROIs
        '''
        
        g = []
        for ROI in ROIs:
            g.append(np.mean(ROI[:,:,1]))
        output_val = np.mean(g)
        return output_val
    
    def normalization(self, data_buffer):
        '''
        normalize the input data buffer
        '''
        
        normalized_data = data_buffer/np.linalg.norm(data_buffer)
        
        return normalized_data
    
    def signal_detrending(self, data_buffer):
        '''
        remove overall trending
        
        '''
        detrended_data = signal.detrend(self.clean(np.array(data_buffer)))
        
        return detrended_data
    
    def clean(self, serie):
        """
        remove NaN and Inf values
        """
        output = serie[(np.isnan(serie) == 0) & (np.isinf(serie) == 0)]
        return output
    
    def median_filter(self, data_buffer, kernel_size):
        '''
        removes noise
        '''
        smoothed_data = signal.medfilt(data_buffer, kernel_size)
        return smoothed_data
    
    def running_mean(self, data_buffer, N):
        """ x == an array of data. N == number of samples per average """
        cumsum = np.cumsum(np.insert(data_buffer, [0,0,0], 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
        
        
    def interpolation(self, data_buffer, times):
        '''
        interpolation data buffer to make the signal become more periodic (advoid spectral leakage)
        
        '''
        L = len(data_buffer)
        
        even_times = np.linspace(times[0], times[-1], L)
        
        interp = np.interp(even_times, times, data_buffer)
        interpolated_data = np.hamming(L) * interp
        return interpolated_data
        
    def fft(self, data_buffer, fps):
        '''
        Apply Fast Fourier Transform
        
        '''
        
        L = len(data_buffer)
        
        freqs = float(fps) / L * np.arange(L / 2 + 1)
        
        freqs_in_minute = 60. * freqs
        
        raw_fft = np.fft.rfft(data_buffer*30)
        fft = np.abs(raw_fft)**2
        
        interest_idx = np.where((freqs_in_minute > 50) & (freqs_in_minute < 180))[0]
        interest_idx_sub = interest_idx[:-1].copy() #advoid the indexing error
        freqs_of_interest = freqs_in_minute[interest_idx_sub]
        
        fft_of_interest = fft[interest_idx_sub]
                        
        return fft_of_interest, freqs_of_interest


    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):
        '''
        Bandpass Filter 
        
        '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
        filtered_data = signal.lfilter(b, a, data_buffer)
        
        return filtered_data