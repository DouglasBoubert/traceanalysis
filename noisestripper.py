""" Script that identifies and removes noise from a trace. Also the median. """

##### Note that this script operates in seconds/Hz. Be careful about the transformations!

# Dependencies
import stf
import numpy as np
import utilities
import copy
from scipy import signal
from scipy import stats
import time

# Define a class
class NoiseStripper:
    # Class that takes in data and removes significant spikes.
    # Also removes median.
    def __init__(self, trace, data_prob_bound=0.1, median_window=1.0, median_spike_window = 100., freq_bound=5.0, min_freq=20.0, notch_band=1.0):
        # Initial sorting of data
        self.data = copy.copy(trace)
        self.dt = utilities.dt()
        self.N = self.data.size
        self.t = self.dt*np.array(range(self.N))
        self.median_window = int(median_window/self.dt)
        self.start_end_offset = int(median_window/(2.0*self.dt))

        # Remove median
        self.data_clean = self.data-utilities.rolling(self.data,WINDOW=self.median_window,FUNC='MEDIAN')

        # Frequencies
        self.f = np.fft.rfftfreq(self.N, d=self.dt)
        self.df = self.f[1]-self.f[0]
        self.log_prob_bound = np.log(data_prob_bound)
        self.freq_bound = freq_bound
        self.min_freq=min_freq
        self.notch_band=notch_band
        self.median_spike_window = int(median_spike_window/self.df)
        
    # Required input defintions are as follows;
    # time:   Time between samples
    # band:   The bandwidth around the centerline freqency that you wish to filter
    # freq:   The centerline frequency to be filtered
    # ripple: The maximum passband ripple that is allowed in db
    # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
    #         IIR filters are best suited for high values of order.  This algorithm
    #         is hard coded to FIR filters
    # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
    # data:         the data to be filtered
    #from scipy.signal import iirfilter, lfilter
    def _implement_notch_filter(self, time, band, freq, order, filter_type, data,ripple=None):

        fs   = 1/time
        nyq  = fs/2.0
        low  = freq - band/2.0
        high = freq + band/2.0
        low  = low/nyq
        high = high/nyq
        b, a = signal.iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                         analog=False, ftype=filter_type)
        filtered_data = signal.lfilter(b, a, data)
        return filtered_data

    def _obtain_fourier_amplitudes(self): 
        # Fourier transform data
        _w = signal.blackman(self.N)
        self.ft_data_clean = utilities.fourier(_w*self.data_clean,self.dt)/self.dt
        self.amp = 2.0/self.N * np.abs(self.ft_data_clean)

    def _significance_of_amplitudes(self):
        # Calculate significance of spikes
        _median_amp = utilities.rolling(self.amp,WINDOW=self.median_spike_window,FUNC='MEDIAN')
        _sig_of_spike = self.amp/_median_amp
        _spike_med, _spike_std = utilities.sufficient_statistics(np.log(_sig_of_spike))
        self.spike_significance = np.exp((np.log(_sig_of_spike)-_spike_med)/_spike_std)
        self.significant_spikes = np.where((stats.lognorm.logcdf(self.spike_significance,s = 1.0)*self.f.size > self.log_prob_bound)&(self.f>self.min_freq))
    
    def _group_spikes_together(self):
        # Group spikes
        _spike_box = {}
        _group_i = 0
        _s_last = self.significant_spikes[0][0]
        _group_tmp = []
        for _s in self.significant_spikes[0]:
            if abs(_s-_s_last)< self.freq_bound:
                _group_tmp.append(_s)
            else:
                _spike_box[str(_group_i)] = {}
                _spike_box[str(_group_i)]['indices'] = _group_tmp
                _group_i +=1
                _group_tmp = [_s]
            _s_last = _s
        _spike_box[str(_group_i)] = {}
        _spike_box[str(_group_i)]['indices'] = _group_tmp
        self.spike_box = _spike_box

    def _process_spike_groups(self):
        # Sort through spikes
        for _k,_v in self.spike_box.items():
            self.spike_box[_k]['freqs'] = self.f[_v['indices']]
            self.spike_box[_k]['weights'] = self.spike_significance[_v['indices']]
            self.spike_box[_k]['FREQ'] = ((self.spike_box[_k]['freqs']*self.spike_box[_k]['weights'])/self.spike_box[_k]['weights'].sum()).sum()
        self.spikes = [self.spike_box[_k]['FREQ'] for _k in self.spike_box.keys()]

    def _remove_spikes(self):
        # Remove the identified spikes
        _data_squeaky_clean = copy.copy(self.data_clean)
        print("There are",len(self.spikes),"spikes")
        for _freq in self.spikes:
            _data_squeaky_clean = self._implement_notch_filter(self.dt,band=self.notch_band,freq=_freq,order =2.,filter_type='butter',data=_data_squeaky_clean)
        self.data_squeaky_clean = _data_squeaky_clean
        

    def run(self):
        # Obtain Fourier amplitudes
        _t = time.time()
        self._obtain_fourier_amplitudes()
        print 'Obtained Fourier amplitudes'
        print time.time()-_t

        # Assess significance of Fourier amplitudes
        _t = time.time()
        self._significance_of_amplitudes()
        print 'Assessed significance of Fourier amplitudes'
        print time.time()-_t

        # Group candidate spikes together
        _t = time.time()
        self._group_spikes_together()
        print 'Grouped spikes together'
        print time.time()-_t

        # Process spike groups
        _t = time.time()
        self._process_spike_groups()
        print 'Processed spike groups'
        print time.time()-_t
            
        # Remove spikes
        _t = time.time()
        self._remove_spikes()
        print 'Removed spikes'
        print time.time()-_t
        