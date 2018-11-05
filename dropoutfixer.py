""" Script that identifies and removes dropouts. """

##### Note that this script operates in seconds/Hz. Be careful about the transformations!

# Dependencies
import stf
import numpy as np
import utilities
import copy

# Define a class
class DropoutFixer:
    # Class that takes in data and fixes the dropouts.
    def __init__(self, trace, dropout_time = 0.01, median_window=3.0, min_score=10.0):
        # Initial sorting of data
        self.data = copy.copy(trace)
        self.dt = utilities.dt()
        self.N = self.data.size
        self.t = self.dt*np.array(range(self.N))
        self.median_window = int(median_window/self.dt)
        self.dropout_time = dropout_time
        self.dropout_window = int(dropout_time/self.dt)
        self.fixing_interval = 10
        self.start_end_offset = int(median_window/(2.0*self.dt))
        self.f = np.fft.rfftfreq(self.N, d=self.dt)
        self.min_score = min_score

        # Remove median
        self.data_clean = self.data-utilities.rolling(self.data,WINDOW=self.median_window,FUNC='MEDIAN')

    def _boxcar(self,T,AMP=1.0,DELTA_T=0.001):
        retarray = np.zeros(T.shape[0])
        boxtime = np.where((T>0.) & (T<DELTA_T))
        retarray[boxtime] = AMP
        return retarray

    def _characterise_correlations(self):
        # Note that the correlation in the first and last half-median-window is set to zero.
        _template = self._boxcar(self.t,DELTA_T=self.dropout_time)
        _ft_template = utilities.fourier(_template,self.dt)
        _ft_data_clean = utilities.fourier(self.data_clean,self.dt)

        _score = -np.fft.irfft(_ft_data_clean * _ft_template.conjugate())
        _score_med, _score_std = utilities.sufficient_statistics(_score)
        _SCORE = (_score-_score_med)/_score_std
        _SCORE[:self.start_end_offset] = 0.0
        _SCORE[-self.start_end_offset:] = 0.0
        self.score = _SCORE

    def _identify_dropouts(self):
        _score = copy.copy(self.score)

        # Create fluff
        _dropout_box = {}
        _loop, _dropout_cnt = True, 0
        while _loop == True:
            if _score.max() < self.min_score:
                _loop = False
                continue

            _baseline = np.where(_score<1)
            _max_score_idx = np.argmax(_score)
            _baseline_return_idx = np.searchsorted(_baseline[0],_max_score_idx)

            # Account for case where _baseline_return_idx is at start of trace
            if _baseline_return_idx > 0:
                _dropout_start_idx = _baseline[0][_baseline_return_idx-1]
            else:
                _dropout_start_idx = 0

            # Account for case where _baseline_return_idx is at end of trace
            if _baseline_return_idx < _baseline[0].size:
                _dropout_end_idx = _baseline[0][_baseline_return_idx]
            else:
                _dropout_end_idx = -1
            
            
            _dropout_start_time = self.t[_dropout_start_idx]
            _dropout_end_time = self.t[_dropout_end_idx]
            _score[_dropout_start_idx:_dropout_end_idx] = 0.0
            
            _dropout_box[_dropout_cnt] = {'dropout_duration':_dropout_end_time-_dropout_start_time,'dropout_start_time':_dropout_start_time,'dropout_end_time':_dropout_end_time,'dropout_start_idx':_dropout_start_idx,'dropout_end_idx':_dropout_end_idx}
            _dropout_cnt += 1
            if _dropout_cnt>10:
                _loop=False

        # Store dropouts that were found
        self.dropout_box = _dropout_box
        self.dropout_cnt = _dropout_cnt

    def _merge_nearby_dropouts(self):
        # Create a copy of dropout_box
        _dropout_box = copy.copy(self.dropout_box)

        # Loop through dropouts
        _dropout_idx = 0
        _start_idx = np.array([v['dropout_start_idx'] for k,v in _dropout_box.items()]) - self.fixing_interval
        _end_idx = np.array([v['dropout_end_idx'] for k,v in _dropout_box.items()]) + self.fixing_interval
        while _dropout_idx < self.dropout_cnt:

            # Create list of start and end indicies
            
            overlap = np.where( (_start_idx[_dropout_idx] <= _end_idx) & (_start_idx <= _end_idx[_dropout_idx]) )[0]

            for _sub_dropout_idx in overlap:
                if _sub_dropout_idx == _dropout_idx:
                    continue

                _dropout_box[_dropout_idx]['dropout_start_time'] = min(_dropout_box[_dropout_idx]['dropout_start_time'],_dropout_box[_sub_dropout_idx]['dropout_start_time'])
                _dropout_box[_dropout_idx]['dropout_end_time'] = max(_dropout_box[_dropout_idx]['dropout_end_time'],_dropout_box[_sub_dropout_idx]['dropout_end_time'])
                _dropout_box[_dropout_idx]['dropout_start_idx'] = min(_dropout_box[_dropout_idx]['dropout_start_idx'],_dropout_box[_sub_dropout_idx]['dropout_start_idx'])
                _dropout_box[_dropout_idx]['dropout_end_idx'] = max(_dropout_box[_dropout_idx]['dropout_end_idx'],_dropout_box[_sub_dropout_idx]['dropout_end_idx'])
                _dropout_box[_dropout_idx]['dropout_duration'] = _dropout_box[_dropout_idx]['dropout_end_time']-_dropout_box[_dropout_idx]['dropout_start_time']
                del _dropout_box[_sub_dropout_idx]

            _dropout_idx += 1

            # Loop until hit valid index
            _loop = True
            while _loop and _dropout_idx < self.dropout_cnt:
                try:
                    _dump = _dropout_box[_dropout_idx]
                    _loop = False
                except KeyError:
                    _dropout_idx+=1

        # Output clean dropout box
        self.dropout_box = _dropout_box

    def _quantify_dropouts(self):
        # Create data copy
        _data_residual = copy.copy(self.data)

        # Loop through dropouts and measure true amplitude
        for _dropout_key in self.dropout_box.keys():
            _box = self.dropout_box[_dropout_key]
            _dropout_interval_idx = range(_box['dropout_start_idx'],_box['dropout_end_idx']+1)
            _peak_idx = _dropout_interval_idx[np.argmin(self.data[_dropout_interval_idx])]
            self.dropout_box[_dropout_key]['peak_idx'] = _peak_idx
            self.dropout_box[_dropout_key]['peak_time'] = self.t[_peak_idx]
            self.dropout_box[_dropout_key]['peak_drop'] = self.data[_peak_idx]

            _data_med_inner, _data_std_inner = utilities.sufficient_statistics(self.data[_box['dropout_start_idx']-self.fixing_interval:_box['dropout_start_idx']])
            _data_med_outer, _data_std_outer = utilities.sufficient_statistics(self.data[_box['dropout_end_idx']+1:_box['dropout_end_idx']+1+self.fixing_interval])
            _data_std = (_data_std_inner+_data_std_outer)/2.0
            _data_residual[_dropout_interval_idx] = _data_med_inner+(_data_med_outer-_data_med_inner)*(self.t[_dropout_interval_idx]-_box['dropout_start_time'])/_box['dropout_duration']
            self.dropout_box[_dropout_key]['dropout_charge'] = self.dt*np.sum(self.data[_dropout_interval_idx]-_data_residual[_dropout_interval_idx])
            #print _data_std_inner, _data_std_outer, _box['dropout_start_idx']-self.fixing_interval
            _data_residual[_dropout_interval_idx] += np.random.normal(0.0,_data_std,len(_dropout_interval_idx))

        # Store residual data
        self.data_residual = _data_residual

    def _show_dropouts(self):
        # Place markers
        stf.erase_markers()
        for k in self.dropout_box.keys():
            stf.set_marker(self.dropout_box[k]['peak_idx'],self.dropout_box[k]['peak_drop'])

        # Create results table
        _result_box = {}
        _interesting = ['dropout_charge','dropout_start_time','dropout_end_time']
        for k in _interesting:
            _result_box[k] = [self.dropout_box[kbox][k] for kbox in self.dropout_box.keys()]
        self.result_box = _result_box
        stf.show_table_dictlist(self.result_box,caption='Dropout table')

    def run(self):
        # Generate median and std of charge
        self._characterise_correlations()

        # Identify the dropouts
        self._identify_dropouts()
        print "Identified dropouts"

        # If there are any dropouts near others then merge them
        self._merge_nearby_dropouts()
        print "Merged dropouts"

        # Quantify the charge in dropouts and remove them
        self._quantify_dropouts()
        print "Quantified dropouts"

        # Show result table
        self._show_dropouts()
        print "Showed big dropouts"

