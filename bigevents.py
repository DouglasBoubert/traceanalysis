""" Script that identifies all big events, finds the area beneath them, and masks them."""

# Dependencies
import stf
import numpy as np
import utilities
import copy

# Identify bigevents
class BigEventHandler:
    def __init__(self, trace, min_charge = -200, event_time = 100, median_window=1000.0,fixing_interval=100):
        # Initial sorting of data
        self.data = copy.copy(trace)
        self.dt = stf.get_sampling_interval()
        self.N = self.data.size
        self.t = self.dt*np.array(range(self.N))
        self.median_window = int(median_window/self.dt)
        self.min_charge = min_charge
        self.event_time = event_time
        self.event_window = int(event_time/self.dt)
        self.fixing_interval = fixing_interval
        
        self.big_event_mask=np.array([])

        # Remove median
        if median_window <0.0:
            self.data_clean = self.data-np.median(self.data)
            self.start_end_offset = 10
        else:
            self.data_clean = self.data-utilities.rolling(self.data,WINDOW=self.median_window,FUNC='MEDIAN')
            self.start_end_offset = int(median_window/(2.0*self.dt))

    def _rolling_charge(self,TARGET):
        # Note that the charge in the first and last half-median-window is set to zero.
        _charge = self.dt*utilities.rolling(TARGET,WINDOW=self.event_window,FUNC='SUM')
        _charge[:self.start_end_offset] = 0.0
        _charge[-self.start_end_offset:] = 0.0
        return _charge

    def _characterise_rolling_charge(self):
        self.charge_med, self.charge_std = utilities.sufficient_statistics(self._rolling_charge(self.data_clean))

    def _identify_big_events(self):
        # Create copy of clean data
        _data_clean = copy.copy(self.data_clean)

        # Create fluff
        _event_box = {}
        _loop, _event_cnt = True, 0
        while _loop == True:
            _charge = self._rolling_charge(_data_clean)
            if _charge.min() > self.min_charge or (_charge.min()-self.charge_med)/self.charge_std>-5:
                _loop = False
                continue

            _baseline = np.where((_charge-self.charge_med)/self.charge_std>0)
            _maxchargetime = self.t[np.argmin(_charge)]
            _interval = np.where(np.abs(self.t-_maxchargetime)<self.event_time/2.0)[0]
            _peaktime = _interval[0]+np.argmin(_data_clean[_interval])
            _baseline_return_idx = np.searchsorted(_baseline[0],_peaktime)

            # Account for case where _baseline_return_idx is at start of trace
            if _baseline_return_idx > 0:
                _event_start_time = self.t[_baseline][_baseline_return_idx-1]#+bg_eventtime/4.0
            else:
                _event_start_time = self.t[0]#+bg_eventtime/4.0

            # Account for case where _baseline_return_idx is at end of trace
            if _baseline_return_idx < _baseline[0].size:
                _event_end_time = self.t[_baseline][_baseline_return_idx]#-bg_eventtime/4.0
            else:
                _event_end_time = self.t[-1]#-bg_eventtime/4.0
            
            
            _event_start_idx = int((_event_start_time-self.t[0])/self.dt)
            _event_end_idx = int((_event_end_time-self.t[0])/self.dt)
            _data_clean[_event_start_idx:_event_end_idx] = 0.0
            
            _event_box[_event_cnt] = {'event_duration':_event_end_time-_event_start_time,'event_start_time':_event_start_time,'event_end_time':_event_end_time,'event_start_idx':_event_start_idx,'event_end_idx':_event_end_idx}
            _event_cnt += 1
            if _event_cnt>100:
                _loop=False

        # Store events that were found
        self.event_box = _event_box
        self.event_cnt = _event_cnt

    def _merge_nearby_big_events(self):
        # Create a copy of event_box
        _event_box = copy.copy(self.event_box)

        

        # Loop through events
        _event_idx = 0
        _start_idx = np.array([v['event_start_idx'] for k,v in _event_box.items()]) - self.fixing_interval
        _end_idx = np.array([v['event_end_idx'] for k,v in _event_box.items()]) + self.fixing_interval
        while _event_idx < self.event_cnt:

            # Create list of start and end indicies
            
            overlap = np.where( (_start_idx[_event_idx] <= _end_idx) & (_start_idx <= _end_idx[_event_idx]) )[0]

            for _sub_event_idx in overlap:
                if _sub_event_idx == _event_idx:
                    continue

                _event_box[_event_idx]['event_start_time'] = min(_event_box[_event_idx]['event_start_time'],_event_box[_sub_event_idx]['event_start_time'])
                _event_box[_event_idx]['event_end_time'] = max(_event_box[_event_idx]['event_end_time'],_event_box[_sub_event_idx]['event_end_time'])
                _event_box[_event_idx]['event_start_idx'] = min(_event_box[_event_idx]['event_start_idx'],_event_box[_sub_event_idx]['event_start_idx'])
                _event_box[_event_idx]['event_end_idx'] = max(_event_box[_event_idx]['event_end_idx'],_event_box[_sub_event_idx]['event_end_idx'])
                _event_box[_event_idx]['event_duration'] = _event_box[_event_idx]['event_end_time']-_event_box[_event_idx]['event_start_time']
                del _event_box[_sub_event_idx]

            _event_idx += 1

            # Loop until hit valid index
            _loop = True
            while _loop and _event_idx < self.event_cnt:
                try:
                    _dump = _event_box[_event_idx]
                    _loop = False
                except KeyError:
                    _event_idx+=1

        # Output clean event box
        self.event_box = _event_box

    def _quantify_big_events(self):
        # Create data copy
        _data_residual = copy.copy(self.data)

        # Loop through events and measure true amplitude
        for _event_key in self.event_box.keys():
            _box = self.event_box[_event_key]
            _event_interval_idx = range(_box['event_start_idx'],_box['event_end_idx']+1)
            self.big_event_mask = np.concatenate([self.big_event_mask,_event_interval_idx])
            _peak_idx = _event_interval_idx[np.argmin(self.data[_event_interval_idx])]
            self.event_box[_event_key]['peak_idx'] = _peak_idx
            self.event_box[_event_key]['peak_time'] = self.t[_peak_idx]
            

            _data_med_inner, _data_std_inner = utilities.sufficient_statistics(self.data[_box['event_start_idx']-self.fixing_interval:_box['event_start_idx']])
            _data_med_outer, _data_std_outer = utilities.sufficient_statistics(self.data[_box['event_end_idx']+1:_box['event_end_idx']+1+self.fixing_interval])
            _data_std = (_data_std_inner+_data_std_outer)/2.0
            _data_residual[_event_interval_idx] = _data_med_inner+(_data_med_outer-_data_med_inner)*(self.t[_event_interval_idx]-_box['event_start_time'])/_box['event_duration']
            self.event_box[_event_key]['event_charge'] = self.dt*np.sum(self.data[_event_interval_idx]-_data_residual[_event_interval_idx])
            #print _data_std_inner, _data_std_outer, _box['event_start_idx']-self.fixing_interval
            _data_residual[_event_interval_idx] += np.random.normal(0.0,_data_std,len(_event_interval_idx))
            

        # Store residual data
        self.data_residual = _data_residual

    def _show_big_events(self):
        # Place markers
        stf.erase_markers()
        for k in self.event_box.keys():
            _peak_idx = self.event_box[k]['peak_idx']
            self.event_box[k]['peak_current'] = self.data[_peak_idx]-self.data_residual[_peak_idx]
            stf.set_marker(self.event_box[k]['peak_idx'],self.event_box[k]['peak_current'])

        # Create results table
        _result_box = {}
        _interesting = ['event_charge','event_start_time','event_end_time','peak_current']
        for k in _interesting:
            _result_box[k] = [self.event_box[kbox][k] for kbox in self.event_box.keys()]
        self.result_box = _result_box
        stf.show_table_dictlist(self.result_box,caption='Big event table')

    def run(self):
        # Generate median and std of charge
        self._characterise_rolling_charge()

        # Identify the big events
        self._identify_big_events()
        print "Identified events"

        # If there are any events near others then merge them
        self._merge_nearby_big_events()
        print "Merged events"

        # Quantify the charge in big events and remove them
        self._quantify_big_events()
        print "Quantified events"

        # Show result table
        self._show_big_events()
        print "Showed big events"

#BEH = BigEventHandler(stf.get_trace())
