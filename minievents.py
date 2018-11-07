""" Script that identifies all mini events and fits them."""

# Dependencies
import stf
import numpy as np
import utilities
import copy
from scipy import stats, optimize, signal, interpolate
import time
import matplotlib.pyplot as plt

# Identify minievents
class MiniEventHandler:
    def __init__(self, trace, template_name = 'biexponential', event_time = 100.0, median_window=1000.0, bayes_bound=-0.5, bayes_weight=0.1,score_bound = 1.5, min_peak_current=5.0, mask=np.array([]),event_number_barrier=10,event_threshold = {'current_threshold':0.0,'charge_threshold':0.0,'significance_threshold':0.0,'rchi2_threshold':np.inf}):
        # Initial sorting of data
        self.data = copy.copy(trace)
        self.dt = stf.get_sampling_interval()
        self.N = self.data.size
        self.t = self.dt*np.array(range(self.N))
        self.median_window = int(median_window/self.dt)
        self.event_time = event_time
        self.event_window = int(event_time/self.dt)
        self.t_short = self.t[:self.event_window*5]
        self.fixing_interval = 10
        self.start_end_offset = int(median_window/(2.0*self.dt))
        self.bayes_bound = bayes_bound
        self.log_bayes_weight = np.log(bayes_weight)
        self.score_bound = score_bound
        inrange = np.where(mask<self.N)
        self.mask = mask[inrange].astype(np.int_)
        self.event_number_barrier = event_number_barrier
        self.data_list=[self.data]
        self.min_peak_current = min_peak_current
        self.event_threshold = event_threshold
        self.print_bool = True
        self.template_name = template_name
        self._template, self._template_peak, self._template_area, self._template_params_names, self._template_params_ranges, self._template_params_defaults = utilities.obtain_template(TEMPLATE_NAME=self.template_name)

    def _initial_classify_events(self):
        # Calculate correlations
        _score = -np.fft.irfft(self.ft_data * self.ft_template.conjugate(),n=self.data.size)
        
        # Fit a Gaussian to the inner 50% of events
        #self.score_med, self.score_std = utilities.sufficient_statistics(_score)
        self.score_med, self.score_std = utilities.rolling(_score,FUNC='SUFFICIENT_STATISTICS_LEFT',WINDOW=int(0.1*self.median_window))
        
        # Normalise scores in Gaussian
        self.score = (_score-self.score_med)/self.score_std

        # Calculate function for determining signficance
        if False:
            _H1_logpdf = stats.gaussian_kde(self.score).logpdf
            _H0_logpdf = stats.norm.logpdf
            self.bayes_factor_calculator = lambda _s: 2.0*(_H1_logpdf(_s)-_H0_logpdf(_s)+self.log_bayes_weight)
        elif False:
            _points = np.arange(start=self.score.min()-1.0,stop=self.score.max()+1.0,step=0.1)
            _logkde = stats.gaussian_kde(self.score).logpdf(_points)
            _H1_logpdf = interpolate.interp1d(_points,_logkde)
            _H0_logpdf = stats.norm.logpdf
            self.bayes_factor_calculator = lambda _s: 2.0*(_H1_logpdf(_s)-_H0_logpdf(_s)+self.log_bayes_weight)
        else:
            self.bayes_factor_calculator = lambda _s: _s
        

        # Mask out bad regions
        self.score[self.mask] = 0.0
        self.score_initial = copy.copy(self.score)

    def _is_evidence_significant(self,_score):
        _bayes_factor = self.bayes_factor_calculator(_score)
        if  _score > self.score_bound: #_bayes_factor > self.bayes_bound and 
            return True
        else:
            return False

    def _decide_strategy(self):
        # Is the probability of an event with that score or higher less than prob_bound?
        _highest_score_index = np.argmax(self.score)
        return _highest_score_index, self._is_evidence_significant(self.score[_highest_score_index])
        
    
    def _fit_and_remove_event(self,_mean_start_index,_n_points=100):
        # Are there multiple significant peaks?

        NOISE = self.noise_std[_mean_start_index]
        MEDIAN = self.noise_med[_mean_start_index]
        _peaks_bool = True
        _peaks = [_n_points]
        _previous_peaks = []
        _first_peak = _mean_start_index+min(_peaks)-_n_points
        _last_peak = _mean_start_index+max(_peaks)-_n_points
        _default_peak_time = self._template_peak(PARAMS=self._template_params_defaults())
        _left_extension = _n_points
        _right_extension = int(2.0*_n_points)

        while _peaks_bool:
            _score = self.score[_first_peak-_left_extension:_last_peak+_right_extension]
            _data = self.data_residual[_first_peak-_left_extension:_last_peak+_right_extension]
            _time = self.t[_first_peak-_left_extension:_last_peak+_right_extension]
            _peaks = []
            _peaks_heights = []
            _suggested_peaks, _peaks_props  = signal.find_peaks(-_data,prominence=NOISE*2,height = self.min_peak_current+NOISE,width=2)
            if len(_suggested_peaks) == 0:
                self.score[-10+_mean_start_index:10+_mean_start_index] = 0.0
                return False
            for _peak_idx,_peak_height in zip(_suggested_peaks,_peaks_props['peak_heights']):
                _mad_idx = int((_time[_peak_idx]-_default_peak_time)/self.dt)
                _max_score = self.score[_mad_idx]
                if _max_score>0.5:
                    _peaks.append(_peak_idx)
                    _peaks_heights.append(-_peak_height)

            if _peaks == _previous_peaks and len(_peaks)>0 and _n_points in _peaks:
                _peaks_bool=False
            elif len(_peaks) == 0:
                self.score[-10+_mean_start_index:10+_mean_start_index] = 0.0
                return False

            _n_peaks = len(_peaks)
            _first_peak = _first_peak+min(_peaks)-_left_extension
            _last_peak = _first_peak+max(_peaks)-_left_extension
            _previous_peaks = _peaks
        
        # Pull out time and data information
        _score = self.score[_first_peak-_left_extension:_last_peak+_right_extension]
        _data = self.data_residual[_first_peak-_left_extension:_last_peak+_right_extension]
        _time = self.t[_first_peak-_left_extension:_last_peak+_right_extension]

        # Gather useful quantities
        _data_scale = np.abs(_data.max()-_data.min())
        _range_start_t = [2.0*_default_peak_time,_default_peak_time]
        _weights= np.exp(-((_time[:,np.newaxis]-_time[_peaks])**2.0/(2.0*(10.0*_default_peak_time)**2.0))).sum(axis=1)
        
        #_PEAKS = [_p+_mean_start_index-_first_peak for _p in _peaks] # Calculate where the peaks are in the new system

        def _mod_template(T,X):
            START_T, SCALE, _PARAMS  = X[0], X[1], X[2:]
            try:
                return SCALE * self._template(T-START_T,_PARAMS)
            except RuntimeError:
                print _PARAMS

        def _super_model(T,X,_NMODEL):
            OFFSET, GRADIENT, EVENT_PARAMS = X[0], X[1], X[2:].reshape((_NMODEL,_N_PARAMS))
            _model = OFFSET + GRADIENT*(T-T[_n_points]) # The base-line is centred on the first event
            for _model_index in range(_NMODEL):
                _model += _mod_template(T,EVENT_PARAMS[_model_index])
            return _model


        def _target(X,_NMODEL=1):
            _model = _super_model(_time,X,_NMODEL)
            return (-(-0.5*(_data-_model)**2.0/NOISE**2.0)*_weights).sum()

        # Set up initial location and bounds
        _X0 = [MEDIAN,0.0]
        _bnds = [(_data.min(), _data.max()),(-1.0,1.0)]
        _N_MODEL = len(_peaks)
        _N_PARAMS = 2+len(self._template_params_names())
        for _p, _h in zip(_peaks,_peaks_heights):
            _mean_start_t = _time[_p]-_default_peak_time
            _X0 += [_mean_start_t,-10.0] + self._template_params_defaults()
            _bnds += [(_mean_start_t-_range_start_t[0], _mean_start_t+_range_start_t[1]),(1.5*_h, 0.5*_h)] + self._template_params_ranges()
        _opt = {'gtol':1e-10,'ftol':1e-10,'maxfun':100000}
        _res = optimize.minimize(_target,_X0,method='L-BFGS-B', tol=1e-10, bounds=_bnds, options=_opt, args=(_N_MODEL))

        # Pull out quantities of interest and check success
        OFFSET, GRADIENT, EVENT_PARAMS = _res['x'][0], _res['x'][1], _res['x'][2:].reshape((_N_MODEL,_N_PARAMS))
        RCHI2 = np.mean(_weights*np.abs(_data-_super_model(_time,_res['x'],len(_peaks)))/NOISE)
        if _res['success'] == False and RCHI2 > 1.5:
            self.quit_bool = True
            print _res
            plt.plot(_time,_data)
            plt.plot(_time,_super_model(_time,_res['x'],_N_MODEL))
            return False

        # We want to calculate the fourier transform over only a small section. Loop over events

        
        for _e_index in range(_N_MODEL):
            # Unpack event
            START_T, SCALE, PARAMS = EVENT_PARAMS[_e_index][0],EVENT_PARAMS[_e_index][1], EVENT_PARAMS[_e_index][2:]

            # Generate template fit and adjust running totals
            _START_T_offset = START_T%self.dt
            _fit_template_short = _mod_template(self.t_short+START_T-_START_T_offset,EVENT_PARAMS[_e_index]) # Does not contain offset.
            _f = np.fft.rfftfreq(self.t_short.size,d=self.dt)
            _ft_fit_template_short = utilities.fourier(_fit_template_short,self.dt)*np.exp(-2.0*np.pi*1j*_f*(START_T-_START_T_offset))
            _intermediate_score = np.fft.irfft(_ft_fit_template_short * self.ft_template_short.conjugate(),n=self.t_short.size)
            _event_idx = int(START_T/self.dt)
            _event_mod = int(_event_idx/self.t_short.size)
            _intermediate_score = np.concatenate([1e-18*np.ones(_event_mod*self.t_short.size),_intermediate_score])
            _intermediate_score = np.pad(_intermediate_score, (0,self.score.size-_intermediate_score.size), 'constant', constant_values=(1e-18,1e-18))
            _score_adjust = _intermediate_score/self.score_std
            _fit_template = np.concatenate([1e-18*np.ones(_event_mod*self.t_short.size+_event_idx%self.t_short.size),_fit_template_short])
            _fit_template = np.pad(_fit_template, (0,self.t.size-_fit_template.size), 'constant', constant_values=(0.0,0.0))
            self.data_residual -= _fit_template
            self.score += _score_adjust
            self.data_list.append(self.data_residual)

            # Store event in box
            _peak_score = self.score[int(START_T/self.dt)]
            self.raw_event_box['offset'].append(OFFSET)
            self.raw_event_box['gradient'].append(GRADIENT)
            self.raw_event_box['amplitude'].append(SCALE)
            self.raw_event_box['params'].append(PARAMS)
            for key, ik in zip(self._template_params_names(), range(len(self._template_params_names()))):
                self.raw_event_box[key].append(PARAMS[ik])
            self.raw_event_box['t'].append(START_T)
            self.raw_event_box['noise'].append(NOISE)
            self.raw_event_box['score'].append(_peak_score)
            self.raw_event_box['siblings'].append(_N_MODEL-1)
            self.raw_event_box['rchi2'].append(RCHI2)
        self.mask = np.union1d(self.mask,range(_first_peak-_left_extension,_last_peak+_right_extension)).astype(np.int_)
        self.score[self.mask] = 0.0
    
    
    def _process_event_box(self):
        # Turn values into numpy arrays
        for _k,_v in self.raw_event_box.items():
            self.raw_event_box[_k] = np.array(_v)            
                
    def _post_process_event_box(self,event_threshold):
        # Remove events below event threshold
        _raw_event_box_N = len(self.raw_event_box['amplitude'])
        self.raw_event_box['peak_time'] = np.zeros(_raw_event_box_N)
        self.raw_event_box['charge'] = np.zeros(_raw_event_box_N)
        self.raw_event_box['half_life'] = np.zeros(_raw_event_box_N)
        for ie in range(_raw_event_box_N):
            self.raw_event_box['peak_time'][ie] = self._template_peak(PARAMS = self.raw_event_box['params'][ie])
            self.raw_event_box['charge'][ie] = np.abs(self.raw_event_box['amplitude'][ie]*self._template_area(PARAMS = self.raw_event_box['params'][ie]))
            self.raw_event_box['half_life'][ie] = utilities._template_half_life(TEMPLATE = self._template, TEMPLATE_PEAK = self._template_peak, PARAMS = self.raw_event_box['params'][ie])
        _valid_events = np.where(((self.raw_event_box['amplitude'])<=-event_threshold['current_threshold'])&(-(self.raw_event_box['amplitude'])/self.raw_event_box['noise']>event_threshold['significance_threshold'])&(self.raw_event_box['charge']>event_threshold['charge_threshold'])&(self.raw_event_box['rchi2']<event_threshold['rchi2_threshold']))
        self.event_box = copy.copy(event_threshold)
        for _k,_v in self.raw_event_box.items():
            self.event_box[_k] = _v[_valid_events]
        
        # Count number of events
        self.event_box['N'] = self.event_box['rise'].size
        
        self.event_box['max_time'] = self.event_box['t'] + self.event_box['peak_time']
        self.event_box['max_idx'] = (self.event_box['max_time']/self.dt).astype(np.int_)
        

    def _show_mini_events(self):
        # Place markers
        stf.erase_markers()
        for _i in range(self.event_box['N']):
            stf.set_marker(self.event_box['max_idx'][_i],-self.event_box['amplitude'][_i])

        # Create results table
        _result_box = {}
        _interesting = ['t','score','amplitude','charge','peak_time','half_life']+self._template_params_names()
        for k in _interesting:
            _result_box[k] = list(self.event_box[k])
        self.result_box = _result_box
        stf.show_table_dictlist(self.result_box,caption='Mini event table')

        #stf.new_window_list([self.data,self.data_residual])

        # new window with all events
        _event_list = [self.data[int(t/self.dt) - 100:int(t/self.dt) + 600] for t in self.event_box['t']]
        #stf.new_window_list(_event_list)

    def run(self):
        # Create template
        self.template = self._template(self.t-self.t[0],PARAMS=self._template_params_defaults())
        self.template_short = self.template[:self.event_window*5]
        
        # Calculate background noise level
        self.noise_med, self.noise_std = utilities.rolling(self.data,FUNC='SUFFICIENT_STATISTICS',WINDOW=10*self.median_window)
        print "Classified background noise"

        # Get Fourier transform of data
        t = time.time()
        self.ft_data = utilities.fourier(self.data,self.dt)
        print time.time()-t
        
        # Get Fourier transform of template
        self.ft_template = utilities.fourier(self.template,self.dt)
        self.ft_template_short = utilities.fourier(self.template_short,self.dt)
        
        # Get score_of_events
        self._initial_classify_events()
        print "Completed initial classification of events"

        # Create copy of data residuals
        self.data_residual = copy.copy(self.data)
        self.ft_data_residual = copy.copy(self.ft_data)
        
        # Loop over top N_events
        loop_tracker = False
        count_events = 0
        self.last_event_index = 0
        self.quit_bool = False
        
        self.raw_event_box = {'params':[],'offset':[],'gradient':[],'amplitude':[],'t':[],'noise':[],'score':[],'siblings':[],'rchi2':[]}
        for key in self._template_params_names():
                self.raw_event_box[key]=[]
        while loop_tracker == False:
            # Decide whether to continue
            index_of_event, significant_bool = self._decide_strategy()
            if significant_bool == False:
                loop_tracker = True
                continue
            count_events+=1
            if count_events>=self.event_number_barrier or self.quit_bool == True:
                loop_tracker = True
                continue
            
            if self.print_bool == True:
                try:
                    print count_events,index_of_event,self.t[index_of_event]
                except IOError:
                    # Weird error in StimFit...
                    self.print_bool = False
            # Fit event
            self._fit_and_remove_event(index_of_event,_n_points=int(10.0/self.dt))
            #print t2-t1
            
            #print OFFSET,SCALE,RISE,DECAY,START_T,NOISE
            # Emergency brake
            
                
            
            #plt.plot(self.t,self.cleaned_data)
            #plt.plot(self.t,self.clean_data)
            #plt.show()
            self.last_event_index = index_of_event
                
        # Process event box
        self._process_event_box()

        # Post process event box
        self._post_process_event_box(self.event_threshold)

        # Show results
        self._show_mini_events()
        
        # Logging
        print "Found",self.raw_event_box['rise'].size,"events."