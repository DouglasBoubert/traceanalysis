""" Wrapper script to run all of the procedures. """

##### Warning: This script *will* take a while to run. Go get coffee.

# Dependencies
import stf
import numpy as np
import utilities
import copy
import utilities
import bigevents
import noisestripper
import dropoutfixer
import minievents
import json

# Identify bigevents
class TraceAnalysis:
    def __init__(self, trace):
        # We are going to be taking many fourier transforms.
        # The standard FFT in numpy is slow for numbers with large prime factors.
        # We cut the trace down until the largest of its prime factors is < 100.
        def cut(_N,_max_factor = 100):
            def largest_prime_factor(n):
                i = 2
                while i * i <= n:
                    if n % i:
                        i += 1
                    else:
                        n //= i
                return n
            _CUT = 0
            while largest_prime_factor(_N+_CUT) > _max_factor:
                _CUT-=1
            return _CUT
        _cut = cut(trace.size)

        # Initial sorting of data
        self.data = []
        if _cut < 0:
            self.data.append(trace[:_cut])
        else:
            self.data.append(trace)
        self.mask = np.array([])
        with open('controlpanel.json') as f:
            self.control_default = json.load(f)

    def _mask_update(self,_new_mask):
        self.mask = np.union1d(self.mask,_new_mask).astype(np.int_)

    def run(self,CONTROL_OVERRIDE=None):
        # Runs the analysis
        self.control = copy.copy(self.control_default)
        if CONTROL_OVERRIDE != None:
            for _process in CONTROL_OVERRIDE.keys():
                for _key in CONTROL_OVERRIDE[_process].keys():
                    self.control[_process][_key] = CONTROL_OVERRIDE[_process][_key]
        # Handle the big events
        if self.control['bigevents']['active']:
            BEH = bigevents.BigEventHandler(self.data[-1],min_charge = self.control['bigevents']['min_charge'], event_time = self.control['bigevents']['event_time'], median_window=self.control['bigevents']['median_window'], fixing_interval=self.control['bigevents']['fixing_interval'])
            BEH.run()
            self.data.append(BEH.data_residual)
            self._mask_update(BEH.big_event_mask)

        # Fix dropouts
        if self.control['dropoutfixer']['active']:
            DF = dropoutfixer.DropoutFixer(self.data[-1])
            DF.run()
            self.data.append(DF.data_residual)

        # Strip the median and noise
        if self.control['noisestripper']['active']:
            NS = noisestripper.NoiseStripper(self.data[-1],median_window=self.control['noisestripper']['median_window'])
            NS.run()
            self.data.append(NS.data_squeaky_clean)
            _start_end_mask = np.concatenate([np.arange(0,NS.start_end_offset),np.arange(self.data[0].size-NS.start_end_offset,self.data[0].size)])
            self._mask_update(_start_end_mask)

        # Identify and fit mini events
        if self.control['minievents']['active']:
            # It is possible to get to here and have no properly defined mask.
            if self.mask.size == 0:
                _default_start_end_mask = self.control['minievents']['default_start_end_mask']
                _start_end_mask = np.concatenate([np.arange(0,_default_start_end_mask),np.arange(self.data[0].size-_default_start_end_mask,self.data[0].size)])
                self._mask_update(_start_end_mask)
            MEH = minievents.MiniEventHandler(self.data[-1], mask=self.mask, template_name = self.control['minievents']['template_name'], event_time = self.control['minievents']['event_time'], median_window = self.control['minievents']['median_window'], bayes_bound = self.control['minievents']['bayes_bound'], bayes_weight = self.control['minievents']['bayes_weight'], score_bound = self.control['minievents']['score_bound'], min_peak_current = self.control['minievents']['min_peak_current'], event_number_barrier = self.control['minievents']['event_number_barrier'], event_threshold = self.control['minievents']['event_threshold'])
            MEH.run()
            self.data.append(MEH.data_residual)
            self.event_box = MEH.event_box
            self.raw_event_box = MEH.raw_event_box
            self.meh_noise = MEH.noise_std
            print(self.control['minievents']['median_window'])
        # Plot the results
        stf.new_window_list(self.data)

def __main__():
    TA = TraceAnalysis(trace=stf.get_trace())
    TA.run()
    return True

def __big_and_noise__():
    TA = TraceAnalysis(trace=stf.get_trace())
    CONTROL_OVERRIDE = {"bigevents":{"active":True},"dropoutfixer":{"active":False},"noisestripper":{"active":True},"minievents":{"active":False}}
    TA.run(CONTROL_OVERRIDE=CONTROL_OVERRIDE)
    return True

def __noise_and_mini__():
    TA = TraceAnalysis(trace=stf.get_trace())
    CONTROL_OVERRIDE = {"bigevents":{"active":False},"dropoutfixer":{"active":False},"noisestripper":{"active":True},"minievents":{"active":True}}
    TA.run(CONTROL_OVERRIDE=CONTROL_OVERRIDE)
    return True

def __noise_only__():
    TA = TraceAnalysis(trace=stf.get_trace())
    CONTROL_OVERRIDE = {"bigevents":{"active":False},"dropoutfixer":{"active":False},"noisestripper":{"active":True},"minievents":{"active":False}}
    TA.run(CONTROL_OVERRIDE=CONTROL_OVERRIDE)
    return True

def __mini_only__():
    TA = TraceAnalysis(trace=stf.get_trace())
    CONTROL_OVERRIDE = {"bigevents":{"active":False},"dropoutfixer":{"active":False},"noisestripper":{"active":False},"minievents":{"active":True}}
    TA.run(CONTROL_OVERRIDE=CONTROL_OVERRIDE)
    return True

def __big_only__():
    TA = TraceAnalysis(trace=stf.get_trace())
    CONTROL_OVERRIDE = {"bigevents":{"active":True},"dropoutfixer":{"active":False},"noisestripper":{"active":False},"minievents":{"active":False}}
    TA.run(CONTROL_OVERRIDE=CONTROL_OVERRIDE)
    return True

def __very_big_only__():
    TA = TraceAnalysis(trace=stf.get_trace())
    CONTROL_OVERRIDE = {"bigevents":{"active":True,"median_window":-1.0,"fixing_interval":300},"dropoutfixer":{"active":False},"noisestripper":{"active":False},"minievents":{"active":False}}
    TA.run(CONTROL_OVERRIDE=CONTROL_OVERRIDE)
    return True