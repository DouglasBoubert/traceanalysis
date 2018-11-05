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
import json

# Identify bigevents
class EventAnalysis:
    def __init__(self, trace=stf.get_trace()):
        # Initial sorting of data
        self.data = []
        self.data.append(trace)
        self.mask = np.array([])
        with open('document.json') as f:
            self.control = json.load(f)

    def _mask_update(self,_new_mask):
        self.mask = np.union1d(self.mask,_new_mask).astype(np.int_)

    def run(self):
        # Runs the analysis

        # Handle the big events
        if self.control['bigevents']['active']:
            BEH = bigevents.BigEventHandler(self.data[-1],min_charge = self.control['bigevents']['min_charge'], event_time = self.control['bigevents']['event_time'], median_window=self.control['bigevents']['median_window'])
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
            NS = noisestripper.NoiseStripper(self.data[-1],median_window=1.0)
            NS.run()
            self.data.append(NS.data_squeaky_clean)
            _start_end_mask = np.concatenate([np.arange(0,NS.start_end_offset),np.arange(self.data[0].size-NS.start_end_offset,self.data[0].size)])
            self._mask_update(_start_end_mask)
        # Plot the results
        stf.new_window_list(self.data)
