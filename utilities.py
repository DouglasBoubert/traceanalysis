""" Common utilities used by multiple scripts. """

# Dependencies
import stf
import numpy as np
import pandas as pd


# Rolling functions
def rolling(DATA,WINDOW=50,FUNC='MEDIAN',EDGE_METHOD='SHIFTING'):
    def cleaning(RAW_ROLLING):
        RAW_ROLLING[:WINDOW] = np.roll(RAW_ROLLING,-WINDOW)[:WINDOW]
        RAW_ROLLING[-WINDOW:] = np.roll(RAW_ROLLING,WINDOW)[-WINDOW:]
        return RAW_ROLLING
    
    if FUNC=='MEDIAN':
        raw_rolling = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).median().values.flatten()
    elif FUNC=='MEAN':
        raw_rolling = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).mean().values.flatten()
    elif FUNC=='MAX':
        raw_rolling = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).max().values.flatten()
    elif FUNC=='MIN':
        raw_rolling = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).min().values.flatten()
    elif FUNC=='STD':
        raw_rolling = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).std().values.flatten()
    elif FUNC=='SUM':
        raw_rolling = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).sum().values.flatten()
    elif FUNC=='SMOOTH_MEDIAN':
        raw_rolling_q1 = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).quantile(0.40).values.flatten()
        raw_rolling_q3 = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).quantile(0.60).values.flatten()
        raw_rolling = np.median(np.stack([raw_rolling_q1,raw_rolling_q3]),axis=0)
    elif FUNC=='SUFFICIENT_STATISTICS':
        raw_rolling_q1 = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).quantile(0.25).values.flatten()
        raw_rolling_q2 = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).quantile(0.50).values.flatten()
        raw_rolling_q3 = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).quantile(0.75).values.flatten()
        raw_rolling_std = (raw_rolling_q3-raw_rolling_q1)/1.349
        return cleaning(raw_rolling_q2),cleaning(raw_rolling_std)
    elif FUNC=='SUFFICIENT_STATISTICS_LEFT':
        raw_rolling_q1 = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).quantile(0.25).values.flatten()
        raw_rolling_q2 = pd.DataFrame(DATA).rolling(window=WINDOW,center=True).quantile(0.50).values.flatten()
        raw_rolling_std = (raw_rolling_q2-raw_rolling_q1)*2.0/1.349
        return cleaning(raw_rolling_q2),cleaning(raw_rolling_std)
        
    return cleaning(raw_rolling)

# Median and interquartile-derived standard deviation
def sufficient_statistics(DATA):
	percentiles = np.percentile(DATA,[25,50,75])
	return percentiles[1], (percentiles[2]-percentiles[0])/1.349

# Fourier transform
def fourier(TARGET,DT):
    return DT * np.fft.rfft(TARGET)

# Return dt in seconds accounting for the Stimfit unit
def dt():
	stf_xunits = stf.get_xunits()
	stf_dt = stf.get_sampling_interval()
	if stf_xunits == 's':
		return stf_dt
	if stf_xunits == 'ms':
		return stf_dt*1e-3
	elif stf_xunits == 'ns':
		return stf_dt*1e-6
	else:
		print 'Cannot interpret the response of stf.get_xunits().'

# Splits an array into consecutive blocks
def consecutive(DATA, STEPSIZE=1):
    return np.split(DATA, np.where(np.diff(DATA) != STEPSIZE)[0]+1)