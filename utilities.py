""" Common utilities used by multiple scripts. """

# Dependencies
import stf
import numpy as np
import pandas as pd
from scipy import optimize
import json
from math import exp

# Load in controls
with open('controlpanel.json') as f:
    _control = json.load(f)['utilities']

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

# Biexponential template and related functions
def _biexponential(T,PARAMS=(1.0,1.0)):
    RISE, DECAY = PARAMS
    NORM = 1/((DECAY/(RISE+DECAY))*(RISE/(RISE+DECAY))**(RISE/DECAY))
    retarray = np.zeros(T.shape[0])
    postime = T>0.
    retarray[postime] = NORM*(1.0-np.exp(-T[postime]/RISE))*np.exp(-T[postime]/DECAY)
    return retarray

def _biexponential_peak(PARAMS=(1.0,1.0)):
    RISE, DECAY = PARAMS
    return -RISE*np.log(RISE/(RISE+DECAY))

def _biexponential_area(PARAMS=(1.0,1.0)):
    RISE, DECAY = PARAMS
    return DECAY/((RISE/(RISE+DECAY))**(RISE/DECAY))

def _biexponential_params_names():
    return ['rise','decay']

def _biexponential_params_ranges():
    return [(1e-10,1e5),(1e-1,3e1)]

def _biexponential_params_defaults():
    return [_control['biexponential_params_defaults'][k] for k in _biexponential_params_names()]

# Triexponential template and related functions
def _triexponential_peak(PARAMS=(1.0,1.0,1.0,0.5)):
    RISE1, RISE2, DECAY, F = PARAMS[0], PARAMS[0]*PARAMS[1], PARAMS[2], PARAMS[3]
    G0_A = F*(1.0+DECAY/RISE1)
    G0_B = (1.0-F)*(1.0+DECAY/RISE2)
    G1_A = -G0_A/RISE1
    G1_B = -G0_B/RISE2
    G2_A = -G1_A/RISE1
    G2_B = -G1_B/RISE2
    def _G0(x):
        return -1.0+G0_A*exp(-x/RISE1)+G0_B*exp(-x/RISE2)
    def _G1(x):
        return G1_A*exp(-x/RISE1)+G1_B*exp(-x/RISE2)
    def _G2(x):
        return G2_A*exp(-x/RISE1)+G2_B*exp(-x/RISE2)

    root = optimize.newton(_G0, _biexponential_peak(PARAMS=[RISE2,DECAY]), fprime=_G1,fprime2=_G2,maxiter=1000)
    return root

def _triexponential(T,PARAMS=(1.0,2.0,1.0,0.5),NORMED=True):
    RISE1, RISE2, DECAY, F = PARAMS[0], PARAMS[0]*PARAMS[1], PARAMS[2], PARAMS[3]
    retarray = np.zeros(T.shape[0])
    postime = T>0.
    retarray[postime] = (1.0-F*np.exp(-T[postime]/RISE1)-(1.0-F)*np.exp(-T[postime]/RISE2))*(np.exp(-T[postime]/DECAY))
    if NORMED == True:
        PEAK = _triexponential_peak(PARAMS)
        NORM = _triexponential(np.array([PEAK]),PARAMS,NORMED=False)[0]
        retarray /= NORM
    return retarray

def _triexponential_area(PARAMS=(1.0,2.0,1.0,0.5),NORMED=True):
    RISE1, RISE2, DECAY, F = PARAMS[0], PARAMS[0]*PARAMS[1], PARAMS[2], PARAMS[3]
    AREA = ((DECAY+(1.0-F)*RISE1+F*RISE2)/((1.0+RISE1/DECAY)*(1.0+RISE2/DECAY)))
    if NORMED == True:
        PEAK = _triexponential_peak(PARAMS)
        NORM = _triexponential(np.array([PEAK]),PARAMS,NORMED=False)[0]
        AREA /= NORM
    return AREA

def _triexponential_params_names():
    return ['rise','eta','decay','f']

def _triexponential_params_ranges():
    return [(1e-10,1e3),(1.0,1e2),(1e-1,1e1),(0.001,0.999)]

def _triexponential_params_defaults():
    return [_control['triexponential_params_defaults'][k] for k in _triexponential_params_names()]

# Returns the relevant functions for calculating a template
def obtain_template(TEMPLATE_NAME='biexponential'):
    if TEMPLATE_NAME == 'biexponential':
        return _biexponential, _biexponential_peak, _biexponential_area, _biexponential_params_names, _biexponential_params_ranges, _biexponential_params_defaults
    elif TEMPLATE_NAME == 'triexponential':
        return _triexponential, _triexponential_peak, _triexponential_area, _triexponential_params_names, _triexponential_params_ranges, _triexponential_params_defaults

def _template_half_life(TEMPLATE,TEMPLATE_PEAK,PARAMS):
    _PEAK = TEMPLATE_PEAK(PARAMS)
    def _G0(x):
        if x<0:
            return 1.0
        else:
            #return NORM*(1.0-exp(-(1.0+x)*t_peak/RISE))*exp(-(1.0+x)*t_peak/DECAY) - 0.5
            return TEMPLATE(np.array([(1.0+float(x))*_PEAK]),PARAMS)-0.5
    _ROOT = optimize.newton(_G0, 0.1,maxiter=1000)
    return float(_ROOT*_PEAK)