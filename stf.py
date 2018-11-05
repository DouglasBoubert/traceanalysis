import numpy as np
from neo.io import WinEdrIO
import matplotlib.pyplot as plt

##### Load in file and format
print("Reading in file")
#targetfile = '/Users/douglasboubert/Documents/Science/AlisonNeuro/Data/180502_003.1.edr'
#targetfile = '/Users/douglasboubert/Documents/Science/AlisonNeuro/Data/180904_006.1[LP=5327Hz RD=20].edr'
#targetfile = '/Users/douglasboubert/Documents/Science/AlisonNeuro/Data/180904_006.1[LP=5327Hz RD=20].edr'
targetfile = '/Users/douglasboubert/Documents/Science/AlisonNeuro/Data/180927_001[LP=5327Hz RD=20]40-100s.edr'
#targetfile = '/Users/douglasboubert/Documents/Science/AlisonNeuro/Data/180919_005.1[LP=5327Hz RD=20].edr'

reader = WinEdrIO(filename=targetfile)
sampling_rate = reader.get_signal_sampling_rate()
t_start,t_stop = reader.segment_t_start(0,0),reader.segment_t_stop(0,0)
n_samples = int(round((t_stop-t_start)*sampling_rate))
raw_sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0,i_start=0, i_stop=n_samples, channel_indexes=None)
float_sigs = reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64')
XRAW = np.linspace(t_start,t_stop,n_samples)
YRAW = np.array(float_sigs.tolist()).T.ravel()

##### To-do smooth signal properly
def compress(ARRAY,WINDOW=1):
    if WINDOW==1:
        return ARRAY
    else:
        return ARRAY[:-(ARRAY.shape[0]%WINDOW)].reshape(-1, WINDOW).mean(axis=1)

X = compress(XRAW)
Y = compress(YRAW)

firsthalf = np.where(X<34.0)
X = X[firsthalf]
Y = Y[firsthalf]

def get_trace():
    return Y

def erase_markers():
    return True

def set_marker(X,Y):
    return True

def show_table_dictlist(B,caption='Mini event table'):
    return True

def new_window_list(L):
    x = range(L[0].size)
    for l in L:
        plt.plot(x,l)

def get_sampling_interval():
    return (X[1]-X[0])*1e3

def get_xunits():
    return 'ms'

