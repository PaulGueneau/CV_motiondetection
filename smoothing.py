import numpy as np
import cv2
# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html


def movingAverage(x,radius):
    window_size = 2*radius+1
    moving_filter = np.ones(window_size)/window_size
    gaussian_filter = cv2.getGaussianKernel(window_size,1)
    gaussian_filter = gaussian_filter.reshape(window_size,)
    padding = np.lib.pad(x,(radius,radius),'edge')
    smoothed_x = np.convolve(padding,moving_filter,mode='same')
    smoothed_x = smoothed_x[radius:-radius]
    return smoothed_x
def smooth(trajectory):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    smoothed_trajectory = np.copy(trajectory)
    for i in range(2):
        #smoothed_trajectory[:,i] = movingAverage(trajectory[:,i],radius=10)
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i],50)

    return smoothed_trajectory





def medianFilter(x):
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(x,-1,kernel)
    return dst
    #if x.ndim != 1:
     #   raise ValueError

    #if x.size < window_len:
      #  raise ValueError


    #if window_len<3:
     #   return x


    #if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
       # raise ValueError


    #s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    #if window == 'flat': #moving average
     #   w=numpy.ones(window_len,'d')/window_len
    #else:
     #   w=eval('numpy.'+window+'(window_len)')
    
    

    

   


  

    #y=numpy.convolve(w/w.sum(),s,mode='valid')
    
   




