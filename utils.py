
import pywt
import numpy as np

def _maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    sigma = (1/0.6745) * _maddest( coeff[-level] )
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    return pywt.waverec( coeff, wavelet, mode='per' )

def denoise_signal_stub(x, wavelet='db4', level=1):
    return x

def cross_correlation(x, y):
    mx = np.mean(x)
    my = np.mean(y)

    dx = (x.flatten() - mx)
    dy = (y.flatten() - my)

    dxx = np.sqrt(np.dot(dx, dx))
    dyy = np.sqrt(np.dot(dy, dy))

    return np.dot(dx, dy) / (dxx * dyy)
