
import alg_radmm_base
import numpy as np
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
import pywt

TOFFS = 30
MAX_ENERGY = 2500
ENERGY_STEP = 500
EBINS = int(128 * (MAX_ENERGY / ENERGY_STEP))
KEV_PER_EBIN = MAX_ENERGY / EBINS

SOURCE_THRESH = [ 0, 4.0, 3.5, 4.0, 5.5, 6.0, 6.0, ]

ENABLED_SOURCES = [1,2,3,4,5,6]

SOURCE_METADATA = [ 
    [], #0
    [[87,107],[170,200]], #1
    [[50,75]], #2
    [[350,380]], #3
    [[1160,1190],[1320,1350]], #4
    [[125,155]], #5
    [], #6
]

def _maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def _denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    sigma = (1/0.6745) * _maddest( coeff[-level] )
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    return pywt.waverec( coeff, wavelet, mode='per' )


class AlgRadMMImgProcBand(alg_radmm_base.AlgRadMMBase):
    def __init__(self, base_path):
        alg_radmm_base.AlgRadMMBase.__init__(self, base_path)
    def _prepare(self, ids, is_train=True, validation=False, cache=True):
        filename = "train.pkl" if is_train else "test.pkl"
        if validation:
            filename = "validation.pkl"
        tcache_path = os.path.join(self._base_path, filename)
        if os.path.exists(tcache_path):
            fd = open(tcache_path, "rb")
            ret = pkl.load(fd)
            fd.close()
            return ret
        tpath = self._train_dir_path if is_train else self._test_dir_path
        ret = []
        for id in tqdm(ids):
            dat = pd.read_csv(os.path.join(tpath, "%d.csv" % id), header=None)

            d0=dat[0]*1e-6
            d1=np.cumsum(d0)
            d2=dat[1]
            tmax=int(d1.values[-1])

            bins = EBINS
            zmat = np.zeros((tmax-TOFFS,bins))

            ebins = np.linspace(0,MAX_ENERGY,bins)

            for i in range(tmax-TOFFS):
                dind = np.argwhere((d1 > (TOFFS + i)) & (d1 < (TOFFS + i + 1))).flatten() 
                d3 = d2[dind]
                hist = np.histogram(d3, bins=ebins)[0]
                hist = _denoise_signal(hist)
                zmat[i,:] = hist
            ret.append(zmat)
        if cache:
            fd = open(tcache_path, "wb")
            pkl.dump(ret, fd)
            fd.close()
        return ret

    def get_train_x(self, ids, validation):
        return self._prepare(ids, is_train=True, validation=validation)
    def get_test_x(self, ids):
        return self._prepare(ids, is_train=False)
    def train(self, x, y, ids):
        pass
    def predict(self, x, ids):
        ret = np.zeros((len(ids), 2))
        for (i,id) in enumerate(ids):
            dat = x[i]
            tmax = dat.shape[0]
            final_zscores = []
            for source in ENABLED_SOURCES:
                for ridx in range(len(SOURCE_METADATA[source])):
                    range_t = SOURCE_METADATA[source][ridx]
                    from_t = int(range_t[0] / KEV_PER_EBIN)
                    to_t = int(range_t[1] / KEV_PER_EBIN)
                    mean_z = np.mean(dat[:, from_t:to_t])
                    std_z = np.std(dat[:, from_t:to_t]) + 1e-9

                    zscores = []
                    for tcur in range(tmax):
                        mean_c = np.mean(dat[tcur, from_t:to_t])
                        zscores.append(np.abs((mean_c - mean_z) / std_z))
                    idx = np.argmax(zscores)
                    if zscores[idx] > SOURCE_THRESH[source]:
                        final_zscores.append([zscores[idx], idx, source])

            if final_zscores:
                idx = np.argmax(final_zscores, axis=0)[0]
                ret[i, 0] = final_zscores[idx][2]
                ret[i, 1] = final_zscores[idx][1] + TOFFS + 0.5
        return ret
                
