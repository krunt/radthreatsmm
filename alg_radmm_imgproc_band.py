
import alg_radmm_base
import numpy as np
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
from utils import denoise_signal

TOFFS = 30
MAX_ENERGY = 2500
ENERGY_STEP = 500
EBINS = int(128 * (MAX_ENERGY / ENERGY_STEP))
KEV_PER_EBIN = MAX_ENERGY / EBINS

SOURCE_THRESH = [ 0, 4.0, 3.5, 4.0, 5.5, 6.0, 6.0, ]
SOURCE_NEI1_THRESH = [ 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]

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
                hist = denoise_signal(hist)
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

                    zscores1 = [0]*tmax
                    for j in range(1,len(zscores)-1):
                        if zscores[j] < SOURCE_THRESH[source]:
                            continue
                        if zscores[j] < SOURCE_NEI1_THRESH[source] * zscores[j-1]:
                            continue
                        if zscores[j] < SOURCE_NEI1_THRESH[source] * zscores[j+1]:
                            continue
                        zscores1[j] = zscores[j]

                    idx = np.argmax(zscores1)
                    if zscores1[idx]:
                        final_zscores.append([zscores1[idx], idx, source])

            if final_zscores:
                idx = np.argmax(final_zscores, axis=0)[0]
                ret[i, 0] = final_zscores[idx][2]
                ret[i, 1] = final_zscores[idx][1] + TOFFS + 0.5
        return ret
                
