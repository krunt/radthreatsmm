
import alg_radmm_base
import numpy as np
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
from utils import denoise_signal
from sklearn.decomposition import NMF

NCOMP = 3
MAX_ENERGY = 2500
EBINS = 128
KEV_PER_EBIN = int(MAX_ENERGY / EBINS)
THRESHOLD = 3.3

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

class AlgRadMMNmfDecomp(alg_radmm_base.AlgRadMMBase):
    def __init__(self, base_path):
        alg_radmm_base.AlgRadMMBase.__init__(self, base_path)

        sdata = self._source_data
        self.source_hist = np.zeros((7,EBINS))
        for source in range(1,6):
            arr = []
            for binidx in range(EBINS):
                energyFrom = binidx / EBINS * MAX_ENERGY
                energyTo = (binidx + 1) / EBINS * MAX_ENERGY
                dat = sdata[(sdata["SourceID"] == source) & (sdata["PhotonEnergy"] > energyFrom) & (sdata["PhotonEnergy"] < energyTo)]
                arr.append(dat["CountRate"].mean())
            self.source_hist[source] = np.array(arr)

    def _prepare(self, ids, is_train=True, validation=False, cache=True):
        filename = "train_nmf.pkl" if is_train else "test_nmf.pkl"
        if validation:
            filename = "validation_nmf.pkl"
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
            zmat = np.zeros((tmax,bins))

            ebins = np.linspace(0,MAX_ENERGY,bins)

            for i in range(tmax):
                dind = np.argwhere((d1 > i) & (d1 < (i + 1))).flatten() 
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
            dat = np.abs(x[i])
            tmax = dat.shape[0]

            model = NMF(NCOMP, init='random', random_state=0)
            model.fit(dat) #[:30])
            W = model.transform(dat)

            fitS = np.dot(W, model.components_)

            zscores = []
            for i in range(tmax):
                zscores.append(np.linalg.norm(fitS[i] - dat[i]))
            zscores = np.array(zscores)

            mean_z = np.mean(zscores)
            std_z = np.std(zscores)

            for i in range(tmax):
                zscores[i] = (zscores[i] - mean_z) / std_z

            idx = np.argmax(zscores)

            if zscores[idx] > 5.0 and idx >= 30:
                ret[i, 0] = 3
                ret[i, 1] = idx + 0.5

        return ret


#        ret = np.zeros((len(ids), 2))
#        for (i,id) in enumerate(ids):
#            dat = np.abs(x[i])
#            tmax = dat.shape[0]
#
#            model = NMF(NCOMP, init='random', random_state=0)
#            model.fit(dat) #[:30])
#            W = model.transform(dat)
#
#            fitS = np.dot(W, model.components_)
#
#            zscores = []
#            for i in range(tmax):
#                zscores.append(np.linalg.norm(fitS[i] - dat[i]))
#            zscores = np.array(zscores)
#
#            mean_z = np.mean(zscores)
#            std_z = np.std(zscores)
#
#            for i in range(tmax):
#                zscores[i] = (zscores[i] - mean_z) / std_z
#
#            idx = np.argmax(zscores)
#
#            if zscores[idx] > 5.0 and idx >= 30:
#                ret[i, 0] = 3
#                ret[i, 1] = idx + 0.5
#
#        return ret
#
