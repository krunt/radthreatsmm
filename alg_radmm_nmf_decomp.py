
import alg_radmm_base
import numpy as np
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import denoise_signal
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier

NCOMP_BG = 8
NTRAIN_BG = 128
MAX_ENERGY = 2500
EBINS = 128
KEV_PER_EBIN = int(MAX_ENERGY / EBINS)
SIGNAL_THRESHOLD = 1.8
BG_THRESHOLD = 13.0
SIGNAL5_THRESHOLD = 0.4

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
        self.source_hist = np.zeros((10,EBINS))
        for shielding in range(2):
            for source in range(5):
                arr = []
                for binidx in range(EBINS):
                    energyFrom = binidx / EBINS * MAX_ENERGY
                    energyTo = (binidx + 1) / EBINS * MAX_ENERGY
                    dat = sdata[(sdata["Shielding"] == shielding) & (sdata["SourceID"] == source + 1) & (sdata["PhotonEnergy"] > energyFrom) & (sdata["PhotonEnergy"] < energyTo)]
                    arr.append(dat["CountRate"].mean())
                self.source_hist[shielding * 5 + source, :] = np.abs(denoise_signal(np.array(arr)))
                self.source_hist[shielding * 5 + source, :] /= np.max(self.source_hist[shielding * 5 + source, :])

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
        runid_list = []
        for (i,runid) in enumerate(ids):
            if self._train_metadata.loc[runid]["SourceID"] == 0:
                runid_list.append((i,runid))
        np.random.shuffle(runid_list)
        runid_list = runid_list[:NTRAIN_BG]

        xlist = []
        for (idx,runid) in runid_list:
            xlist.append(np.abs(x[idx]))

        xlist = np.vstack(xlist)

        ncomp_bg = NCOMP_BG
        self.model_bg = NMF(ncomp_bg, init='random', random_state=0)
        self.model_bg.fit(xlist)
        self.comps_bg = self.model_bg.components_

        self.model_bgs = NMF(ncomp_bg+10, init='random', random_state=0)
        self.model_bgs.fit(xlist)
        for i in range(ncomp_bg):
            self.model_bgs.components_[i] = self.model_bg.components_[i]
        for i in range(10):
            self.model_bgs.components_[-10 + i] = self.source_hist[i]
        self.comps_bgs = self.model_bgs.components_

        self.model_arr_bgs = []
        for i in range(6):
            naddcomp = 4 if i == 5 else 2
            self.model_arr_bgs.append(NMF(ncomp_bg+naddcomp, init='random', random_state=0))
            self.model_arr_bgs[-1].fit(xlist)
            for j in range(ncomp_bg):
                self.model_arr_bgs[-1].components_[j] = self.model_bg.components_[j]
            if i == 5:
                self.model_arr_bgs[-1].components_[-naddcomp:] = (self.source_hist[0], self.source_hist[5], 
                        self.source_hist[4], self.source_hist[9])
            else:
                self.model_arr_bgs[-1].components_[-naddcomp:] = (self.source_hist[i], self.source_hist[i+5])

    def predict(self, x, ids):
        ret = np.zeros((len(ids), 2))

        for i in tqdm(range(len(ids))):
            id = ids[i]
            dat = np.abs(x[i])
            tmax = dat.shape[0]

            weigh = self.model_bg.transform(dat[30:])

            weigh_arr_s = []
            for source in range(len(self.model_arr_bgs)):
                weigh_arr_s.append(self.model_arr_bgs[source].transform(dat[30:]))

            arr = []
            tiarr = []
            sourcearr = []
            for ti in range(30, tmax):
                fit_bg = np.dot(weigh[ti - 30], self.comps_bg)
                norm_bg = np.linalg.norm(fit_bg - dat[ti, :])

                sres = []
                sres_bgs = []
                for source in range(len(self.model_arr_bgs)):
                    fit_bgs = np.dot(weigh_arr_s[source][ti - 30], self.model_arr_bgs[source].components_)
                    norm_bgs = np.linalg.norm(fit_bgs - dat[ti, :])
                    if source == 5:
                        warr = weigh_arr_s[source][ti - 30]
                        w01 = np.abs(warr[0] + warr[1])
                        w23 = np.abs(warr[2] + warr[3])
                        if w01 / (w01 + w23) < SIGNAL5_THRESHOLD:
                            continue

                    sres.append(norm_bg / norm_bgs)
                    sres_bgs.append(norm_bgs)

                if sres:
                    sresi = np.argmax(sres)
                    coeff = 1.1 if sresi == 5 else 1.0
                    if sres_bgs[sresi] > BG_THRESHOLD * coeff and sres[sresi] > SIGNAL_THRESHOLD * coeff:
                        arr.append(sres[sresi])
                        tiarr.append(ti)
                        sourcearr.append(sresi)

            if arr:
                idx = np.argmax(arr)
                ti = tiarr[idx]
                si = sourcearr[idx]
                ret[i, 0] = 1 + si
                ret[i, 1] = ti + 0.5

        return ret
