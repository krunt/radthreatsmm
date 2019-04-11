
import alg_radmm_base
import numpy as np
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import denoise_signal, denoise_signal_stub
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier

NCOMP_BG = 12
NTRAIN_BG = 128
MAX_ENERGY = 2500
EBINS = 128
KEV_PER_EBIN = int(MAX_ENERGY / EBINS)
SIGNAL_THRESHOLD = 1.4
BG_THRESHOLD = 10.0
SIGNAL_COEFF = [1.0,1.1,1.0,0.9,1.1,1.5]
#SIGNAL_THRESHOLD_ARR = [1.4,1.3,1.5,1.3,1.5,1.6]
#SIGNAL_THRESHOLD_ARR = [1.4,1.3,1.5,1.3,1.4,1.56]
#BG_THRESHOLD_ARR = [9.33,10.66,9.33,8,12,9.6]
SIGNAL_THRESHOLD_ARR = [1.45,1.3375,1.45,1.1125,1.45,1.7693]
#BG_THRESHOLD_ARR = [9.33,10.66,9.33,8,12,9.6]
#TSCALE_LIST = [0.5,1.0,2.0]
TSCALE_LIST = [0.125,0.25,0.5,1.0,2.0,4.0,8.0]

# results for TSCALE_LIST = [0.25,0.5,1.0,2.0,4.0]
#  Public part:
#    TP =   697 | FP =     1
#    FL =    20 |-----------
#    FN =   250 | TN =   972
#
#    Correct type:  569 / 697 = 81.64 %
#    Average distance bonus: 82.35 %
#
#Score: 44.234507

#SOURCE_METADATA = [ 
#    [[80,120],[170,210]], #1
#    [[50,75]], #2
#    [[330,430]], #3
#    [[0,1400]], #[[1100,1400]], #4
#    [[130,180]], #5
#    [], #6
#]

UTHR = 1500
SOURCE_METADATA = [
    [[0,UTHR]], #1
    [[0,UTHR]], #2
    [[0,UTHR]], #3
    [[0,UTHR]], #4
    [[0,UTHR]], #5
    [[0,UTHR]], #6
    ]

#SOURCE_METADATA[5].append(SOURCE_METADATA[0][0])
#SOURCE_METADATA[5].append(SOURCE_METADATA[0][1])
#SOURCE_METADATA[5].append(SOURCE_METADATA[4][0])

def _rdown(x, kev):
    return int(x/kev)
def _rup(x, kev):
    return int((x + kev-1)/kev)


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
                self.source_hist[shielding * 5 + source, :] = np.abs(denoise_signal_stub(np.array(arr)))
                self.source_hist[shielding * 5 + source, :] /= np.max(self.source_hist[shielding * 5 + source, :])

        kev_per_bin = int(MAX_ENERGY / EBINS)
        self.bin_map_arr = []
        for i in range(len(SOURCE_METADATA)):
            bin_map = dict()
            for elem in SOURCE_METADATA[i]:
                from_idx = _rdown(elem[0], kev_per_bin)
                to_idx = _rup(elem[1], kev_per_bin)
                for idx in range(from_idx, to_idx + 1):
                    bin_map[idx] = 1
            self.bin_map_arr.append(bin_map)
        min_mp_sz = min([len(mp) for mp in self.bin_map_arr])
        self.weigh_thresh_arr = []
        self.weigh_bin_map_arr = []
        for i in range(len(self.bin_map_arr)):
            self.weigh_bin_map_arr.append(min_mp_sz / len(self.bin_map_arr[i]))
            self.bin_map_arr[i] = list(self.bin_map_arr[i])
            self.weigh_thresh_arr.append(len(self.bin_map_arr[i]) / EBINS)

    def _calc_source_norm(self, dvec, source):
        slice_vec = self.bin_map_arr[source]
        return np.linalg.norm(dvec[slice_vec]) * self.weigh_bin_map_arr[source]

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
            for tscale in TSCALE_LIST:
                dat = pd.read_csv(os.path.join(tpath, "%d.csv" % id), header=None)
    
                d0=dat[0]*1e-6
                d1=np.cumsum(d0)
                d2=dat[1]
                invtscale=1/tscale
                tmax=int(d1.values[-1]*tscale)
    
                bins = EBINS
                zmat = np.zeros((tmax,bins))
    
                ebins = np.linspace(0,MAX_ENERGY,bins+1)
    
                for i in range(int(30*tscale),tmax):
                    dind = np.argwhere((d1 > i * invtscale) & (d1 < (i + 1) * invtscale)).flatten() 
                    d3 = d2[dind]
                    hist = np.histogram(d3, bins=ebins)[0]
                    hist = denoise_signal_stub(hist)
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

    def predict(self, x, ids, export=False):
        ret = np.zeros((len(ids), 2))

        export_data = []

        for i in tqdm(range(len(ids))):
            id = ids[i]

            arr = []
            tiarr = []
            sourcearr = []
            tscalearr = []

            g_arr = []
            g_tiarr = []
            g_sourcearr = []
            g_tscalearr = []
            g_sres_bgs = []

            for (j, tscale) in enumerate(TSCALE_LIST):
                dat = np.abs(x[len(TSCALE_LIST)*i + j])
                tmax = dat.shape[0]

                weigh = self.model_bg.transform(dat)

                weigh_arr_s = []
                for source in range(len(self.model_arr_bgs)):
                    weigh_arr_s.append(self.model_arr_bgs[source].transform(dat))

                for ti in range(int(30*tscale),tmax):
                    fit_bg = np.dot(weigh[ti], self.comps_bg)
                    diff_fit_bg = fit_bg - dat[ti, :]
    
                    sres = []
                    sres_bg = []
                    sres_bgs = []
                    for source in range(len(self.model_arr_bgs)):
                        fit_bgs = np.dot(weigh_arr_s[source][ti], self.model_arr_bgs[source].components_)
                        diff_fit_bgs = fit_bgs - dat[ti, :]
    
                        norm_bg = self._calc_source_norm(diff_fit_bg, source)
                        norm_bgs = self._calc_source_norm(diff_fit_bgs, source)
    
                        sres.append(norm_bg / norm_bgs)
                        sres_bg.append(norm_bg)
                        sres_bgs.append(norm_bgs)
    
                    if sres:
                        sresi = np.argmax(sres)
                        coeff = SIGNAL_COEFF[sresi]
                        thresh = SIGNAL_THRESHOLD_ARR[sresi]
                        bgthresh = BG_THRESHOLD #BG_THRESHOLD_ARR[sresi]
                        #if sres_bgs[sresi] > BG_THRESHOLD * coeff and sres[sresi] > SIGNAL_THRESHOLD * coeff:
                        if sres_bgs[sresi] > bgthresh and sres[sresi] > thresh:
                            arr.append(sres[sresi])
                            tiarr.append(ti / tscale)
                            sourcearr.append(sresi)
                            tscalearr.append(tscale)

                        g_arr.append(sres[sresi])
                        g_tiarr.append(ti / tscale)
                        g_sourcearr.append(sresi)
                        g_tscalearr.append(tscale)
                        g_sres_bgs.append(sres_bgs[sresi])
    
            if arr:
                idx = np.argmax(arr)
                ti = tiarr[idx]
                si = sourcearr[idx]
                toffs = 1/tscalearr[idx] * 0.5
                ret[i, 0] = 1 + si
                ret[i, 1] = ti + toffs

            export_data.append([g_arr, g_tiarr, g_sourcearr, g_tscalearr, g_sres_bgs])

        if export:
            tcache_path = os.path.join(self._base_path, "export.pkl")
            fd = open(tcache_path, "wb")
            pkl.dump(export_data, fd)
            fd.close()

        return ret

    def export_predict_trace(self, ids):
        tcache_path = os.path.join(self._base_path, "export.pkl")
        fd = open(tcache_path, "rb")
        list_dat = pkl.load(fd)
        fd.close()

        print("runid,snr,ti,source,toffs,sresbgs")
        for i in range(len(ids)):
            runid = ids[i]
            dat = list_dat[i]

            for si in range(6):
                fdat = (np.array(dat[2]) == si).astype(np.float64)
                idx = np.argmax(dat[0] * fdat)
                toffs = 1/dat[3][idx] * 0.5
                print("%d,%f,%f,%d,%f,%f" % (runid, dat[0][idx], dat[1][idx], dat[2][idx]+1, toffs, dat[4][idx]))
