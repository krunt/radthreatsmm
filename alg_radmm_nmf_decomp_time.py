
import alg_radmm_base
import numpy as np
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import denoise_signal, denoise_signal_stub
from multiprocessing import Process, Queue, Pool
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

SCORE_BINS=76
NCOMP_BG = 12
NTRAIN_BG = 128
MAX_ENERGY = 2500
TREE_BINS=96
EBINS = 128
KEV_PER_EBIN = int(MAX_ENERGY / EBINS)
SIGNAL_THRESHOLD = 1.4
BG_THRESHOLD = 10.0
TWIN_PER_SCALE=[9,9,9,9,9]
SIGNAL_COEFF = [1.0,1.1,1.0,0.9,1.1,1.5]
#PROBA_THRESHOLD_ARR = [0.95,0.95,0.95,0.95,0.95,0.95]
PROBA_THRESHOLD_ARR = [0.97,0.85,0.89,0.97,0.97,0.97]
#SIGNAL_THRESHOLD_ARR = [1.4,1.3,1.5,1.3,1.5,1.6]
#SIGNAL_THRESHOLD_ARR = [1.4,1.3,1.5,1.3,1.4,1.56]
#BG_THRESHOLD_ARR = [9.33,10.66,9.33,8,12,9.6]
#SIGNAL_THRESHOLD_ARR = [1.45,1.3375,1.45,1.1125,1.45,1.7693]
#SIGNAL_THRESHOLD_ARR = [ 1.66667, 1.88889, 2, 1.22222, 2, 2.27273 ]
SIGNAL_THRESHOLD_ARR = [1.44444, 1.55556, 1.77778, 1.11111, 1.88889, 1.72727]
#[1.72,1.9,2.09,1.18,2.27,2.38]
#SIGNAL_THRESHOLD_ARR = [ 1.46667, 1.46667, 1.46667, 1.2, 1.46667, 1.84, 2, 1.73333, 1.73333, 1.2, 2, 2.24 ]
#SIGNAL_THRESHOLD_ARR = [1.75,1.5,1.625,1.125,1.875,1.69231]
#BG_THRESHOLD_ARR = [9.33,10.66,9.33,8,12,9.6]
#TSCALE_LIST = [0.5,1.0,2.0]
TSCALE_LIST = [0.25,0.5,1.0,2.0,4.0]
TSTEP_PER_SCALE = [0.5,0.5,0.5,0.5,0.25]
#TSCALE_COEFF_LIST = [0.90,0.93,1.0,1.06,1.10]
TSCALE_COEFF_LIST = [0.95,0.97,1.0,1.03,1.05]
#TSCALE_COEFF_LIST = [1.0,1.0,1.0,1.0,1.0]

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


class AlgRadMMNmfDecompTime(alg_radmm_base.AlgRadMMBase):
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
        return np.linalg.norm(dvec[:SCORE_BINS])

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

    def get_test_ids(self):
        tlist_path = os.path.join(self._base_path, "solution2.csv")
        dat = pd.read_csv(tlist_path)
        return dat["RunID"]

    def _pkl_dump_to_fname(self, obj, fpath):
        fd = open(fpath, "wb")
        pkl.dump(obj, fd)
        fd.close()

    def _pkl_load_from_fname(self, fpath):
        fd = open(fpath, "rb")
        ret = pkl.load(fd)
        fd.close()
        return ret

    def train(self, x, y, ids):
        #self._train_nmf(x, ids)
        self._load_nmf()
        self._load_trees()
        self._load_trees_type()

    # source starts 0
    def _get_model_tree(self, scaleidx):
        return self.model_trees[scaleidx]

    def _get_model_tree_type(self, scaleidx):
        return self.model_trees_type[scaleidx]

    def _train_nmf(self, x, ids):
        runid_list = []
        for (i,runid) in enumerate(ids):
            if self._train_metadata.loc[runid]["SourceID"] == 0:
                runid_list.append((i,runid))
        np.random.shuffle(runid_list)
        runid_list = runid_list[:NTRAIN_BG]

        xlist = []
        for (idx,runid) in runid_list:
            for j in [2]:
                tscale = TSCALE_LIST[j]
                toffs = int(30*tscale)
                xlist.append(np.abs(x[idx * len(TSCALE_LIST) + j][toffs:]))

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

        self.model_bgs5 = NMF(ncomp_bg+5, init='random', random_state=0)
        self.model_bgs5.fit(xlist)
        for i in range(ncomp_bg):
            self.model_bgs5.components_[i] = self.model_bg.components_[i]
        for i in range(5):
            self.model_bgs5.components_[-5 + i] = self.source_hist[i]
        self.comps_bgs5 = self.model_bgs5.components_

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

        self._pkl_dump_to_fname(self.model_bg, os.path.join(self._base_path, "models_nmf", "model_bg.pkl"))
        self._pkl_dump_to_fname(self.model_bgs, os.path.join(self._base_path, "models_nmf", "model_bgs.pkl"))
        self._pkl_dump_to_fname(self.model_bgs5, os.path.join(self._base_path, "models_nmf", "model_bgs5.pkl"))
        for i in range(6):
            self._pkl_dump_to_fname(self.model_arr_bgs[i], 
                    os.path.join(self._base_path, "models_nmf", "model_arr_bgs_%d.pkl" % (i)))

    def _load_nmf(self):
        self.model_bg = self._pkl_load_from_fname(os.path.join(self._base_path, "models_nmf", "model_bg.pkl"))
        self.model_bgs = self._pkl_load_from_fname(os.path.join(self._base_path, "models_nmf", "model_bgs.pkl"))
        self.model_bgs5 = self._pkl_load_from_fname(os.path.join(self._base_path, "models_nmf", "model_bgs5.pkl"))
        self.model_arr_bgs = []
        for i in range(6):
            self.model_arr_bgs.append(self._pkl_load_from_fname(os.path.join(
                            self._base_path, "models_nmf", "model_arr_bgs_%d.pkl" % (i))))
        self.comps_bg = self.model_bg.components_
        self.comps_bgs = self.model_bgs.components_
        self.comps_bgs5 = self.model_bgs5.components_


    def _load_trees(self):
        self.model_trees = []
        for scaleidx in range(0,len(TSCALE_LIST)):
            tcache_path = os.path.join(self._base_path, "trees", "tree_%d.pkl" % (scaleidx))
            fd = open(tcache_path, "rb")
            xgb_model = pkl.load(fd)
            fd.close()

            self.model_trees.append(xgb_model)

    def _load_trees_type(self):
        self.model_trees_type = []
        for scaleidx in range(0,len(TSCALE_LIST)):
            tcache_path = os.path.join(self._base_path, "trees_type", "tree_%d.pkl" % (scaleidx))
            fd = open(tcache_path, "rb")
            xgb_model = pkl.load(fd)
            fd.close()

            self.model_trees_type.append(xgb_model)

    def _row2record(self, model_signal, rows):
        ret = []
        for row in rows:
            weigh = self.model_bg.transform(row.reshape((1,-1)))
    
            fit_bg = np.dot(weigh, self.comps_bg)
            diff_fit_bg = np.abs(row - fit_bg)
    
            weigh_s = model_signal.transform(row.reshape((1,-1)))
    
            fit_bgs = np.dot(weigh_s, model_signal.components_)
            diff_fit_bgs = np.abs(row - fit_bgs)
    
            diff_fit_bg = diff_fit_bg[:TREE_BINS]
            diff_fit_bgs = diff_fit_bgs[:TREE_BINS]
    
            norm_bg = np.linalg.norm(diff_fit_bg)
            norm_bgs = np.linalg.norm(diff_fit_bgs)
    
            xrow = np.array([norm_bg, norm_bgs, norm_bg / (norm_bgs + 1e-9)])
            #xrow = np.array([norm_bg / (norm_bgs + 1e-9)])
            ret.append(xrow)

        return np.hstack(ret)

    def predict(self, x, ids, export=False, validation=False):
        ret = np.zeros((len(ids), 2))

        export_data = []

        tpath = self._train_dir_path if validation else self._test_dir_path

        for i in tqdm(range(len(ids))):
            id = ids[i]

            g_dat = pd.read_csv(os.path.join(tpath, "%d.csv" % id), header=None)
            d0=g_dat[0]*1e-6
            d1=np.cumsum(d0)
            d2=g_dat[1]
            bins = EBINS
            ebins = np.linspace(0,MAX_ENERGY,bins+1)

            hist_list = []
            tiarr = []
            tscalearr = []
            tscaleidxarr = []
            tscalecoeffarr = []

            for (j, tscale) in enumerate(TSCALE_LIST):
                invtscale=1/tscale
                tmax=d1.values[-1]

                tstep = TSTEP_PER_SCALE[j]
                tcurr = 30

                while tcurr + invtscale < tmax:
                    dind = np.argwhere((d1 > tcurr) & (d1 < tcurr + invtscale)).flatten() 

                    d3 = d2[dind]
                    hist = np.histogram(d3, bins=ebins)[0]

                    hist_list.append(hist)
                    tiarr.append(tcurr)
                    tscalearr.append(tscale)
                    tscaleidxarr.append(j)
                    tscalecoeffarr.append(TSCALE_COEFF_LIST[j])

                    tcurr += tstep

            hist_list = np.vstack(hist_list)

            weigh = self.model_bg.transform(hist_list)

            weigh_arr_s = []
            for source in range(len(self.model_arr_bgs)):
                weigh_arr_s.append(self.model_arr_bgs[source].transform(hist_list))

            weigh5 = self.model_bgs5.transform(hist_list)

            fit_bg = np.dot(weigh, self.comps_bg)
            diff_fit_bg = fit_bg - hist_list
            diff_fit_bg = diff_fit_bg[:, :SCORE_BINS]

            norm_bg = np.linalg.norm(diff_fit_bg, axis=1)

            norm_bgs_arr = []
            score_bg_arr = []
            bonus_arr = []

            for source in range(len(self.model_arr_bgs)):
                fit_bgs = np.dot(weigh_arr_s[source], self.model_arr_bgs[source].components_)
                diff_fit_bgs = fit_bgs - hist_list
                norm_bgs = np.linalg.norm(diff_fit_bgs[:, :SCORE_BINS], axis=1)

                score_bg = norm_bg / (norm_bgs + 1e-9)

                norm_bgs_arr.append(norm_bgs)

                fdat = np.array((norm_bgs > BG_THRESHOLD)).astype(np.float64)
                score_bg_arr.append(score_bg * fdat)
                bonus_arr.append(tscalecoeffarr)

            norm_bgs_arr = np.hstack(norm_bgs_arr)
            score_bg_arr = np.hstack(score_bg_arr)
            bonus_arr = np.hstack(bonus_arr)

            export_proba = -1
            if score_bg_arr.shape[0] > 0:
                idx = np.argmax(score_bg_arr * bonus_arr)
                if score_bg_arr[idx] > 0:
                    idx0 = idx % norm_bg.shape[0]
                    idx1 = int(idx / norm_bg.shape[0])
                    ti = tiarr[idx0]
                    source = idx1
                    toffs = 1/tscalearr[idx0] * 0.5
                    thresh = SIGNAL_THRESHOLD_ARR[source] * 0.95
                    tscaleidx = tscaleidxarr[idx0]
                    model_tree_type = self._get_model_tree_type(tscaleidx)

                    if score_bg_arr[idx] > thresh:
                        source_p = int(model_tree_type.predict(weigh5[idx0].reshape(1, -1))[0]) if source in [1,5,6] else (1 + source)
                        ret[i, 0] = source_p # source
                        ret[i, 1] = ti + toffs
                        export_proba = 0

                        # trying to improve accuracy
                        fdat = score_bg_arr
                        fdat_mask = np.zeros(score_bg_arr.shape[0])
                        #fdat = fdat * fdat_mask

                        tscaleidx += 1

                        while tscaleidx < len(TSCALE_LIST):
                            fdat_mask[source * norm_bg.shape[0]:(source + 1) * norm_bg.shape[0]] = np.array(tscaleidxarr) >= tscaleidx

                            fdat1 = fdat * fdat_mask
                            idx = np.argmax(fdat1)
                            if fdat1[idx] < 1e-5:
                                break;
                            idx0 = idx % norm_bg.shape[0]
                            idx1 = int(idx / norm_bg.shape[0])
                            ti = tiarr[idx0]
                            toffs = 1/tscalearr[idx0] * 0.5

                            thresh = SIGNAL_THRESHOLD_ARR[source] * 0.95
                            if score_bg_arr[idx] > thresh:
                                model_tree_type = self._get_model_tree_type(tscaleidx)
                                source_p = int(model_tree_type.predict(weigh5[idx0].reshape(1, -1))[0]) if source in [1,5,6] else (1 + source)
                                ret[i, 0] = source_p
                                ret[i, 1] = ti + toffs

                            tscaleidx += 1

                    else:
                        twin = TWIN_PER_SCALE[tscaleidx]
                        twinoffs = int(twin/2)

                        inp = []
                        for tinc in range(twin):
                            ttoffs = tinc - twinoffs
                            if idx + ttoffs < 0 or idx0 + ttoffs < 0 or idx0 + ttoffs >= norm_bg.shape[0] or idx + ttoffs >= norm_bgs_arr.shape[0] or tscaleidxarr[idx0 + ttoffs] != tscaleidx:
                                break
                            inp.append(hist_list[idx0 + ttoffs])

                        #print(idx,idx0,norm_bg.shape[0],tscaleidx,len(inp),twin)

                        if len(inp) == twin:
                            xrow = self._row2record(self.model_bgs, inp)
                            model = self._get_model_tree(tscaleidx)
                            proba = model.predict_proba(xrow.reshape(1, -1))[0][1]
                            proba_thresh = PROBA_THRESHOLD_ARR[tscaleidx]
                            #print(id,1+source,ti+toffs,tscaleidx,proba,proba_thresh)
                            if proba > proba_thresh:
                                ret[i, 0] = 1 + source
                                ret[i, 1] = ti + toffs
                            export_proba = proba

            if export:
                export_data.append([norm_bg, tiarr, tscalearr, norm_bgs_arr, score_bg_arr, export_proba])

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

#        print("runid,snr,tscale")
#        for i in range(len(ids[:20])):
#            runid = ids[i]
#            dat = list_dat[i]
#            base_size = dat[0].shape[0]
#
#            for si in range(6):
#                score_arr = dat[0] / dat[3][base_size * si:(base_size * (si + 1))]
#                for j in range(base_size):
#                    print("%d,%f,%f" % (runid, score_arr[j], dat[2][j]))

        print("runid,snr,ti,source,toffs,sresbg,sresbgs,eproba")
        for i in range(len(ids)):
            runid = ids[i]
            dat = list_dat[i]
            base_size = dat[0].shape[0]

            for si in range(6):
                score_arr = dat[0] / dat[3][base_size * si:(base_size * (si + 1))]
                idx = np.argmax(score_arr)
    
                score = score_arr[idx]
                ti = dat[1][idx]
                tscale = dat[2][idx]
                toffs = 1/tscale * 0.5
                sresbg = dat[0][idx]
                sresbgs = dat[3][base_size*si + idx]
                eproba = dat[5]
                print("%d,%f,%f,%d,%f,%f,%f,%f" % (runid, score, ti, si+1, toffs, sresbg, sresbgs, eproba))
