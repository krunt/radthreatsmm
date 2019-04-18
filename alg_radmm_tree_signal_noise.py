
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

import alg_radmm_base
import numpy as np
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import denoise_signal, denoise_signal_stub, cross_correlation
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from scipy.signal import find_peaks
import xgboost as xgb


NCOMP_BG = 12
NTRAIN_BG = 128
MAX_ENERGY = 2500
TWIN_PER_SCALE=[9,9,9,9,9]
TOFFS=30
TTHRESH = TOFFS + 5
EBINS = 128
TREE_BINS = 96
KEV_PER_EBIN = int(MAX_ENERGY / EBINS)
SIGNAL_THRESHOLD = 1.4
BG_THRESHOLD = 10.0
SIGNAL_COEFF = [1.0,1.1,1.0,0.9,1.1,1.5]
#SIGNAL_THRESHOLD_ARR = [1.461,1.307,1.53,1.107,1.53,1.58]
PROBA_THRESHOLD_ARR = [0,0,0,0,0,0]
TSCALE_LIST = [0.25,0.5,1.0,2.0,4.0]
TSTEP_PER_SCALE = [0.5,0.5,0.5,0.5,0.25]
UTHR = 1500
NN_BINS=EBINS
NN_PROBA = 0.5

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


class AlgRadMMTreeSignalNoise(alg_radmm_base.AlgRadMMBase):
    def __init__(self, base_path):
        alg_radmm_base.AlgRadMMBase.__init__(self, base_path)
        self.smooth_signal = False

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
                dat1 = denoise_signal(np.array(arr)) if self.smooth_signal else np.array(arr)
                self.source_hist[shielding * 5 + source, :] = np.abs(dat1)
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

    def _prepare(self, ids, is_train=True, validation=False, cache=True):
        fname = "train_nmf" if is_train else "test_nmf"
        if validation:
            fname = "validation_nmf"
        filename = "%s%s.pkl" % (fname, "_smooth" if self.smooth_signal else "")
        tcache_path = os.path.join(self._base_path, filename)
        if os.path.exists(tcache_path):
            fd = open(tcache_path, "rb")
            ret = pkl.load(fd)
            fd.close()
            return ret
        tpath = self._train_dir_path if is_train else self._test_dir_path
        ret = []
        for id in tqdm(ids, desc=fname):
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
    
                for i in range(int(TOFFS*tscale),tmax):
                    dind = np.argwhere((d1 > i * invtscale) & (d1 < (i + 1) * invtscale)).flatten() 
                    d3 = d2[dind]
                    hist = np.histogram(d3, bins=ebins)[0]
                    if self.smooth_signal:
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

    def _row2record_batch(self, model_signal, dat):
        weigh = self.model_bg.transform(dat)

        fit_bg = np.dot(weigh, self.comps_bg)
        diff_fit_bg = dat - fit_bg

        weigh_s = model_signal.transform(dat)

        fit_bgs = np.dot(weigh_s, model_signal.components_)
        diff_fit_bgs = dat - fit_bgs

        diff_fit_bg = diff_fit_bg[:, :TREE_BINS]
        diff_fit_bgs = diff_fit_bgs[:, :TREE_BINS]

        norm_bg = np.linalg.norm(diff_fit_bg, axis=1)
        norm_bgs = np.linalg.norm(diff_fit_bgs, axis=1)

        #return np.transpose(np.vstack([norm_bg, norm_bgs, norm_bg / (norm_bgs + 1e-9)]))
        return (norm_bg / (norm_bgs + 1e-9)).reshape((-1, 1))

    def _row2record(self, model_signal, rows, toffs_arr, peaks, tmax):
        ret = []
        for (i,row) in enumerate(rows):
            toffs = toffs_arr[i]

            weigh = self.model_bg.transform(row.reshape((1,-1)))
    
            fit_bg = np.dot(weigh, self.comps_bg)
            diff_fit_bg = np.abs(row - fit_bg)
    
            weigh_s = model_signal.transform(row.reshape((1,-1)))
    
            fit_bgs = np.dot(weigh_s, model_signal.components_)
            diff_fit_bgs = np.abs(row - fit_bgs)
    
            diff_fit_bg = diff_fit_bg[:, :TREE_BINS]
            diff_fit_bgs = diff_fit_bgs[:, :TREE_BINS]

            norm_bg = np.linalg.norm(diff_fit_bg)
            norm_bgs = np.linalg.norm(diff_fit_bgs)
    
            tdist = np.abs(toffs - peaks)
            idx = np.argmin(tdist)

            xrow = np.array([norm_bg / (norm_bgs + 1e-9)]) # * (tdist[idx] / tmax)])
            #xrow = np.array([norm_bg / (norm_bgs + 1e-9)])

            ret.append(xrow)

        return np.hstack(ret)


    def _get_train_tree_data(self, x, scaleidx, ids):
        tscale1 = TSCALE_LIST[scaleidx]
        sig_list = []
        bg_list = []
        for (i,runid) in enumerate(ids):
            source_id = self._train_metadata.loc[runid]["SourceID"]
            source_time = self._train_metadata.loc[runid]["SourceTime"]
            if source_id != 0:
                if source_time < TTHRESH:
                    continue
                if x[i * len(TSCALE_LIST) + 2].shape[0] < (source_time + 5 + 5):
                    continue
                sig_list.append((i,runid))
            else:
                if x[i * len(TSCALE_LIST) + 2].shape[0] < (TTHRESH + 5 + 5):
                    continue
                bg_list.append((i,runid))
        np.random.shuffle(sig_list)
        np.random.shuffle(bg_list)

        min_sz = min(len(sig_list), len(bg_list))
        sig_list = sig_list[:min_sz]
        bg_list = bg_list[:min_sz]

        tpath = self._train_dir_path

        xlist = []
        ylist = []
        for elem in ((1, sig_list), (0, bg_list)):
            for (idx,runid) in tqdm(elem[1], desc="train(%d)" % (scaleidx)):
                source_time = 0
                if elem[0]:
                    source_time = self._train_metadata.loc[runid]["SourceTime"]
                else:
                    source_time = TTHRESH
    
                for j in [scaleidx]: #range(len(TSCALE_LIST)):
                    tscale = TSCALE_LIST[j]
                    invtscale=1/tscale

                    g_dat = pd.read_csv(os.path.join(tpath, "%d.csv" % runid), header=None)
                    d0=g_dat[0]*1e-6
                    d1=np.cumsum(d0)
                    d2=g_dat[1]
                    bins = EBINS
                    ebins = np.linspace(0,MAX_ENERGY,bins+1)
                    tmax=d1.values[-1]

                    tstep = TSTEP_PER_SCALE[j]
                    tcurr = source_time

                    timeHist = np.histogram(d1, bins=1024)[0]
                    timeHist = denoise_signal(timeHist)
                    peaks, _ = find_peaks(timeHist, prominence=(5))
                    peaksS = peaks / 1024 * tmax

                    hist_list = []
                    tiarr = []
                    tscalearr = []

                    twin = TWIN_PER_SCALE[scaleidx]
                    twinoffs = int(twin/2)

                    inp = []
                    toffs_arr = []
                    for tinc in range(twin):
                        ttoffs_s = (tinc - twinoffs) * tstep
                        assert(tcurr + ttoffs_s > TOFFS)
                        assert(tcurr + ttoffs_s + invtscale < tmax)
                        dind = np.argwhere((d1 > tcurr + ttoffs_s) & (d1 < tcurr + ttoffs_s + invtscale)).flatten() 
                        d3 = d2[dind]
                        hist = np.histogram(d3, bins=ebins)[0]
    
                        inp.append(hist)
                        toffs_arr.append(tcurr + ttoffs_s)

                    xrow = self._row2record(self.model_bgs, inp, toffs_arr, peaksS, tmax)

                    xlist.append(xrow)
                    ylist.append(elem[0])

        xlist = np.vstack(xlist)
        ylist = np.vstack(ylist)

        return (xlist, ylist)

    def _get_train_tree_data_type(self, x, scaleidx, ids):
        tscale1 = TSCALE_LIST[scaleidx]
        source_time_arr = []
        sig_list = []
        for (i,runid) in enumerate(ids):
            source_id = self._train_metadata.loc[runid]["SourceID"]
            source_time = self._train_metadata.loc[runid]["SourceTime"]
            if source_id in [1,5,6]:
                sig_list.append((i,runid))
                source_time_arr.append(source_time)

        tpath = self._train_dir_path

        xlist = []
        ylist = []
        for (idx,runid) in tqdm(sig_list, desc="train(%d)" % (scaleidx)):
            source_time = self._train_metadata.loc[runid]["SourceTime"]
            source_id = self._train_metadata.loc[runid]["SourceID"]

            for j in [scaleidx]:
                tscale = TSCALE_LIST[j]
                invtscale=1/tscale

                g_dat = pd.read_csv(os.path.join(tpath, "%d.csv" % runid), header=None)
                d0=g_dat[0]*1e-6
                d1=np.cumsum(d0)
                d2=g_dat[1]
                bins = EBINS
                ebins = np.linspace(0,MAX_ENERGY,bins+1)
                tmax=d1.values[-1]

                tcurr = source_time
                dind = np.argwhere((d1 > tcurr - invtscale / 2) & (d1 < tcurr + invtscale / 2)).flatten() 
                d3 = d2[dind]
                hist = np.histogram(d3, bins=ebins)[0]

                xrow = self.model_bgs5.transform(hist.reshape((1,-1)))

                xlist.append(xrow)
                ylist.append(source_id)

        xlist = np.vstack(xlist)
        ylist = np.vstack(ylist)

        return (xlist, ylist)

    def _train_trees(self, x, ids):
         for scaleidx in range(0,len(TSCALE_LIST)):
             xdata, ydata = self._get_train_tree_data(x, scaleidx, ids)
             xtrain_data, xtest_data, ytrain_data, ytest_data = train_test_split(xdata, ydata, 
                 test_size=0.2, random_state=13)
             xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
             xgb_model.fit(xtrain_data, ytrain_data)
             ytest_predict_data = xgb_model.predict(xtest_data)

             conf_mat = confusion_matrix(ytest_data, ytest_predict_data).ravel()

             print("(%d) tn=%f, fp=%f, fn=%f, tp=%f" % (scaleidx, conf_mat[0], conf_mat[1], conf_mat[2], conf_mat[3]))
             print("(%d) (fp+fn)/(tn+tp)=%f" % (scaleidx, (conf_mat[1] + conf_mat[2]) / (conf_mat[0] + conf_mat[3])))
             print("(%d) mse=%f" % (scaleidx, mean_squared_error(ytest_predict_data, ytest_data)))

             tcache_path = os.path.join(self._base_path, "trees", "tree_%d.pkl" % (scaleidx))
             fd = open(tcache_path, "wb")
             pkl.dump(xgb_model, fd)
             fd.close()

    def _train_trees_type(self, x, ids):
         for scaleidx in range(0,len(TSCALE_LIST)):
             xdata, ydata = self._get_train_tree_data_type(x, scaleidx, ids)
             xtrain_data, xtest_data, ytrain_data, ytest_data = train_test_split(xdata, ydata, 
                 test_size=0.2, random_state=13)
             xgb_model = xgb.XGBClassifier(random_state=42)
             xgb_model.fit(xtrain_data, ytrain_data)
             ytest_predict_data = xgb_model.predict(xtest_data)

             conf_mat = confusion_matrix(ytest_data, ytest_predict_data).ravel()

             print("(%d) tn=%f, fp=%f, fn=%f, tp=%f" % (scaleidx, conf_mat[0], conf_mat[1], conf_mat[2], conf_mat[3]))
             print("(%d) (fp+fn)/(tn+tp)=%f" % (scaleidx, (conf_mat[1] + conf_mat[2]) / (conf_mat[0] + conf_mat[3])))
             print("(%d) mse=%f" % (scaleidx, mean_squared_error(ytest_predict_data, ytest_data)))

             tcache_path = os.path.join(self._base_path, "trees_type", "tree_%d.pkl" % (scaleidx))
             fd = open(tcache_path, "wb")
             pkl.dump(xgb_model, fd)
             fd.close()

    # source starts 0
    def _get_model_tree(self, scaleidx):
        return self.model_trees[scaleidx]

    def _load_trees(self):
        self.model_trees = []
        for scaleidx in range(0,len(TSCALE_LIST)):
            tcache_path = os.path.join(self._base_path, "trees1", "tree_%d.pkl" % (scaleidx))
            fd = open(tcache_path, "rb")
            xgb_model = pkl.load(fd)
            fd.close()

            self.model_trees.append(xgb_model)

    def train(self, x, y, ids):
        runid_list = []
        for (i,runid) in enumerate(ids):
            if self._train_metadata.loc[runid]["SourceID"] == 0:
                runid_list.append((i,runid))
        np.random.shuffle(runid_list)
        runid_list = runid_list[:NTRAIN_BG]

        xlist = []
        for (idx,runid) in runid_list:
            for j in [2]: #range(len(TSCALE_LIST)):
                xlist.append(np.abs(x[idx * len(TSCALE_LIST) + j][TOFFS:]))

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

        #self._train_trees(x, ids)
        #self._load_trees()
        self._train_trees_type(x, ids)

    def _calc_source_norm(self, dvec, source):
        return np.linalg.norm(dvec[:TREE_BINS])

    def predict(self, x, ids, export=False):
        ret = np.zeros((len(ids), 2))

        nn_stat = []
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

            dat_batch = []
            for (j, tscale) in enumerate(TSCALE_LIST):
                dat = np.abs(x[i * len(TSCALE_LIST) + j])
                tmax = dat.shape[0]
                toffs = int(30*tscale)

                for ti in range(toffs,tmax):
                    twin = TWIN_PER_SCALE[j]
                    twinoffs = int(twin/2)
                    if ti - twinoffs < toffs or ti + twinoffs >= tmax:
                        continue

                    for tinc in range(twin):
                        stime = ti + tinc - twinoffs
                        dat_batch.append(dat[stime])

            dat_batch = np.vstack(dat_batch)

            for source in range(len(self.model_arr_bgs)):
                dat_batch_p = self._row2record_batch(self.model_arr_bgs[source], dat_batch)

                iidx = 0
                for (j, tscale) in enumerate(TSCALE_LIST):
                    dat = x[i * len(TSCALE_LIST) + j]
                    tmax = dat.shape[0]
                    toffs = int(30*tscale)
    
                    model_data_batch = []
                    for ti in range(toffs,tmax):
                        twin = TWIN_PER_SCALE[j]
                        twinoffs = int(twin/2)
                        if ti - twinoffs < toffs or ti + twinoffs >= tmax:
                            continue
    
                        inp = []
                        for tinc in range(twin):
                            stime = ti + tinc - twinoffs
                            inp.append(dat_batch_p[iidx])
                            iidx += 1

                        model_data_batch.append(np.hstack(inp))

                    model_data_batch = np.vstack(model_data_batch)
                    proba_arr = self._get_model_tree(source, j).predict_proba(model_data_batch)

                    proba_thresh = PROBA_THRESHOLD_ARR[source]

                    iidx1 = 0
                    for ti in range(toffs,tmax):
                        twin = TWIN_PER_SCALE[j]
                        twinoffs = int(twin/2)
                        if ti - twinoffs < toffs or ti + twinoffs >= tmax:
                            continue

                        proba = proba_arr[iidx1][1]
                        iidx1 += 1

                        if proba > proba_thresh:
                            arr.append(proba)
                            tiarr.append((ti + toffs) / tscale)
                            sourcearr.append(source)
                            tscalearr.append(tscale)

                        g_arr.append(proba)
                        g_tiarr.append((ti + toffs) / tscale)
                        g_sourcearr.append(source)
                        g_tscalearr.append(tscale)

            if arr:
                idx = np.argmax(arr)
                ti = tiarr[idx]
                si = sourcearr[idx]
                toffs = 1/tscalearr[idx] * 0.5
                ret[i, 0] = 1 + si
                ret[i, 1] = ti + toffs
                print((arr[idx], si, tscalearr[idx]))

            export_data.append([g_arr, g_tiarr, g_sourcearr, g_tscalearr])

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

        print("runid,snr,ti,source,toffs,tscale")
        for i in range(len(ids)):
            runid = ids[i]
            dat = list_dat[i]

            for si in range(6):
                fdat = (np.array(dat[2]) == si).astype(np.float64)
                idx = np.argmax(dat[0] * fdat)
                toffs = 1/dat[3][idx] * 0.5
                print("%d,%f,%f,%d,%f,%f" % (runid, dat[0][idx], dat[1][idx], dat[2][idx]+1, toffs, dat[3][idx]))
