
import alg_radmm_base
import numpy as np
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import denoise_signal, denoise_signal_stub
from keras import backend as k
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Activation, Dense, Flatten
from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D
from sklearn.decomposition import NMF

NCOMP_BG = 12
NTRAIN_BG = 32 #128
MAX_ENERGY = 2500
TOFFS=30
EBINS = 128
KEV_PER_EBIN = int(MAX_ENERGY / EBINS)
SIGNAL_THRESHOLD = 1.4
BG_THRESHOLD = 10.0
SIGNAL_COEFF = [1.0,1.1,1.0,0.9,1.1,1.5]
SIGNAL_THRESHOLD_ARR = [1.461,1.307,1.53,1.107,1.53,1.58]
TSCALE_LIST = [0.25,0.5,1.0,2.0,4.0]
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


class AlgRadMMNeuralNetSignalNoise(alg_radmm_base.AlgRadMMBase):
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

    def _get_neural_network_graph(self):
        model = Sequential()
        model.add(Flatten(input_shape=(NN_BINS,1)))
        model.add(Dropout(0.5, seed=23087, name='drop1'))
        model.add(Dense(1024, kernel_initializer='random_uniform', activation='relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5, seed=23087, name='drop9'))
        model.add(Dense(1024, kernel_initializer='random_uniform', activation='relu'))
        model.add(BatchNormalization())

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def _normalize_data(self, vec):
        #return vec / np.max(vec)
        return (vec - np.mean(vec)) / np.std(vec)

    def _get_train_nn_data(self, x, ids):
        sig_list = []
        bg_list = []
        for (i,runid) in enumerate(ids):
            if self._train_metadata.loc[runid]["SourceID"] != 0:
                sig_list.append((i,runid))
            else:
                bg_list.append((i,runid))
        np.random.shuffle(sig_list)
        np.random.shuffle(bg_list)

        xlist = []
        ylist = []
        for (idx,runid) in tqdm(sig_list, "nn_sig_list"):
            source_time = self._train_metadata.loc[runid]["SourceTime"]

            for j in range(len(TSCALE_LIST)):
                tscale = TSCALE_LIST[j]
                stime = int(source_time * tscale)
                dat = np.abs(x[idx * len(TSCALE_LIST) + j])

                weigh = self.model_bg.transform(dat[stime].reshape((1,-1)))
                fit_bg = np.dot(weigh, self.comps_bg)
                dat = np.abs(dat[stime] - fit_bg)

                #xlist.append(self._normalize_data(dat)[:NN_BINS])
                xlist.append(dat[:NN_BINS])
                ylist.append(1)

        iterate_until = 2 * len(xlist)

        for (idx,runid) in tqdm(bg_list[:len(sig_list)], "nn_bg_list"):
            for j in range(len(TSCALE_LIST)):
                if len(xlist) > iterate_until:
                    break
                tscale = TSCALE_LIST[j]
                dat = np.abs(x[idx * len(TSCALE_LIST) + j])
                weigh = self.model_bg.transform(dat)
                for stime in range(int(TOFFS*tscale), dat.shape[0]):
                    fit_bg = np.dot(weigh[stime], self.comps_bg)
                    diff_fit_bg = np.abs(fit_bg - dat[stime])
                    #xlist.append(self._normalize_data(diff_fit_bg)[:NN_BINS])
                    xlist.append(diff_fit_bg[:NN_BINS])
                    ylist.append(0)

        xlist = np.vstack(xlist)
        ylist = np.vstack(ylist)

        return (xlist, ylist)


    def train(self, x, y, ids):
        runid_list = []
        for (i,runid) in enumerate(ids):
            if self._train_metadata.loc[runid]["SourceID"] == 0:
                runid_list.append((i,runid))
        np.random.shuffle(runid_list)
        runid_list = runid_list[:NTRAIN_BG]

        xlist = []
        for (idx,runid) in runid_list:
            for j in range(len(TSCALE_LIST)):
                xlist.append(np.abs(x[idx * len(TSCALE_LIST) + j]))

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

        # neural network training
#        xdata, ydata = self._get_train_nn_data(x, ids)
#        xtrain_data, xtest_data, ytrain_data, ytest_data = train_test_split(xdata, ydata, 
#                test_size=0.2, random_state=13)
#        xtrain_data = xtrain_data.reshape((-1,NN_BINS,1))
#        xtest_data = xtest_data.reshape((-1,NN_BINS,1))
#        model = self._get_neural_network_graph()
#        best_model_file = os.path.join(self._base_path, "weight_{epoch:02d}-val_acc{val_acc:.3f}.h5")
#        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
#        best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)
#        model.fit(xtrain_data, ytrain_data, validation_data=[xtest_data, ytest_data],
#                epochs=50, batch_size=32, shuffle=True, callbacks=[best_model,early_stopping], verbose=1)

    def _calc_source_norm(self, dvec, source):
        slice_vec = self.bin_map_arr[source]
        return np.linalg.norm(dvec[slice_vec]) * self.weigh_bin_map_arr[source]

    def predict(self, x, ids, export=False):
        model = load_model("/mnt/ssd/radiologicalthreatsmm/weight_01-val_acc0.820.h5")

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
            g_sres_bgs = []
            g_smooth_arr = []
            g_diff_fit_bg = []

            for is_smooth in range(1):
                for (j, tscale) in enumerate(TSCALE_LIST):
                    dat = np.abs(x[len(TSCALE_LIST)*i + j])
                    tmax = dat.shape[0]

                    if is_smooth:
                        dat = np.abs(denoise_signal(dat))
    
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
                            thresh = SIGNAL_THRESHOLD_ARR[sresi + is_smooth * 6]
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
                            g_smooth_arr.append(is_smooth)
                            g_diff_fit_bg.append(diff_fit_bg)
    
            if arr:
                idx = np.argmax(arr)
                ti = tiarr[idx]
                si = sourcearr[idx]
                toffs = 1/tscalearr[idx] * 0.5
                ret[i, 0] = 1 + si
                ret[i, 1] = ti + toffs
                nn_stat.append((-1, -1))
            else:
                idx = np.argmax(g_arr)
                ti = g_tiarr[idx]
                si = g_sourcearr[idx]
                toffs = 1/g_tscalearr[idx] * 0.5
                diff_fit_bg = np.abs(g_diff_fit_bg[idx][:NN_BINS])
                proba = model.predict(diff_fit_bg.reshape(-1,NN_BINS,1))[0][0]
                if proba > NN_PROBA:
                    ret[i, 0] = 1 + si
                    ret[i, 1] = ti + toffs
                nn_stat.append((id, proba))

            export_data.append([g_arr, g_tiarr, g_sourcearr, g_tscalearr, g_sres_bgs, g_smooth_arr])

        if export:
            tcache_path = os.path.join(self._base_path, "export.pkl")
            fd = open(tcache_path, "wb")
            pkl.dump(export_data, fd)
            fd.close()

        if nn_stat:
            tnnstat_path = os.path.join(self._base_path, "nn_stat.pkl")
            fd = open(tnnstat_path, "wb")
            pkl.dump(nn_stat, fd)
            fd.close()

        return ret

