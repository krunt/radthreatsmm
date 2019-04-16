
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
import xgboost as xgb

tdata = pd.read_csv("../trainingAnswers.csv")
export_data = pd.read_csv("../export.csv")
answer_key = pd.read_csv("../answerKey.csv")
submission = pd.read_csv("../submission.csv")

tdata = tdata.set_index(["RunID"])
answer_key = answer_key.set_index(["RunID"])
export_data = export_data.set_index(["runid"])

thresh_arr = [0.97,0.85,0.89,0.97,0.97,0.97]

res = [0,0,0,0]

for run_id in submission["RunID"]:
    edat = export_data.loc[int(run_id)]
    idx = np.argmax(list(edat["snr"]))
    erow = edat.values[idx]
    eproba = erow[-1]
    if eproba == 0 or eproba == -1:
        continue
    tdat = tdata.loc[int(run_id)]
    tsource_id = tdat["SourceID"]
    gt_has_signal = int(tsource_id != 0)
    thresh = thresh_arr[idx]
    pr_has_signal = int(eproba > thresh)
    res[gt_has_signal * 2 + pr_has_signal] += 1

print(thresh,(res[1]+res[2])/(res[0]+res[3]))

#print(res)
#print((res[1]+res[2])/(res[0]+res[3]))
        



