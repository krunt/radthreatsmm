import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

EBINS = 128
MAX_ENERGY = 2500
TSCALE_LIST = [0.25,0.5,1.0,2.0,4]
TWIN_PER_TSCALE = [1, 1, 3, 1, 1]

def dump_obj(obj, fpath):
    fd = open(fpath,"wb")
    pkl.dump(obj, fd)
    fd.close()

def main(part_id, num_parts):
    np.random.seed(13)

    submission = pd.read_csv("../solution.csv")

    plen = submission.shape[0]
    psize = int((plen + num_parts - 1) / num_parts)
    from_idx = part_id * psize
    to_idx = min((part_id + 1) * psize, plen)

    hist_list = []
    for row in tqdm(submission.values[from_idx:to_idx]):
        runid = int(row[0])
        source_id = int(row[1])
        source_time = row[2]

        g_dat = pd.read_csv(os.path.join("../testing", "%d.csv" % runid), header=None)
        g_dat = pd.concat([g_dat, g_dat])
        g_dat = g_dat.reset_index()
        
        d0=g_dat[0]*1e-6
        d1=np.cumsum(d0)
        d2=g_dat[1]
        
        ebins = np.linspace(0,MAX_ENERGY,EBINS+1)
        tmax = d1.values[-1]
        
        for (k, tscale) in enumerate(TSCALE_LIST):
            invtscale=1/tscale
            for j in range(TWIN_PER_TSCALE[k]):
                if TWIN_PER_TSCALE[k] == 0:
                    continue
                twinhalf = TWIN_PER_TSCALE[k]/2
                dind = np.argwhere((d1 > source_time + (twinhalf - j - 1) * invtscale) & (d1 < source_time + (1 + twinhalf - j - 1) * invtscale)).flatten()
                d3 = d2[dind]
                hist = np.histogram(d3, bins=ebins)[0]
                hist_list.append(hist)

    dump_obj(hist_list, "../hist_data/hist_%d.pkl" % (part_id))


if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    main(int(sys.argv[1]), int(sys.argv[2]))
