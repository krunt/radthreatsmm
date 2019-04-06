
import alg_base
import os
import numpy as np
import pandas as pd
import math

MAX_SCORE = 100
MIN_SCORE = 0
S_FN = -2 
S_FP = -2
S_TN = 6
S_distance = 1
S_type = 1

NUMBER_OF_FIELDS = 6
NUMBER_OF_ANS_FIELDS = 3
RUN_ID = 0
SRC_ID = 1
SRC_TIME = 2
PUBLIC = 3
VELOCITY = 4
STANDOFF = 5

def _is_integer(text):
    try:
        int(text)
        return True
    except ValueError:
        return False

def _is_float(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


class AlgRadMMBase(alg_base.AlgBase):
    def __init__(self, base_path):
        self._base_path = base_path
        self._train_dir_path = os.path.join(base_path, "training")
        self._test_dir_path = os.path.join(base_path, "testing")
        tlist_path = os.path.join(self._base_path, "trainingAnswers.csv")
        self._train_metadata = pd.read_csv(tlist_path)
        self._train_metadata.set_index("RunID", inplace=True)
        self._gtruth_key_path = os.path.join(base_path, "answerKey.csv")
        self._gtruth = pd.read_csv(self._gtruth_key_path)
        self._gtruth.set_index("RunID", inplace=True)
        self._source_data = pd.read_csv(os.path.join(base_path, "SourceInfov3/SourceData.csv"))

    def get_train_ids(self):
        return self._train_metadata.index
    def get_test_ids(self):
        tlist_path = os.path.join(self._base_path, "submittedAnswers.csv")
        dat = pd.read_csv(tlist_path)
        return dat["RunID"]
    def get_train_y(self, ids):
        ret = np.zeros((len(ids), 2))
        ret[:, 0] = self._train_metadata.loc[ids]["SourceID"]
        ret[:, 1] = self._train_metadata.loc[ids]["SourceTime"]
        return ret
    def score(self, pred_y, ids, verbose=False):
        public_with = public_without = private_with = private_without = 0

        truth_lines = [];
        try:
            file = open(self._gtruth_key_path, "r")
            truth_lines = file.readlines()
            file.close() 
        except IOError:
            print("Can't open truth file '" + self._gtruth_key_path + "'.") 
            return -1
    
        truth = {}
        for line in truth_lines:
            parts = line.strip().split(',')
            if (len(parts) == NUMBER_OF_FIELDS or len(parts) == NUMBER_OF_FIELDS - 1) and _is_integer(parts[RUN_ID]):
                if len(parts) == NUMBER_OF_FIELDS - 1:
                    parts.append("1")
                run_id = int(parts[RUN_ID])
                if int(parts[PUBLIC]) == 2:
                    parts[PUBLIC] = "1"
                if int(parts[PUBLIC]) == 3:
                    parts[PUBLIC] = "0"
                truth[run_id] = {SRC_ID : int(parts[SRC_ID]), SRC_TIME : float(parts[SRC_TIME]), PUBLIC : int(parts[PUBLIC]), VELOCITY : float(parts[VELOCITY]), STANDOFF : float(parts[STANDOFF]), "found" : False}
                if truth[run_id][PUBLIC] == 1:
                    if truth[run_id][SRC_ID] == 0:
                        public_without += 1
                    else:
                        public_with += 1
                else:
                    if truth[run_id][SRC_ID] == 0:
                        private_without += 1
                    else:
                        private_with += 1

        p_public = (MAX_SCORE - MIN_SCORE) / (public_with * (S_distance + S_type) + public_without * S_TN - public_with * min(S_FP, S_FN) - public_without * S_FP) 
        public_score = MAX_SCORE - (public_with * (S_distance + S_type) * p_public + public_without * S_TN * p_public)
        if private_with + private_without > 0:
            p_private = (MAX_SCORE - MIN_SCORE) / (private_with * (S_distance + S_type) + private_without * S_TN - private_with * min(S_FP, S_FN) - private_without * S_FP) 
            private_score = MAX_SCORE - (private_with * (S_distance + S_type) * p_private + private_without * S_TN * p_private)
        else:
            private_score = 0

        TP = [0, 0]
        TN = [0, 0]
        FP = [0, 0]
        FN = [0, 0]
        FL = [0, 0]
        TPtype = [0, 0]
        TPdist = [0, 0]

        for (i, run_id) in enumerate(ids):
            solution = [run_id, pred_y[i][0], pred_y[i][1]]
            try:
                score = 0
                if truth[run_id][PUBLIC] == 1:
                    p = p_public
                    part = 0
                else:
                    p = p_private
                    part = 1
                
                # If there is a source in this trial:
                if truth[run_id][SRC_ID] != 0:
                    
                    v = float(truth[run_id][VELOCITY])
                    d0 = float(truth[run_id][STANDOFF])
                    distance_in_meters = abs(float(solution[SRC_TIME]) - float(truth[run_id][SRC_TIME])) * v
                    
                    # False negative:
                    if (int(solution[SRC_ID]) == 0):
                        score += S_FN * p
                        FN[part] += 1
                    else:
                    
                        # Something is detected really close?
                        if distance_in_meters < d0:
                            distance_bonus = math.cos((distance_in_meters/d0) * (math.pi/2));
                            score += S_distance * distance_bonus * p
                            TP[part] += 1
                            TPdist[part] += distance_bonus
                            # Good identification?
                            if (int(solution[SRC_ID]) == truth[run_id][SRC_ID]):
                                score += S_type * p
                                TPtype[part] += 1
                            else:
                                #print("badtype: %d" % run_id)
                                pass
                        else:
                            score += S_FP * p
                            FL[part] += 1
                            #print("FL: %d" % run_id)
                            #FP[part] += 1
                            #FN[part] += 1
                else:
                    # There is no source in this trial
                    if (int(solution[SRC_ID]) == 0):
                        # True negative:
                        score += S_TN * p
                        TN[part] += 1
                    else:
                        # False positive:
                        score += S_FP * p
                        FP[part] += 1
                        print("FP: %d" % run_id)
                
                # If Public field == 1 in ground truth file, add to public score:
                if truth[run_id][PUBLIC] == 1:
                    public_score += score
                else:
                    private_score += score
            except Exception as e:
                print(str(e))
                return -1

        if verbose:
            part = 0
            print("  Public part:")
            print("    TP =", '{:>5}'.format(TP[part]), "|", "FP =", '{:>5}'.format(FP[part]))
            print("    FL =", '{:>5}'.format(FL[part]), "|-----------")
            print("    FN =", '{:>5}'.format(FN[part]), "|", "TN =", '{:>5}'.format(TN[part]))
            print("")
            print("    Correct type: ", TPtype[part], "/", TP[part], "=", "{0:.2f}".format(100 * TPtype[part] / max(1, TP[part])), "%")
            print("    Average distance bonus:", "{0:.2f}".format(100 * TPdist[part] / max(1, TP[part])),"%")
            print("")

        return round(public_score, 6)

    def write_submission(self, pred_y, ids, fname):
        fpath = os.path.join(self._base_path, fname)
        dat=np.zeros((len(ids),3))
        dat[:,0] = ids
        dat[:,1:] = pred_y[:,0:]
        data = pd.DataFrame(columns=["RunID","SourceID","SourceTime"],data=dat)
        data["RunID"] = data["RunID"].astype(np.int64)
        data["SourceID"] = data["SourceID"].astype(np.int64)
        data.set_index("RunID",inplace=True)
        data.to_csv(fpath)
