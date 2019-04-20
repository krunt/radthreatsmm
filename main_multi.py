import sys
import numpy as np
import alg_radmm_simple
import alg_radmm_imgproc_band
import alg_radmm_nmf_decomp
import alg_radmm_nn_signal_noise
import alg_radmm_tree_signal_noise
import  alg_radmm_nmf_decomp_time
from sklearn.model_selection import train_test_split

def main(part_id, num_parts):
    np.random.seed(13)
    radmmAlg = alg_radmm_nmf_decomp_time.AlgRadMMNmfDecompTime("/mnt/ssd/radiologicalthreatsmm")
    
    train_ids = radmmAlg.get_train_ids()
    train_ids_set, val_ids_set = train_test_split(train_ids, random_state=13, test_size=0.20, shuffle=True)
    #train_x = radmmAlg.get_train_x(train_ids_set, False)
    train_x = None
    radmmAlg.train(train_x, train_ids_set)

    psize = int((len(val_ids_set) + num_parts - 1) / num_parts)
    from_idx = part_id * psize
    to_idx = min((part_id + 1) * psize, len(val_ids_set))

    val_ids_set = val_ids_set[from_idx:to_idx]
    pred_y = radmmAlg.predict(val_ids_set, export=False, validation=True)
    radmmAlg.write_submission(pred_y, val_ids_set, "submission_%d.csv" % (part_id))

    #print("Score: %f" % radmmAlg.score(pred_y, val_ids_set, verbose=True))
    #radmmAlg.write_submission(pred_y, val_ids_set, "submission.csv")

#    test_ids = radmmAlg.get_test_ids()
#    test_x = radmmAlg.get_test_x(test_ids)
#    pred_y = radmmAlg.predict(test_x, test_ids, validation=False)
#    radmmAlg.write_submission(pred_y, test_ids, "solution.csv")

    #radmmAlg.export_predict_trace(val_ids_set)

if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    main(int(sys.argv[1]), int(sys.argv[2]))
