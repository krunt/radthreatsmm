import alg_radmm_simple
from sklearn.model_selection import train_test_split

def main():
    radmmAlg = alg_radmm_simple.AlgRadMMSimple("/mnt/ssd/radiologicalthreatsmm")
    train_ids = radmmAlg.get_train_ids()
    train_ids_set, val_ids_set = train_test_split(train_ids, random_state=13, test_size=0.20, shuffle=True)
    train_x = radmmAlg.get_train_x(train_ids_set)
    train_y = radmmAlg.get_train_y(train_ids_set)
    val_x = radmmAlg.get_train_x(val_ids_set)
    radmmAlg.train(train_x, train_y, train_ids_set)
    pred_y = radmmAlg.predict(val_x, val_ids_set)
    print(radmmAlg.score(pred_y, val_ids_set))


if __name__ == '__main__':
    main()
