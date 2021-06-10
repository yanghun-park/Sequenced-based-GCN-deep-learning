# =================================================================
# 이 파일은 딥러닝 관련 함수가 내장되어 있습니다.
# 최종수정 : 2021/06/09 (1.0v)
# =================================================================
import os, time, datetime
import utils
import matplotlib.pyplot as plt

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from kegra.layers.graph import GraphConvolution

import gcn_main

def DeepLearning(X, y, A, fold, y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask, FTime):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    if gcn_main.Filter == 'localpool':
        # Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016)
        print("로컬 폴링 필터 사용...")
        A_ = utils.preprocess_adj(A, gcn_main.SYM_norm)
        support = 1
        graph = [X, A_]
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

    elif gcn_main.Filter == 'chebyshev':
        # Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)
        print("Chebyshev 다항식 기저 필터 사용")
        L = utils.normalized_laplacian(A, gcn_main.SYM_norm)
        L_scaled = utils.rescale_laplacian(L)
        T_k = utils.chebyshev_polynomial(L_scaled, gcn_main.MaxDegree)
        support = gcn_main.MaxDegree + 1
        graph = [X] + T_k
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

    else:
        raise Exception('올바르지 않는 필터 타입')

    X_in = Input(shape=(X.shape[1],))

    H = Dropout(0.5)(X_in)
    H = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([H] + G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax')([H] + G)

    # Compile model
    model = Model(inputs=[X_in] + G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999

    print('Build model...')
    model.summary()  # 모델 정보 출력

    # 그래프를 그리기 위한 배열
    train_loss_array = []
    train_acc_array = []
    test_loss_array = []
    test_acc_array = []


    for epoch in range(1, gcn_main.Epoch + 1):
        # Log wall-clock time
        t = time.time()

        # 딥러닝 시작
        model.fit(graph, y_train, sample_weight=train_mask, batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

        # 전체 데이터 예측
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        y_val = y_train
        train_val_loss, train_val_acc = utils.evaluate_preds(preds, [y_train, y_val], [idx_train, idx_val])
        test_val_loss, test_val_acc = utils.evaluate_preds(preds, [y_test, y_val], [idx_test, idx_val])
        train_loss = []
        train_acc = []

        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "test_loss= {:.4f}".format(test_val_loss[0]),
              "test_acc= {:.4f}".format(test_val_acc[0]),
              "time= {:.4f}".format(time.time() - t))

        train_loss_array.append(train_val_loss[0])
        train_acc_array.append(train_val_acc[0])
        test_loss_array.append(test_val_loss[0])
        test_acc_array.append(test_val_acc[0])

        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= gcn_main.Patience:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

    # Testing
    test_loss, test_acc = utils.evaluate_preds(preds, [y_test], [idx_test])
    test_normal_start = int(y.shape[0] * 0.1 * (fold - 1))
    test_normal_end = int(y.shape[0] * 0.1 * fold)

    test_mal_start = int(y.shape[0] * 0.1 * (fold - 1) + y.shape[0] * 0.5)
    test_mal_end = int(y.shape[0] * 0.1 * fold + y.shape[0] * 0.5)

    idx_test_normal = range(test_normal_start, test_normal_end)
    idx_test_mal = range(test_mal_start, test_mal_end)
    idx_test = list(idx_test_normal) + list(idx_test_mal)

    # for i in range(int(total_file_num*0.8), int(total_file_num*0.8+normal_file_num*0.2)):
    # for i in range(int(normal_file_num*0.8), normal_file_num):
    for i in idx_test_normal:
        if (preds[i][0] > preds[i][1]):
            tn += 1
        else:
            fp += 1

    # for i in range(int(total_file_num*0.8+normal_file_num*0.2), total_file_num):
    # for i in range(normal_file_num+int(mal_file_num*0.8), total_file_num):
    for i in idx_test_mal:
        if (preds[i][0] < preds[i][1]):
            tp += 1
        else:
            fn += 1

    # Testing!
    test_loss, test_acc = utils.evaluate_preds(preds, [y_test], [idx_test])
    try:
        Precision = tp / (tp + fp)
    except ZeroDivisionError:
        Precision = 0

    try:
        Recall = tp / (tp + fn)
    except ZeroDivisionError:
        Recall = 0

    try:
        F1_score = 2 * ((Precision * Recall) / (Precision + Recall))
    except ZeroDivisionError:
        F1_score = 0

    try:
        FPR = fp / (fp + tn)
    except ZeroDivisionError:
        FPR = 0

    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))
    print("TP = ", tp, "FP = ", fp, "FN = ", fn, "TN = ", tn)
    print("Precision = ", Precision)
    print("Recall = ", Recall)
    print("FPR = ", FPR)
    print("F1 score = ", F1_score)

    LTime = datetime.datetime.now()  # 종료시간 기록
    print("처리시간 : ", LTime - FTime)  # 처리시간 출력

    #plt.plot(train_loss_array)
    #plt.ylabel("Train Loss")
    #plt.show()

    #plt.plot(train_acc_array)
    #plt.ylabel("Train Acc")
    #plt.show()

    plt.plot(test_loss_array)
    plt.ylabel("Test Loss")
    plt.show()

    plt.plot(test_acc_array)
    plt.ylabel("Test Acc")
    plt.show()

    return tp, fp, fn, tn, Precision, Recall, FPR, F1_score, test_loss[0], test_acc[0]