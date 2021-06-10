import os, datetime
import numpy as np
import scipy.sparse as sp

import gcn_train
import gcn_dataprocess
import gcn_jaccard

# =================================================================
#                          * 환경 설정 *
# =================================================================
# 경로 설정
Normal_seq = os.getcwd() + "\\normal"
Attack_seq = os.getcwd() + "\\attack"
Result_file = "Result.txt"

# 데이터 관련
Testrate = 0.8 # 테스트 데이터 조정
Maxlen = 10896 # 시퀀스 길이(실험 데이터에 따라 변경해야함)
Maxlen_seq = 800 # 시퀀스 길이(프리퀀시용)
SeqMax_features = 10896 # 처리하는 최대 토큰의 수
Filter = "localpool" # 필터
SYM_norm = True
MaxDegree = 2  # 최대 다항식 차수
Epoch = 50
Patience = 10 # 조기 종료 설정
JaccardTop = 3 # 자카드 지수 순위
Fivefold = False # 5-Fold 사용 여부
# =================================================================

def main():
    print("GCN Deep-Learning Program")
    print("----- 초코 5호기 -----")
    print("딥러닝 매커니즘을 선택해 주세요. ")
    print("1 - Identity Matrix(시퀀스)")
    print("2 - JACCARD 유사도 - 전체(시퀀스)")
    print("3 - JACCARD 유사도 - 한 행(시퀀스)")
    print("4 - Identity Matrix(프리퀀시)")
    print("5 - JACCARD 유사도 - 전체(프리퀀시)")
    print("6 - JACCARD 유사도 - 한 행(프리퀀시)")
    print("** 프리퀀시 데이터를 자카드 유사도로 적용할때 시퀀스 파일이 필요합니다. ")
    print("** 이 경우 파일 확장자를 .seqcsv로 변경하시기 바랍니다. ")

    print("선택 : ")
    select = int(input())

    if (select > 0 and select <= 6):
        deeplearningMain(select)
    else:
        print("다시 선택하여 주시기 바랍니다. ")
        main()


def deeplearningMain(select):
    Result = open(Result_file, 'a')

    # Fivefold 전용 변수
    y_train = [0] * 5
    y_val = [0] * 5
    y_test = [0] * 5
    idx_train = [0] * 5
    idx_val = [0] * 5
    idx_test = [0] * 5
    train_mask = [0] * 5

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    FTime = datetime.datetime.now()  # 시작시간 기록

    # 데이터 불러오기
    Xarr, yTotal, fileNum = gcn_dataprocess.fileRead_Xarr(Normal_seq, Attack_seq, 0)

    # y행렬을 만드는 과정
    y = np.zeros((fileNum, 2))
    for i in range(0, fileNum):
        if int(yTotal[i]) == 0: # 만약 정상파일일 경우
            y[i][0] = 1 # 정상 구간에 1추가
        else:
            y[i][1] = 1 # 악성 구간에 1추가

    yTrain = np.array(y)

    X = np.array(Xarr)
    X = X / X.sum(1).reshape(-1, 1)

    if Fivefold: # 만약 FiveFold가 활성화 되있다면
        for i in range(0, 5):
            y_train[i], y_val[i], y_test[i], idx_train[i], idx_val[i], idx_test[i], train_mask[i] = gcn_dataprocess.getSplits(y, Testrate, i + 1)
    else:
        y_train[0], y_val[0], y_test[0], idx_train[0], idx_val[0], idx_test[0], train_mask[0] = gcn_dataprocess.getSplits(y, Testrate, 1)

    (X_train, Y_train), (X_test, Y_test) = gcn_dataprocess.splitData(Testrate, X, y)

    print("--------------------------------------------")
    print(len(X_train), '개의 학습 시퀀스')
    print(len(X_test), '개의 테스트 시퀀스')

    # =================================================================
    #                       * 인접행렬 파트 *
    # =================================================================
    # 만약 프리퀀시 데이터에 자카드 유사도를 적용한다면
    # 자카드 유사도에 계산되는 값은 시퀀스 데이터로 사용
    Xarr_seq = []
    yTotal_seq = []
    fileNum_seq = []
    JACCARD_ROW = []
    A = []

    if select == 2 or select == 3: # 시퀀스 데이터에 자카드 유사도 사용시
        JACCARD_ROW = gcn_jaccard.JaccardSim(Xarr, fileNum, 0)
    elif select == 5 or select == 6: # 프리퀀시 데이터에 자카드 유사도 사용시
        Xarr_seq, yTotal_seq, fileNum_seq = gcn_dataprocess.fileRead_Xarr(Normal_seq, Attack_seq, 1)

        if select == 5: # 전체
            JACCARD_ROW = gcn_jaccard.JaccardSim(Xarr_seq, fileNum, 0)
        elif select == 6: # 한행
            JACCARD_ROW = gcn_jaccard.JaccardSim(Xarr_seq, fileNum, 1)

    if select == 2 or select == 3 or select == 5 or select == 6:
        A = sp.csr_matrix(JACCARD_ROW)  # 희소행렬로 변경

    if select == 1 or select == 4:
        IdentityMatrix = np.identity(fileNum)
        A = sp.csr_matrix(IdentityMatrix)

    # =================================================================
    #                         Deep-Learning
    # =================================================================
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    Precision = 0
    Recall = 0
    FPR = 0
    F1_score = 0
    loss = 0
    acc = 0

    if (Fivefold): # Fivefold 사용시
        for i in range(0, 5):
            TP2, FP2, FN2, TN2, Precision2, Recall2, FPR2, F1_score2, loss2, acc2 = gcn_train.DeepLearning(X, y, A, i + 1, y_train[i], y_val[i], y_test[i], idx_train[i], idx_val[i], idx_test[i], train_mask[i], FTime)
            TP += TP2 # 더하기
            FP += FP2
            FN += FN2
            TN += TN2
            Precision += Precision2
            Recall += Recall2
            FPR += FPR2
            F1_score += F1_score
            loss += loss2
            acc += acc2
        TP /= 5 # 나누기
        FP /= 5
        FN /= 5
        TN /= 5
        Precision /= 5
        Recall /= 5
        FPR /= 5
        F1_score /= 5
        loss /= 5
        acc /= 5
    else:
        TP, FP, FN, TN, Precision, Recall, FPR, F1_score, loss, acc = gcn_train.DeepLearning(X, y, A, 1, y_train[0], y_val[0], y_test[0], idx_train[0], idx_val[0], idx_test[0], train_mask[0], FTime)

    # 결과를 파일로 기록
    Result.write("-----------------------------------------" + "\n")
    Result.write(str(datetime.datetime.now()) + "\n")
    if select == 1:
        Result.write("Identity Matrix(시퀀스)")
    elif select == 2:
        Result.write("JACCARD 유사도 - 전체(시퀀스)")
    elif select == 3:
        Result.write("JACCARD 유사도 - 한 행(프리퀀시)")
    elif select == 4:
        Result.write("Identity Matrix(프리퀀시)")
    elif select == 5:
        Result.write("JACCARD 유사도 - 전체(프리퀀시)")
    elif select == 6:
        Result.write("JACCARD 유사도 - 한 행(프리퀀시)")

    if Fivefold:
        Result.write(" - 5 Fold 적용 \n")
    else:
        Result.write("\n")

    Result.write("TP : " + str(TP) + "\n")
    Result.write("FP : " + str(FP) + "\n")
    Result.write("FN : " + str(FN) + "\n")
    Result.write("TN : " + str(TN) + "\n")
    Result.write("Precision : " + str(Precision) + "\n")
    Result.write("Recall : " + str(Recall) + "\n")
    Result.write("FPR : " + str(FPR) + "\n")
    Result.write("F1_score : " + str(F1_score) + "\n")
    Result.write("loss : " + str(loss) + "\n")
    Result.write("Acc : " + str(acc) + "\n")
    Result.close()


if __name__ == '__main__':
    main()
