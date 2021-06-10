# =================================================================
# 이 파일은 파일 처리 관련 함수가 내장되어 있습니다.
# 최종수정 : 2021/06/09 (1.0v)
# =================================================================
import numpy as np
import os
import gcn_main

# 파일을 읽는 함수 (0=일반 시퀀스, 1=프리퀀시 자카드(시퀀스))
def fileRead_Xarr(Normal_seq, Attack_seq, type):
    try:
        for i in os.listdir(Normal_seq): # 정상파일 처리
            if i.endswith(".csv"):
                normal = splitCSV(Normal_seq + "\\" + i, 0, singleLineToArray)
            if type == 1 and i.endswith(".seqcsv"): # 정상 시퀀스
                normal = splitCSV(Normal_seq + "\\" + i, 1, singleLineToArray)
        print("정상파일에 대한 시퀀스 개수 : " + str(len(normal)))

        yTotal = np.zeros(len(normal)) # 정상파일의 개수만큼 0으로 채움

        for i in os.listdir(Attack_seq): # 악성파일 처리
            if i.endswith(".csv"):
                attack = splitCSV(Attack_seq + "\\" + i, 0, singleLineToArray)
            if type == 1 and i.endswith(".seqcsv"): # 악성 시퀀스
                attack = splitCSV(Attack_seq + "\\" + i, 1, singleLineToArray)
        print("악성파일에 대한 시퀀스 개수 : " + str(len(attack)))

        Xarr = np.concatenate((normal, attack), axis=0) # 정상과 악성데이터 합치기
        yTotal = np.hstack((yTotal, np.ones(len(Xarr) - len(yTotal)))) # 악성파일의 개수만큼 1로 채움
        fileNum = len(Xarr) # 파일개수 기록

        print("총 시퀀스 개수 : ", str(len(Xarr)))
        print("--------------------------------------------")

    except:
        return -1

    return Xarr, yTotal, fileNum




# 파일 하나의 시퀀스를 나열하는 함수
def singleLineToArray(line):
    split = line.split(",")
    s = split[1:] # 첫번째칸은 파일이름이므로 제외
    seq = [int(x) for x in s if int_check(x)]
    return seq


# csv 시퀀스파일이 정상적인 정수로 이루어졌는지 확인하는 함수
def int_check(x):
    try:
        int(x) # X를 int형으로 형변환에 성공하면
        return True # 참값 반환
    except Exception:
        return False


# 파일 하나에 대하여 구문분석하는 함수
def parse_single_line(line): #이부분 검사해보기. -> 패러미터 Split 값 변경.
    split = line.split(",")
    name = split[0]
    length = int(split[1])
    seq = split[1:]
    seq = map(int, seq) #convert into int

    return Sequence(name, length, seq)


# CSV파일을 실행하고 콤마 단위로 자르는 함수 (type=0 시퀀스, type=1 프리퀀시)
def splitCSV(csv_file, type, func = parse_single_line):
    sequences = []

    with open(csv_file) as f:
        for line in f:
            try:
                split = line.split(",")
                sequence = func(line)

            except Exception:
                print("Error at file: {}".format(csv_file))
            sequences.append(sequence)

    rows = []

    if type == 0:
        for line in sequences: # 행을 맞추기 위해 0으로 패딩
            rows.append(np.pad(line, (0, gcn_main.Maxlen), 'constant', constant_values=0)[:gcn_main.Maxlen])
            np.concatenate(rows, axis=0).reshape(-1, gcn_main.Maxlen)
    elif type == 1:
        for line in sequences: # 행을 맞추기 위해 0으로 패딩
            rows.append(np.pad(line, (0, gcn_main.Maxlen_seq), 'constant', constant_values=0)[:gcn_main.Maxlen_seq])
            np.concatenate(rows, axis=0).reshape(-1, gcn_main.Maxlen_seq)

    seq = np.array(rows)
    return seq


# 학습데이터와 테스트데이터 개수 조절
def splitData(Testrate, X, y):
    train_num = int(np.floor(Testrate * len(y)) / 2) # 학습 갯수

    normal_X = X[0:int(len(y)/2)]
    attack_X = X[int(len(y)/2):len(y)]
    normal_Y = y[0:int(len(y)/2)]
    attack_Y = y[int(len(y)/2):len(y)]

    normalX_train, normalX_test = normal_X[0:train_num], normal_X[train_num:len(normal_X)]
    normalY_train, normalY_test = normal_Y[0:train_num], normal_Y[train_num:len(normal_Y)]

    attackX_train, attackX_test = attack_X[0:train_num], attack_X[train_num:len(attack_X)]
    attackY_train, attackY_test = attack_Y[0:train_num], attack_Y[train_num:len(attack_Y)]

    X_train, Y_train = np.concatenate((normalX_train, attackX_train), axis=0), np.concatenate((normalY_train, attackY_train), axis=0)
    X_test, Y_test = np.concatenate((normalX_test, attackX_test), axis=0), np.concatenate((normalY_test, attackY_test), axis=0)

    return (X_train, Y_train), (X_test, Y_test)


# 데이터를 나누는 함수
def getSplits(y, Testrate, fold):
    trainRate = Testrate / 2
    testRate = 0.5 - trainRate

    test_normal_start = int(y.shape[0] * testRate * (fold-1))
    test_normal_end = int(y.shape[0] * testRate * fold)

    test_mal_start = int(y.shape[0] * testRate * (fold-1) + y.shape[0] * 0.5)
    test_mal_end = int(y.shape[0] * testRate * fold + y.shape[0] * 0.5)

    print("test_normal_start=", test_normal_start)
    print("test_normal_end=", test_normal_end)
    print("test_mal_start=", test_mal_start)
    print("test_mal_end=", test_mal_end)

    idx_train_normal1 = range(0, test_normal_start)
    idx_train_normal2 = range(test_normal_end, int(y.shape[0]*0.5))
    idx_train_normal = list(idx_train_normal1) + list(idx_train_normal2)
    idx_val_normal = idx_train_normal
    idx_test_normal = range(test_normal_start, test_normal_end)

    idx_train_mal1 = range(int(y.shape[0]*0.5), test_mal_start)
    idx_train_mal2 = range(test_mal_end, y.shape[0])
    idx_train_mal = list(idx_train_mal1) + list(idx_train_mal2)
    idx_val_mal = idx_train_mal
    idx_test_mal = range(test_mal_start, test_mal_end)

    idx_train = list(idx_train_normal) + list(idx_train_mal)
    idx_val = list(idx_val_normal) + list(idx_val_mal)
    idx_test = list(idx_test_normal) + list(idx_test_mal)

    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)

    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])

    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
