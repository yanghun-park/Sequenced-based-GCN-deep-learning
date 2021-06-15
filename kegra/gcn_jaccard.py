# =================================================================
# 이 파일은 자카드 유사도 관련 함수가 내장되어 있습니다.
# 최종수정 : 2021/06/09 (1.0v)
# =================================================================
import numpy as np
import gcn_main

def JACCARD_seq(file1, file2):
    totalAB = set(file1).union(set(file2))
    interAB = set(file1).intersection(set(file2))

    try:
        return len(interAB) / len(totalAB)  # 교집합AB 분의 합집합AB
    except ZeroDivisionError:
        return 0


# 중복되는 데이터가 있는지 검색하는 함수
def Search_unique(data, unique_array):
	for i in range(0, len(unique_array)):
		if (data == unique_array[i]):
			return True # 중복이 있으면 참 반환
	return False


# 자카드 유사도 전체 프로세스 (타입0-전체, 타입1-한행)
def JaccardSim(Xarr, fileNum, type):
    # 자카드 파일비교를 위한 배열 생성
    JACCARD_all = [[0 for col in range(len(Xarr))] for row in range(len(Xarr))]
    JACCARD_unique_array = []

    # 자카드 파일비교
    for i in range(0, len(Xarr)):
        for j in range(0, len(Xarr)):
            print("i :", str(i) + " / j :", str(j));
            if (i == j):  # i와 j가 같을경우 같은파일이므로 0행렬 처리
                JACCARD_all[i][j] = 0
            if (i < j):
                JACCARD_all[i][j] = JACCARD_seq(Xarr[i], Xarr[j])
            elif (i > j): # 어자피 i < j랑 값이 같으니 최적화를 위해서 0행렬 처리
                JACCARD_all[i][j] = JACCARD_seq(Xarr[i], Xarr[j])

    if type == 0:
        # 배열에 중복된값 제거 (ex: 0...)
        for i in range(0, len(Xarr)):
            for j in range(0, len(Xarr)):
                if not Search_unique(JACCARD_all[i][j], JACCARD_unique_array):
                    JACCARD_unique_array.append(JACCARD_all[i][j])

        # 정렬(내림차순)
        JACCARD_unique_array.sort(reverse=True)

        # 최종 자카드 배열 만들기
        JACCARD_ROW = np.zeros((fileNum, fileNum))
        for i in range(0, len(Xarr)):
            for j in range(0, len(Xarr)):
                if (i > j):
                    if (JACCARD_all[i][j] >= JACCARD_unique_array[gcn_main.JaccardTop]):
                        JACCARD_ROW[i][j] = 1
                elif (i < j):
                    if (JACCARD_all[i][j] >= JACCARD_unique_array[gcn_main.JaccardTop]):
                        JACCARD_ROW[i][j] = 1


    elif type == 1:
        # 배열에 중복된값 제거 (ex: 0...)
        for i in range(0, len(Xarr)):
            JACCARD_unique_array.append([])
            for j in range(0, len(Xarr)):
                if not Search_unique(JACCARD_all[i][j], JACCARD_unique_array):
                    JACCARD_unique_array[i].append(JACCARD_all[i][j])

        # 정렬(내림차순)
        for i in range(0, len(JACCARD_unique_array)):
            JACCARD_unique_array[i].sort(reverse=True)

        # 최종 자카드 배열 만들기
        JACCARD_ROW = np.zeros((fileNum, fileNum))
        for i in range(0, len(Xarr)):
            for j in range(0, len(Xarr)):
                if (i > j):
                    if (JACCARD_all[i][j] >= JACCARD_unique_array[i][gcn_main.JaccardTop]):
                        JACCARD_ROW[i][j] = 1
                elif (i < j):
                    if (JACCARD_all[i][j] >= JACCARD_unique_array[i][gcn_main.JaccardTop]):
                        JACCARD_ROW[i][j] = 1

    return JACCARD_ROW

