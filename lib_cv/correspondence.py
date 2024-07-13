import cv2


def bf_match(des1, des2, distance=50):
    bf_obj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Сопоставление дескрипторов
    matches = bf_obj.match(des1, des2)

    # Сортировка сопоставлений по расстоянию
    matches = sorted(matches, key=lambda x: x.distance)

    # Выбор наилучших совпадающих точек (можно настроить пороговое значение)
    good_matches = []
    for m in matches:
        if m.distance < distance:  # Пороговое значение для фильтрации совпадений
            good_matches.append(m)
    return good_matches


def flann_knn_match(des_1, des_2, k=2, ratio=0.75):
    FLANN_INDEX_LSH = 1
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2, tree=5)
    search_params = dict(checks=50)  # Настройка параметров поиска

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_1, des_2, k=k)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])
    return good_matches


def bf_knn_match(des1, des2, k=2, ratio=0.75):
    bf_obj = cv2.BFMatcher()

    # Сопоставление дескрипторов

    matches = bf_obj.knnMatch(des1, des2, k=k)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])
    return good_matches


def orb(img_1, img_2):
    orb = cv2.ORB_create(
        nfeatures=20_000,  # Увеличение количества ключевых точек
        # scaleFactor=1.1,  # Коэффициент масштаба между уровнями пирамиды
        # nlevels=10,  # Количество уровней в пирамиде
        # edgeThreshold=31,  # Размер блока для подавления неинтересных ключевых точек
        # firstLevel=0,  # Первый уровень пирамиды
        # WTA_K=4,  # Количество точек для выбора при вычислении дескрипторов
        scoreType=cv2.ORB_HARRIS_SCORE,  # Тип вычисления оценок ключевых точек
        # patchSize=10,  # Размер патча для вычисления дескрипторов
        # fastThreshold=20  # Порог для алгоритма FAST
    )

    # Нахождение ключевых точек и дескрипторов на обоих изображениях
    kp_1, des_1 = orb.detectAndCompute(img_1, None)
    kp_2, des_2 = orb.detectAndCompute(img_2, None)
    return kp_1, kp_2, des_1, des_2


def sift(img_1, img_2):
    sift = cv2.SIFT_create(
        # nfeatures=50000
    )

    # find the keypoints and descriptors with SIFT
    kp_1, des_1 = sift.detectAndCompute(img_1, None)
    kp_2, des_2 = sift.detectAndCompute(img_2, None)
    return kp_1, kp_2, des_1, des_2
