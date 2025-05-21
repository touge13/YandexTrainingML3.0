import numpy as np 


# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    matches = []

    # Находим ближайшие дескрипторы из des1 к des2 и обратно
    best_des1_to_des2 = []
    best_des2_to_des1 = []

    for i in range(len(des1)):
        min_dist = float('inf')
        best_j = -1
        for j in range(len(des2)):
            dist = np.linalg.norm(des1[i] - des2[j])
            if dist < min_dist:
                min_dist = dist
                best_j = j
        best_des1_to_des2.append((best_j, min_dist))

    for j in range(len(des2)):
        min_dist = float('inf')
        best_i = -1
        for i in range(len(des1)):
            dist = np.linalg.norm(des2[j] - des1[i])
            if dist < min_dist:
                min_dist = dist
                best_i = i
        best_des2_to_des1.append((best_i, min_dist))

    # Проверяем взаимность совпадений
    for i in range(len(des1)):
        j, dist = best_des1_to_des2[i]
        if best_des2_to_des1[j][0] == i:
            match = DummyMatch(i, j, dist)
            matches.append(match)

    # Сортируем по расстоянию
    matches.sort(key=lambda x: x.distance)

    return matches