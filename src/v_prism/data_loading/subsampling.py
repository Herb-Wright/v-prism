
# import numpy as np
# from numpy.typing import NDArray


# def random_subsample_k(X: NDArray, y: NDArray, k: int) -> tuple[NDArray, NDArray]:
#     P = X.shape[0]
#     if P <= k:
#         return X, y
#     idxs =  np.random.permutation(P)[:k]
#     return X[idxs], y[idxs]

