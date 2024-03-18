import numpy as np
import scipy.signal as sps


eps = np.finfo(float).eps


def normalize(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def standardize(a, axis=-1):
    means = np.mean(a, axis=axis, keepdims=True)
    stds = np.std(a, axis=axis, keepdims=True)
    return (a - means) / stds


def embed_data(x, order, lag):
    ch, N = x.shape
    hidx = np.arange(order * lag, step=lag)
    Nv = N - (order - 1) * lag
    u = np.zeros((order*ch, Nv))
    for i in range(order):
        u[i*ch:(i+1)*ch] = x[:, hidx[i]:hidx[i]+Nv]

    return u


def pTE(z, lag=1, model_order=1, to_norm=False):
    NN, C, T = np.shape(z)
    pte = np.zeros((NN, NN))
    if to_norm:
        z = standardize(sps.detrend(z))
    nodes = np.arange(NN, step=1)

    for i in nodes:
        EmbdDumm = embed_data(z[i], model_order + 1, lag)
        Xtau = EmbdDumm[:-C]
        for j in nodes:
            if i != j:
                Yembd = embed_data(z[j], model_order + 1, lag)
                Y = Yembd[-C:]
                Ytau = Yembd[:-C]
                XtYt = np.concatenate((Xtau, Ytau), axis=0)
                YYt = np.concatenate((Y, Ytau), axis=0)
                YYtXt = np.concatenate((YYt, Xtau), axis=0)

                H_XtYt = np.linalg.det(np.cov(XtYt))
                H_YYt = np.linalg.det(np.cov(YYt))
                H_YYtXt = np.linalg.det(np.cov(YYtXt))
                H_Ytau = np.linalg.det(np.cov(Ytau))

                if H_XtYt > 0 and H_YYt > 0 and H_YYtXt > 0 and H_Ytau > 0:
                    pte[i, j] = 0.5 * (np.log(H_XtYt) + np.log(H_YYt) - np.log(H_YYtXt) - np.log(H_Ytau))

    return pte

#
# cov = numpy.array(
#     [[1.0, 0.5, 0.5, 0.2],
#      [0.5, 1.0, 0.5, 0.1],
#      [0.5, 0.5, 1.0, 0.3],
#      [0.2, 0.1, 0.3, 1.0]])
#
# z = np.random.multivariate_normal([1, 2, 3, 4], cov, 3000)
# cov_m = np.cov(z.T)
#
# z = np.reshape(z.T, (2, 2, -1))
# causal_matrix = pTE(z, model_order=2, to_norm=True)
# print(causal_matrix)