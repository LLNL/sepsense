from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
import numpy as np

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def classwise_auc(score_arr, label_arr):
    auc_list = []
    for label in sorted(np.unique(label_arr)):
        mask = label_arr == label
        auc = roc_auc_score(mask, score_arr[:, label])
        auc_list.append(auc)

    return np.mean(auc_list), auc_list

def classwise_ap(score_arr, label_arr):
    ap_list = []
    for label in sorted(np.unique(label_arr)):
        mask = label_arr == label
        ap = average_precision_score(mask, score_arr[:, label])
        ap_list.append(ap)

    return np.mean(ap_list), ap_list

def classwise_fbeta(score_arr, label_arr, beta):
    fbeta_list = []
    score_arr = softmax(score_arr, axis=1)
    for label in sorted(np.unique(label_arr)):
        mask = label_arr == label
        fbeta = fbeta_score(mask, score_arr[:, label].round(), beta, average='macro')
        fbeta_list.append(fbeta)

    return np.mean(fbeta_list), fbeta_list

if __name__=='__main__':
    score_arr = np.load('auc_scores.npy')
    label_arr = np.load('auc_labels.npy')

    avg_auc, auc_list = classwise_auc(score_arr, label_arr)
    avg_ap, ap_list = classwise_ap(score_arr, label_arr)
    avg_fb1, fb1_list = classwise_fbeta(score_arr, label_arr, 1.0)
    avg_fb2, fb2_list = classwise_fbeta(score_arr, label_arr, 2.0)
    print(avg_auc)
    print(avg_ap)
    print(avg_fb1)
    print(avg_fb2)
