import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def correct_accuracy(y_t, y_p, s=None, u=None):
    correct = y_p.eq(y_t).contiguous()
    res = []  # global, mean, tr, ts, h
    correct_k = correct.float().sum(0, keepdim=True)
    correct_m = torch.zeros(4)
    t_list = y_t.unique().tolist()
    for t in t_list:
        c = y_t == t
        correct_m[t] = correct[c].sum(0, keepdim=True) / c.sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / y_t.size(0))[0].item())
    res.append(correct_m.mean().mul_(100.0).item())
    if s is not None and u is not None:
        tr = correct_m[s].mean()
        ts = correct_m[u].mean()
        h = 2. * tr * ts / (tr + ts + 1e-8)
        res.append(tr.mul_(100.0).item())
        res.append(ts.mul_(100.0).item())
        res.append(h.mul_(100.0).item())

    return res


def pos_neg_class(y_t, y_p, nc=4):
    tp, fp, tn, fn = [], [], [], []
    for n in range(nc):
        gtp = y_t == n  # Positive GT
        gtn = y_t != n  # Negative GT
        tp.append((y_p[gtp] == n).sum().item())
        fn.append((y_p[gtp] != n).sum().item())
        fp.append((y_p[gtn] == n).sum().item())
        tn.append((y_p[gtn] != n).sum().item())

    return np.array(tp), np.array(fp), np.array(tn), np.array(fn)


def multiple_accuracy(tp, fp, tn, fn, names):
    precision, recall, F1, G = dict(), dict(), [], dict()
    for i, n in enumerate(names):
        tpr = tp[i] / (tp[i] + fn[i] + 1e-8)
        tnr = tn[i] / (tn[i] + fp[i] + 1e-8)
        tpp = tp[i] / (tp[i] + fp[i] + 1e-8)
        precision[n] = tpp
        recall[n] = tpr
        F1.append(2 * tpr * tpp / (tpr + tpp + 1e-8))
        G[n] = (tpr * tnr) ** 0.5 * 100.

    return precision, recall, np.array(F1), G


def result_compute(output, pred, target, names, s=None, u=None):
    # 1. confusion matrix
    confusion_mat = confusion_matrix(target.tolist(), pred.tolist(), normalize='true') * 100.
    # 2. global acc, mean acc, tr, ts, harmonic acc
    res_accuracy = correct_accuracy(target, pred, s=s, u=u)
    # 3. G mean
    n_class = output.shape[1]
    tp, fp, tn, fn = pos_neg_class(target, pred, nc=n_class)
    precision, recall, F1, G_mean = multiple_accuracy(tp, fp, tn, fn, names)
    # 4. Macro−F1-score
    macro_f1 = np.mean(F1)
    # 5. ROC curve
    y_true = label_binarize(target, classes=[0, 1, 2, 3])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, n in enumerate(names):
        fpr[n], tpr[n], _ = roc_curve(y_true[:, i], output[:, i])
        roc_auc[n] = auc(fpr[n], tpr[n])
    # 6. Macro-average CCR macro-FPR curve
    all_fpr = np.unique(np.concatenate([fpr[n] for n in names]))
    mean_tpr = np.zeros((output.shape[1], all_fpr.shape[0]))
    for i, n in enumerate(names):
        mean_tpr[i] = np.interp(all_fpr, fpr[n], tpr[n])
    ccr_macro_auc = auc(all_fpr, np.mean(mean_tpr, axis=0))
    # 7. harmonic CCR macro-FPR curve
    seen_tpr = np.mean(mean_tpr[s], axis=0)
    unseen_tpr = np.mean(mean_tpr[u], axis=0)
    h_tpr = 2 * seen_tpr * unseen_tpr / (seen_tpr + unseen_tpr)
    h_ccr_macro = auc(all_fpr, h_tpr)

    return confusion_mat, res_accuracy, G_mean, macro_f1, \
        [fpr, tpr, roc_auc], [all_fpr, mean_tpr, ccr_macro_auc], [all_fpr, h_tpr, h_ccr_macro]
