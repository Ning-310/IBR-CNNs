import  numpy as np
from numba import jit

def read(f,c):
    s=[]
    with open(f) as input:
        for column in zip(*(line.strip().split('\t') for line in input)):
            l=list(column)
            #if l[0] in ['Protein IDs','Intensity L','Intensity H']:
            s.append(l[c:])
    if c==0:
       s[0]=[float(i) for i in s[0]]
       s[1] = [float(i) for i in s[1]]
    return s

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)

def eval_mcc(y_true, y_prob, threshold=False,l=[]):
    idx  = np.argsort(y_prob)
    nump = np.sum(y_true)
    numn = idx.size - nump
    tp, tn, fp, fn = nump, 0.0, numn, 0.0
    best_mcc, best_proba, prev_proba = 0.0, -1, -1
    yp=[y_prob[i] for i in idx]
    yt = [y_true[i] for i in idx]
    pl = [l[i] for i in idx]
    for p,proba, y_i in (zip(pl,yp, yt)):
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc, best_proba = new_mcc,  proba
        if y_i == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        sp=tn/(tn+fp)
        sn=tp/(tp+fn)
        fpr=1-sp
        ac=(tp+tn)/(tp+fp+tn+fn)
        f.write(str(p)+'\t'+str(y_i)+'\t'+str(new_mcc)+'\t'+str(tp)+'\t'+str(tn)+'\t'+str(fp)+'\t'+str(fn)+'\t'+str(sp)+'\t'+str(sn)+'\t'+str(fpr)+'\t'+str(ac)+'\n')
    return (best_proba, best_mcc) if threshold else best_mcc

f=open('predict.txt','r')
y=[]
s=[]
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
for l in f:
    sp=l.rstrip().split('\t')
    s.append(float(sp[0]))
    y.append(int(sp[-1]))
fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
mcc=eval_mcc(y,s,threshold=True,l=s)

