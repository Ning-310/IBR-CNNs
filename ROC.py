import matplotlib.pylab as pylab
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def roc(y_tests, y_test_scores):
    font = {'family': 'arial',
            'weight': 'bold',
            'size': 20 }
    params = {'axes.labelsize': '20',
              'xtick.labelsize': '20',
              'ytick.labelsize': '20',
              'lines.linewidth': '4'}
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    pylab.rcParams['font.weight'] = 'bold'
    plt.figure(figsize=(5, 5), dpi=300)
    AUC = roc_auc_score(y_tests, y_test_scores)
    fpr1, tpr1, thresholds1 = roc_curve(y_tests, y_test_scores)
    plt.plot(fpr1, tpr1, linewidth='3', color='tomato', label='AUC = {:.3f}'.format(AUC))
    plt.plot([0, 1], [0, 1], linewidth='1', color='grey', linestyle="--")
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(prop={'size': 20}, loc=4, frameon=False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    plt.xlabel('1–Specificity', font)
    plt.ylabel('Sensitivity', font)
    plt.savefig('roc.jpg')
    plt.show()

roc(y_tests, y_test_scores)