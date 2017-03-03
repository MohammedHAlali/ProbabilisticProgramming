import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

%load_ext autoreload
%autoreload 2

import policy_evaluation as pe

value = np.zeros(16)
v1, iter = pe.PolicyEvaluation(pe.PolicyProbability, value)

v_sq = v1.reshape(4,-1)

plt.clf()
sns.heatmap(v_sq, linewidths=1.5, square=True, annot=True)

