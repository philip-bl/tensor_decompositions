
# coding: utf-8

# In[1]:


from functools import partial

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import polynomial_kernel
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt


# In[2]:


X, y = make_moons(n_samples=500, noise=0.2, random_state=404)


# In[3]:


fig, ax = plt.subplots()
X_label_1 = X[y.astype(bool)]
X_label_0 = X[np.logical_not(y)]
ax.scatter(X_label_1[:, 0], X_label_1[:, 1], color="red")
ax.scatter(X_label_0[:, 0], X_label_0[:, 1], color="blue")


# In[4]:


regularization_strength = 1.0
classifier = KernelRidge(
    kernel="polynomial",
    alpha=regularization_strength,
    degree=3
)
classifier.fit(X, y)
plot_decision_regions(X, y, classifier, legend=1)

