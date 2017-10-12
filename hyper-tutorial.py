#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:27:53 2017

@author: Bogdan Burlacu, Gabriel Marchidan
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # color map
import rastrigin
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from skopt.plots import plot_convergence
from skopt import gp_minimize
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

def plot_rastrigin(n = 100):
    x, y = np.meshgrid(np.linspace(-5.12, 5.12, n), np.linspace(-5.12, 5.12, n), indexing='ij')
    z = (x**2 - 10 * np.cos(2 * 3.14 * x)) + (y**2 - 10 * np.cos(2 * 3.14 * y)) + 20

    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)
    plt.show()
        

plot_rastrigin()

dimensions = 10
num_points = 1000

# fix the seed
np.random.seed(1234)

X = rastrigin.generate(dimensions, num_points)
y = rastrigin.evaluate(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

gb_reg = GradientBoostingRegressor(random_state=1234)
gb_reg.fit(X_train, y_train)
print(gb_reg)
print('Training score: {}'.format(gb_reg.score(X_train, y_train)))
print('Test score: {}'.format(gb_reg.score(X_test, y_test)))
print('-----------------------------------------\n')

# RANDOMIZED SEARCH
# specify parameters and distributions to sample from
xk = np.array([10**x for x in np.arange(-5, 1, 0.5)])
param_dist = { 'max_depth': sp_randint(1, 6), 
         'learning_rate': sp_uniform(0, 1),
         'max_features': ['auto', 'sqrt', 'log2' ]
#         'max_features': sp_randint(1, dimensions+1),
#         'min_samples_split': sp_randint(2, 101),
#         'min_samples_leaf': sp_randint(1, 101) 
         }

# run randomized search
n_iter_search = 200
random_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=1234), param_distributions=param_dist, n_iter=n_iter_search, cv=5)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." % ((time() - start), n_iter_search))

print('Best parameters: {}'.format(random_search.best_params_))
print('Best score: {}'.format(random_search.best_score_))
best_estimator = random_search.best_estimator_
print('training score: {}'.format(best_estimator.score(X_train, y_train)))
print('test score: {}'.format(best_estimator.score(X_test, y_test)))
print('-----------------------------------------\n')

# GRID SEARCH
grid = { 'max_depth': np.arange(1, 6),
         'learning_rate': [10 ** x for x in np.arange(-5, 1, 0.5, dtype='float')],
         'max_features': ['auto', 'sqrt', 'log2' ]
#         'max_features': np.arange(1, dimensions+1),
#         'min_samples_split': np.arange(2, 101),
#         'min_samples_leaf': np.arange(1, 101) }
}
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=1234), param_grid=grid, cv=5)

start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(grid_search.cv_results_['params'])))

print('Best parameters: {}'.format(grid_search.best_params_))
print('Best score: {}'.format(grid_search.best_score_))
best_estimator = grid_search.best_estimator_
print(best_estimator)
print('training score: {}'.format(best_estimator.score(X_train, y_train)))
print('test score: {}'.format(best_estimator.score(X_test, y_test)))
print('-----------------------------------------\n')

# GAUSSIAN PROCESS SEARCH
gb_reg = GradientBoostingRegressor(random_state=1234)

n_calls = 20
def objective(params):
    max_depth, learning_rate, max_features = params

#    gb_reg.set_params(max_depth=max_depth,
#                   learning_rate=learning_rate,
#                   max_features=max_features,
#                   min_samples_split=min_samples_split, 
#                   min_samples_leaf=min_samples_leaf).fit(X_train, y_train)
    
    gb_reg.set_params(max_depth=max_depth,
                      learning_rate=learning_rate,
                      max_features=max_features)

    obj_value = -np.mean(cross_val_score(gb_reg, X_train, y_train, cv=5, n_jobs=1, scoring="neg_mean_absolute_error"))
#    obj_value = 1 - gb_reg.score(X_train, y_train)
    return obj_value

space  = [(1, 5),                           # max_depth
#          (10**-5, 10**0, "log-uniform"),   # learning_rate
          [10 ** x for x in np.arange(-5, 1, 0.5, dtype='float')],
#          (1, dimensions),                  # max_features
          ('auto', 'log2', 'sqrt'),
#          (2, 100),                         # min_samples_split
#          (1, 100)]                         # min_samples_leaf
         ]

start = time()
res_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=1234)

print('gp_minimize took {:.2f} seconds'.format(time() - start))
print('Best score: {}'.format(res_gp.fun))

print("""Best parameters:
- max_depth=%d
- learning_rate=%.6f
- max_features=%s""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2]))

#print("""Best parameters:
#- max_depth=%d
#- learning_rate=%.6f
#- max_features=%d
#- min_samples_split=%d
#- min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1], 
#                            res_gp.x[2], res_gp.x[3], 
#                            res_gp.x[4]))

plot_convergence((res_gp))
max_depth, learning_rate, max_features = res_gp.x
gb_reg = GradientBoostingRegressor(random_state=1234)
gb_reg.set_params(max_depth=max_depth, learning_rate=learning_rate, max_features=max_features)
#gb_reg.set_params(max_depth=max_depth, learning_rate=learning_rate, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf).fit(X_train, y_train)
gb_reg.fit(X_train, y_train)
print(gb_reg)
print('training score: {}'.format(gb_reg.score(X_train, y_train)))
print('test score: {}'.format(gb_reg.score(X_test, y_test)))
#
# a negative score may be observed here, owing to the following explanation:
# The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).


