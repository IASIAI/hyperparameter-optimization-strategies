#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:32:29 2017

@author: Bogdan Burlacu
"""

import numpy as np

def generate(dim, size):
    """ generate random points over each dimension """
    x = np.vstack([np.random.uniform(-5.12, 5.12, size) for d in range(dim)])
#    for i in range(dim):
#        x.append(np.random.uniform(-5.12, 5.12, size))
    return x.T

def evaluate(X, A = 10):
    """ Expects a meshgrid as the input parameter X """
    s = np.shape(X)
    size, dim = s[0], s[1]
    
    Z = np.full(size, A * dim, dtype='float64')
    for d in range(dim):
        x = X[:,d]
        Z += x * x - A * np.cos(2 * np.pi * x)
    return Z
