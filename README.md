# stTrace
Please running stTrace.ipynb with python. 
```python
# Import necessary libraries
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torchvision.transforms as transforms

import os
import time
import math
from math import log2
import numpy as np
import pandas as pd

from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore


from sklearn import metrics
from sklearn.metrics import pairwise_distances,silhouette_samples
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA, NMF, TruncatedSVD, FactorAnalysis, LatentDirichletAllocation, sparse_encode
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import kneighbors_graph

import collections
import umap
import anndata
import random
import scanpy as sc

from sklearn.cluster import AgglomerativeClustering
''' python

### Please enter the folder path of the file and the file name
```python
data_path = 'the folder path of the file'
count_file ='the file name.h5'
'''
