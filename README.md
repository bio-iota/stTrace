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
```

Please enter the folder path of the file, the file name and set the weight parameter.
```python
data_path = 'the folder path of the file'
count_file ='the file name.h5'

# set weight parameter
# Iterative Refine Spatial-Temporal Domain
weight_sc = 0.8
# Reconstruction of developmental path
weight_sr = 0.3
weight_dis = 0.6
weight_corr = 0.1
```
## Dara Processing
### Load Data
```python
def read_10X_Visium(path, 
                    genome=None,
                    count_file=count_file, 
                    library_id=None, 
                    load_images=True, 
                    quality='hires',
                    image_path = None):
    adata = sc.read_visium(path, 
                        genome=genome,
                        count_file=count_file,
                        library_id=library_id,
                        load_images=load_images)
    adata.var_names_make_unique()

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        
        for i in range(len(adata.obsm["spatial"]) ):
            adata.obsm["spatial"][i][0] = int(adata.obsm["spatial"][i][0])
            adata.obsm["spatial"][i][1] = int(adata.obsm["spatial"][i][1])

        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata
```
``` python
adata = read_10X_Visium(data_path)
adata
```
``` python
# save spatial location
# spatial location
coor = pd.DataFrame(adata.obs['imagecol'])
coor['imagerow'] = adata.obs['imagerow']
```
### Extract image feature
``` python
# slice path of image feature
def image_crop(
        adata,
        save_path,
        library_id=None,
        crop_size=50,
        target_size=224,
        verbose=False,
        ):
    if library_id is None:
       library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
            adata.uns["spatial"][library_id]["use_quality"]]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)
    tile_names = []

    with tqdm(total=len(adata),
              desc="Tiling image",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.LANCZOS) ##### 
            tile.resize((target_size, target_size)) ###### 
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)

    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata


# image matrix
class image_feature:
    def __init__(
        self,
        adata,
        pca_components=50,
        cnnType='ResNet50',
        verbose=False,
        seeds=88,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType

    def load_cnn_model(
        self,
        ):

        if self.cnnType == 'ResNet50':
            cnn_pretrained_model = models.resnet50(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Resnet152':
            cnn_pretrained_model = models.resnet152(pretrained=True)
            cnn_pretrained_model.to(self.device)            
        elif self.cnnType == 'Vgg19':
            cnn_pretrained_model = models.vgg19(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg16':
            cnn_pretrained_model = models.vgg16(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'DenseNet121':
            cnn_pretrained_model = models.densenet121(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Inception_v3':
            cnn_pretrained_model = models.inception_v3(pretrained=True)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(
                    f"""\
                        {self.cnnType} is not a valid type.
                        """)
        return cnn_pretrained_model

    def extract_image_feat(
        self,
        ):

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std =[0.229, 0.224, 0.225]),
                          transforms.RandomAutocontrast(),
                          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                          transforms.RandomInvert(),
                          transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                          transforms.RandomSolarize(random.uniform(0, 1)),
                          transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                          transforms.RandomErasing()
                          ]
        img_to_tensor = transforms.Compose(transform_list)

        feat_df = pd.DataFrame()
        model = self.load_cnn_model()
       
        model.eval()

        if "slices_path" not in self.adata.obs.keys():
             raise ValueError("Please run the function image_crop first")

        with tqdm(total=len(self.adata),
              desc="Extract image feature",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
            for spot, slice_path in self.adata.obs['slices_path'].items():
                spot_slice = Image.open(slice_path)
                spot_slice = spot_slice.resize((224,224))
                spot_slice = np.asarray(spot_slice, dtype="int32")
                spot_slice = spot_slice.astype(np.float32)
                tensor = img_to_tensor(spot_slice)
                tensor = tensor.resize_(1,3,224,224)
                tensor = tensor.to(self.device)
                result = model(Variable(tensor))
                result_npy = result.data.cpu().numpy().ravel()
                feat_df[spot] = result_npy
                feat_df = feat_df.copy()
                pbar.update(1)
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata 
```
``` python
save_path=os.path.join(data_path, 'image_slice_result')
if not os.path.exists(save_path):
    os.mkdir(save_path)
    print(f"Directory '{save_path}' created successfully")
crop_adata = image_crop(adata, save_path)
feature_adata = image_feature(crop_adata).extract_image_feat()
```
### Spatial Weight/ Gene Correlation/ Morphological Similarity
``` python
# SW 
def cal_spatial_weight(
    data,
    spatial_k = 50,
    spatial_type = "BallTree",
    ):
    from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
    if spatial_type == "NearestNeighbors":
        nbrs = NearestNeighbors(n_neighbors=spatial_k+1, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)
    elif spatial_type == "KDTree":
        tree = KDTree(data, leaf_size=2) 
        _, indices = tree.query(data, k=spatial_k+1)
    elif spatial_type == "BallTree":
        tree = BallTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k+1)
    indices = indices[:, 1:]
    spatial_weight = np.zeros((data.shape[0], data.shape[0]))
    for i in range(indices.shape[0]):
        ind = indices[i]
        for j in ind:
            spatial_weight[i][j] = 1
    return spatial_weight

# GC
def cal_gene_weight(
    data,
    n_components=50,
    gene_dist_type = "cosine",
    ):
    pca = PCA(n_components = n_components)
    if isinstance(data, np.ndarray):
        data_pca = pca.fit_transform(data)
    elif isinstance(data, csr_matrix):
        data = data.toarray()
        data_pca = pca.fit_transform(data)
    gene_correlation = 1 - pairwise_distances(data_pca, metric = gene_dist_type)
    return gene_correlation

# merge SW，GC，MS
def cal_weight_matrix(
        adata,
        platform = "Visium",
        pd_dist_type="euclidean",
        md_dist_type="cosine",
        gb_dist_type="correlation",
        n_components = 50,
        no_morphological = True,
        spatial_k = 30,
        spatial_type = "KDTree",
        verbose = False,
        ):
##################### SW   
    if platform in ["Visium", "ST"]:
        if platform == "Visium":
            img_row = adata.obs["imagerow"]
            img_col = adata.obs["imagecol"]
            array_row = adata.obs["array_row"]
            array_col = adata.obs["array_col"]
            rate = 3
        elif platform == "ST":
            img_row = adata.obs["imagerow"]
            img_col = adata.obs["imagecol"]
            array_row = adata.obs_names.map(lambda x: x.split("x")[1])
            array_col = adata.obs_names.map(lambda x: x.split("x")[0])
            rate = 1.5
        reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
        reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
        physical_distance = pairwise_distances(
                                    adata.obs[["imagecol", "imagerow"]], 
                                    metric=pd_dist_type)
        unit = math.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
        physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
    else:
        physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type)
    print("Physical distance calculting Done!")
    print("The number of nearest tie neighbors in physical distance is: {}".format(physical_distance.sum()/adata.shape[0]))

####################### GC
    gene_counts = adata.X.copy()
    if platform in ["Visium", "ST", "slideseqv2", "stereoseq"]:
        gene_correlation = cal_gene_weight(data = gene_counts, 
                                            gene_dist_type = gb_dist_type, 
                                            n_components = n_components)
    else:
        gene_correlation = 1 - pairwise_distances(gene_counts, metric = gb_dist_type)
    del gene_counts

    print("Gene correlation calculting Done!")


    if verbose:
        adata.obsm["gene_correlation"] = gene_correlation
        adata.obsm["physical_distance"] = physical_distance
############################ MS
    if platform in ['Visium', 'ST']: 
        morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric=md_dist_type)
        morphological_similarity[morphological_similarity < 0] = 0
        print("Morphological similarity calculting Done!")

        if verbose:
            adata.obsm["morphological_similarity"] = morphological_similarity
        adata.obsm["weights_matrix_all"] = (physical_distance
                                                *gene_correlation
                                                *morphological_similarity)
        if no_morphological:
            adata.obsm["weights_matrix_nomd"] = (gene_correlation
                                                *physical_distance)	
        print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")
    else:
        adata.obsm["weights_matrix_nomd"] = (gene_correlation
                                                * physical_distance)
        print("The weight result of image feature is added to adata.obsm['weights_matrix_nomd'] !")
    return adata
```
```python
# adjacency matrix
def find_adjacent_spot(
    adata,
    use_data = "raw",
    neighbour_k = 4,
    weights='weights_matrix_all',
    verbose = False,
    ):
    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            gene_matrix = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X
        elif isinstance(adata.X, pd.Dataframe):
            gene_matrix = adata.X.values
        else:
            raise ValueError(f"""{type(adata.X)} is not a valid type.""")
    else:
        gene_matrix = adata.obsm[use_data]
    weights_matrix = adata.obsm[weights]
    weights_list = []
    final_coordinates = []
    with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
        for i in range(adata.shape[0]):
            if weights == "physical_distance":
                current_spot = adata.obsm[weights][i].argsort()[-(neighbour_k+3):]
            else:
                current_spot = adata.obsm[weights][i].argsort()[-neighbour_k:]
            spot_weight = adata.obsm[weights][i][current_spot]
            spot_matrix = gene_matrix[current_spot]
            if spot_weight.sum() > 0:
                spot_weight_scaled = (spot_weight / spot_weight.sum())
                weights_list.append(spot_weight_scaled)
                spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1,1), spot_matrix)
                spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
            else:
                spot_matrix_final = np.zeros(gene_matrix.shape[1])
                weights_list.append(np.zeros(len(current_spot)))
            final_coordinates.append(spot_matrix_final)
            pbar.update(1)
        adata.obsm['adjacent_data'] = np.array(final_coordinates)
        adata.obsm['adjacent_weight'] = np.array(weights_list)
        return adata

# gene expression
def augment_gene_data(
    adata,
    adjacent_weight = 0.2,
    ):

    adjacent_gene_matrix = adata.obsm["adjacent_data"].astype(float)
    adjacent_gene_matrix[adjacent_gene_matrix == 0] = np.nan
    # adjusted_count_matrix = np.nanmean(np.array([gene_matrix, adjacent_gene_matrix]), axis=0)
    
    if isinstance(adata.X, np.ndarray):
        # augement_gene_matrix =  adata.X + adata.obsm['adjacent_weight'] * adjacent_gene_matrix
        augement_gene_matrix = np.nanmean(np.array([adata.X, adjacent_gene_matrix]), axis=0)
    elif isinstance(adata.X, csr_matrix):
        # augement_gene_matrix = adata.X.toarray() + adata.obsm['adjacent_weight'] * adjacent_gene_matrix
        augement_gene_matrix = np.nanmean(np.array([adata.X.toarray(), adjacent_gene_matrix]), axis=0)
    adata.obsm["augment_gene_data"] = augement_gene_matrix
    
    
    del adjacent_gene_matrix
    return adata

# combining 3 weight values to find adjacent points and calculate enhanced gene expression
def augment_adata(
    adata,
    platform = "Visium",
    pd_dist_type="euclidean",
    md_dist_type="cosine",
    gb_dist_type="correlation",
    n_components = 50,
    no_morphological = False,
    use_data = "raw",
    neighbour_k = 4,
    weights = "weights_matrix_all",
    # adjacent_weight = 0.2,
    spatial_k = 30,
    spatial_type = "KDTree"
    ):
    adata = cal_weight_matrix(
                adata,
                platform = platform,
                pd_dist_type = pd_dist_type,
                md_dist_type = md_dist_type,
                gb_dist_type = gb_dist_type,
                n_components = n_components,
                no_morphological = no_morphological,
                spatial_k = spatial_k,
                spatial_type = spatial_type,
                )
    adata = find_adjacent_spot(adata,
                use_data = use_data,
                neighbour_k = neighbour_k,
                weights = weights)
    adata = augment_gene_data(adata,
                # adjacent_weight = adjacent_weight,
                             )
    return adata
def get_augment(
    adata,
    # adjacent_weight = 0.3,
    neighbour_k = 4,
    weights = "weights_matrix_all",
    spatial_k = 30,
    ):
    adata_augment = augment_adata(adata, 
                            neighbour_k = neighbour_k,
                            platform = "Visium",
                            weights = weights,
                            spatial_k = spatial_k)

    print("Step 1: Augment gene representation is Done!")
    return adata_augment

```
```python
data_proc = get_augment(feature_adata)
enhanced_gene = pd.DataFrame(data_proc.obsm['augment_gene_data'])
```
## Confidence Score
``` python
cell_name = pd.DataFrame(adata.obs['in_tissue'].index)
sr_file_name = 'SR.csv'
sr = pd.read_csv(os.path.join(data_path, sr_file_name),index_col = 0)
if "Index <- colnames(expressiondata)" in sr.columns:
    sr = sr.set_index("Index <- colnames(expressiondata)")
# construct distance matrix based on SR
sr_matrix = pd.DataFrame(index = sr.index, columns = sr.index)
for i in sr_matrix.index:
    sr_matrix[i] = abs(sr['SR'] - sr['SR'][i])
# construct correlation matrix
enhanced_gene_correlation = 1 - pairwise_distances(enhanced_gene, metric = 'cosine')
enhanced_input_corr = pd.DataFrame(enhanced_gene_correlation)

```
``` python
def get_affinity(X,n_neighbors):
    aff_m = kneighbors_graph(X, n_neighbors).toarray()
    aff_m = (aff_m + aff_m.T)/2
    aff_m[np.nonzero(aff_m)] = 1
    M_df = pd.DataFrame(aff_m)
    return M_df
```
```python
enhanced_input_graphm = get_affinity(enhanced_gene, 30)
```
```python
sr_matrix.index = enhanced_input_graphm.index 
sr_matrix.columns = enhanced_input_graphm.index 
sr.index = enhanced_input_graphm.index
coor.index = enhanced_input_graphm.index
```
### Detect functional similar and spatial continuous region by Structure Entropy
```python
def get_result(result_df, result):
    for i in result.keys():    
        result_df[i] = 0
        for m in result[i]:
            for n in result[i][m]:
                if i == 1:
                    result_df[i][n] = m
                else:
                    result_df[i][result_df[i-1]== n] = m
    return result_df
```
```python
def load_graph(neighbour_matrix, edge_matrix):
    G = collections.defaultdict(dict)
    for i in neighbour_matrix.index:
        for j in neighbour_matrix.index:
            if neighbour_matrix[i][j]>0:
                G[i][j] = edge_matrix[i][j]       
    return G

class Vertex_se():
    def __init__(self, vid, cid, nodes,g,v):
        self._vid = vid
        self._cid = cid
        self._nodes = nodes
        self._g = g
        self._v = v
        
class cluster_node_se():
    def __init__(self,cid,nodes,g,v):
        self._cid = cid
        self._nodes = nodes
        self._g = g
        self._v = v

class Louvain_se():
    def __init__(self, G,edge_matrix):
        self._G = G
        self.edge_matrix = edge_matrix
        self._m = 0
        self.cluster_dic= {}  # cluster_dic{cluser number: node id}
        self._vid_vertex = {} # _vid_vertex{node id: Vertex id)
    
    def get_delta_s(self,method,cid, vid):
        s = 0
        if method == 'remove':
            self.cluster_dic[cid]._nodes.remove(vid)
            for i in self.cluster_dic[cid]._nodes:
                if vid in self._G[i].keys():
                    s+=self._G[i][vid]
                if i in self._G[vid].keys():
                    s+=self._G[vid][i]
            self.cluster_dic[cid]._nodes.add(vid) 
        if method == 'add':
            self.cluster_dic[cid]._nodes.add(vid)
            for i in self.cluster_dic[cid]._nodes:
                if vid in self._G[i].keys():
                    s+=self._G[i][vid]
                if i in self._G[vid].keys():
                    s+=self._G[vid][i]
            self.cluster_dic[cid]._nodes.remove(vid) 
        return s
  
    def first_stage_se(self,delta_se):
        mod_inc = False 

        for vid in self._G.keys():

            self.cluster_dic[vid] = cluster_node_se(vid,set([vid]),sum(self._G[vid].values()),sum(self._G[vid].values()))

            self._vid_vertex[vid] = Vertex_se(vid, vid, set([vid]),sum(self._G[vid].values()),sum(self._G[vid].values()))
        
        visit_sequence = self._G.keys() 

        random.shuffle(list(visit_sequence)) 
        M = self.edge_matrix.sum().sum()
        while True:

            can_stop = True 
            step = 1
   
            for v_vid in visit_sequence:   

                v_cid = self._vid_vertex[v_vid]._cid

                cid_g = self.cluster_dic[v_cid]._g
                cid_v = self.cluster_dic[v_cid]._v
                original_se = float(cid_g)/float(M)*log2(float(M)/float(cid_v))
       
                cid_Q = {}

                for w_vid in self._G[v_vid].keys():
                    
                    w_cid = self._vid_vertex[w_vid]._cid 
                    
                    if w_cid not in cid_Q.keys():
                        
                        if w_cid == v_cid:
                           
                            vid_g = self._vid_vertex[v_vid]._g
                            vid_v = self._vid_vertex[v_vid]._v
                            se_vid = float(vid_g)/float(M)*log2(float(M)/float(vid_v))
                            # update SE
                            new_g = self.cluster_dic[w_cid]._g - self._vid_vertex[v_vid]._g + self.get_delta_s('remove',w_cid, v_vid)
                            new_v = self.cluster_dic[w_cid]._v - self._vid_vertex[v_vid]._v
                            se_new = float(new_g)/float(M)*log2(float(M)/float(new_v))

                            cid_Q[w_cid] = original_se - se_vid - se_new

                            
                        else:
                            wcid_g = self.cluster_dic[w_cid]._g 
                            wcid_v = self.cluster_dic[w_cid]._v 
                            wcid_se = float(wcid_g)/float(M)*log2(float(M)/float(wcid_v))

                            new_g = self.cluster_dic[w_cid]._g + self._vid_vertex[v_vid]._g - self.get_delta_s('add',w_cid, v_vid)
                            new_v = self.cluster_dic[w_cid]._v + self._vid_vertex[v_vid]._v
                            se_new = float(new_g)/float(M)*log2(float(M)/float(new_v))

                            if len(self.cluster_dic[v_cid]._nodes) == 1:
                                cid_Q[w_cid] = original_se + wcid_se - se_new

                            else:
                                new_vcid_g = self.cluster_dic[v_cid]._g - self._vid_vertex[v_vid]._g + self.get_delta_s('remove',v_cid, v_vid)
                                new_vcid_v = self.cluster_dic[v_cid]._v - self._vid_vertex[v_vid]._v 
                                new_vcid_se = float(new_vcid_g)/float(M)*log2(float(M)/float(new_vcid_v))
                                cid_Q[w_cid] = original_se + wcid_se - se_new - new_vcid_se

                
                if len(cid_Q)>0:
                    cid, max_delta_Q = sorted(cid_Q.items(), key=lambda item: item[1], reverse=True)[0]

                if max_delta_Q > delta_se and cid != v_cid:
                    
                    self._vid_vertex[v_vid]._cid = cid
                    
                    self.cluster_dic[cid]._v = self.cluster_dic[cid]._v + self._vid_vertex[v_vid]._v
                    self.cluster_dic[cid]._g = self.cluster_dic[cid]._g + self._vid_vertex[v_vid]._g - self.get_delta_s('add',cid, v_vid)
                    self.cluster_dic[cid]._nodes.add(v_vid)
                    
                    self.cluster_dic[v_cid]._v = self.cluster_dic[v_cid]._v - self._vid_vertex[v_vid]._v
                    self.cluster_dic[v_cid]._g = self.cluster_dic[v_cid]._g - self._vid_vertex[v_vid]._g + self.get_delta_s('remove',v_cid, v_vid)
                    self.cluster_dic[v_cid]._nodes.remove(v_vid)

                    if len(self.cluster_dic[v_cid]._nodes) == 0:

                        del self.cluster_dic[v_cid]
                       
                    can_stop = False
                    mod_inc = True

            if can_stop:

                break

        result = {}
        for i in self.cluster_dic.keys():
            result[i] = self.cluster_dic[i]._nodes
        return mod_inc, result

    def get_communities(self):
        communities = []
        for cid in self.cluster_dic.keys():
            vertices = self.cluster_dic[cid]._nodes

            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(c)
        return communities

    def execute(self):
        iter_time = 1
        result = {}
        delta_se = 0
        mod_inc, cluster_result = self.first_stage_se(delta_se)
        result[iter_time] = dict(cluster_result)
        print('iter_time',iter_time, 'len(cluster_result)',len(cluster_result))

        return self.get_communities(),result,iter_time
```
```python
sample_select_number = len(enhanced_gene.index)*0.8
```

```python
def get_confidence_matrix(matrix_df, result,iter_num):
    for i in result[iter_num].keys():    
        matrix_df.fillna(0, inplace = True)
        matrix_df.loc[list(result[iter_num][i]),i] = 1
        matrix_df.loc[i,list(result[iter_num][i])] = 1
        
    return matrix_df
```

```python
louvain_se_results = {}
for i in range(1,11):
    consensus_list = random.sample(list(enhanced_gene.index), int(sample_select_number))
    consensus_ge = enhanced_gene.loc[consensus_list]
    consensus_dis_matrix = enhanced_input_corr.loc[consensus_list,consensus_list]
    consensus_input_graphm = enhanced_input_graphm.loc[consensus_list,consensus_list]
    consensus_input_corr = enhanced_input_corr.loc[consensus_list]
    enhanced_G = load_graph(consensus_input_graphm,consensus_dis_matrix)
    enhanced_algorithm_se = Louvain_se(enhanced_G,consensus_input_corr)
    enhanced_communities_se,enhanced_result_se,iter_time  = enhanced_algorithm_se.execute()
    result_se_df = pd.DataFrame(index = consensus_input_graphm.index)
    result_se_df = get_result(result_se_df, enhanced_result_se)
    
    matrix_df = pd.DataFrame(index = enhanced_input_graphm.index, 
                             columns = enhanced_input_graphm.index)
    for j in result_se_df[1].unique():
        cluster_list = list(result_se_df[result_se_df[1]==j].index)
        matrix_df.loc[cluster_list,cluster_list] = 1
    matrix_df.fillna(0, inplace = True)
    louvain_se_results[i] = matrix_df

se_confidence = pd.DataFrame(index = enhanced_input_graphm.index, 
                             columns = enhanced_input_graphm.index)
se_confidence.fillna(0, inplace = True)
for value in louvain_se_results.values():
    se_confidence += value 

diagonal_values = np.diag(se_confidence.values)
se_confidence_dia = se_confidence.div(diagonal_values, axis=1)
se_confidence_value = se_confidence_dia.values
np.fill_diagonal(se_confidence_value, 1)

se_confidence_score = pd.DataFrame(se_confidence_value, 
                                   columns=se_confidence_dia.columns, 
                                   index=se_confidence_dia.index)

```
### Temporal continuous region from development degree
```python
class Vertex_sr():
    def __init__(self, vid, cid, nodes):
        self._vid = vid
        self._cid = cid
        self._nodes = nodes

class cluster_node_sr():
    def __init__(self,cid,nodes):
        self._cid = cid
        self._nodes = nodes
        
class Louvain_sc():
    def __init__(self, G):
        self._G = G
        self._m = 0  
        self.cluster_dic= {}  
        self._vid_vertex = {} 
        self.test_cluster = {}
        self.test_vertex = {}

    def single_sc(self,v_vid,vertex, cluster):
        inner = 0
        inner_count = 0
        outer = 0
        outer_count = 0

        v_cid = vertex[v_vid]._cid
        
        if len(cluster[v_cid]) == 1:
            
            sc_single = 1
        else:
            
            for i in cluster[v_cid]:
                if i in self._G[v_vid].keys():
                    inner += self._G[v_vid][i]
                    inner_count+=1
    
            outer = sum(self._G[v_vid].values()) - inner
            outer_count = len(self._G[v_vid]) - inner_count
            
            if inner_count == 0:
                a = 0
            else:
                a = inner/inner_count
            if outer_count == 0:
                b = 0
            else:
                b = outer/outer_count
            # b = outer/outer_count
            sc_single = (a-b)/max(a,b)
        return sc_single
    
    def first_stage(self):
        mod_inc = False  

        for vid in self._G.keys():

            self.cluster_dic[vid] = set([vid])
            self._vid_vertex[vid] = Vertex_sr(vid, vid, set([vid]))
            self.test_cluster[vid] = set([vid])
            self.test_vertex[vid] = Vertex_sr(vid, vid, set([vid]))  
            
        visit_sequence = self._G.keys()

        random.shuffle(list(visit_sequence)) 
        
        while True:

            can_stop = True  
            step = 0
            for v_vid in visit_sequence:   
                step+=1
                v_cid = self._vid_vertex[v_vid]._cid

                cid_Q = {}
                loop = 0
                for w_vid in self._G[v_vid].keys():
                    w_cid = self._vid_vertex[w_vid]._cid 
                    if w_cid not in cid_Q:
                        original_sc = sum(self.single_sc(k,self._vid_vertex,self.cluster_dic) for k in self.cluster_dic[v_cid])/len(self.cluster_dic[v_cid])
                        if w_cid == v_cid:
                            self.test_cluster[w_cid].remove(v_vid)
                            new_sc = sum(self.single_sc(k,self.test_vertex,self.test_cluster) for k in self.test_cluster[w_cid])/len(self.test_cluster[w_cid])
                            if w_cid == v_vid:
                                cid_Q[len(visit_sequence)] = original_sc-new_sc-1
                            else:
                                cid_Q[v_vid] = original_sc-new_sc-1
                            self.test_cluster[w_cid].add(v_vid)
                        else:
                            original_sc += sum(self.single_sc(k,self._vid_vertex,self.cluster_dic) for k in self.cluster_dic[w_cid])/len(self.cluster_dic[w_cid])
                            self.test_cluster[v_cid].remove(v_vid)
                            self.test_cluster[w_cid].add(v_vid)
                            self.test_vertex[v_vid]._cid = w_cid                           
                            if len(self.test_cluster[v_cid]) == 0:
                                new_sc = 0
                            else:
                                new_sc = sum(self.single_sc(k,self.test_vertex,self.test_cluster) for k in self.test_cluster[v_cid])/len(self.test_cluster[v_cid])

                            new_sc += sum(self.single_sc(k,self.test_vertex,self.test_cluster) for k in self.test_cluster[w_cid])/len(self.test_cluster[w_cid])
                            
                            delta_Q = original_sc - new_sc 
                            cid_Q[w_cid] = delta_Q
                            
                            self.test_cluster[v_cid].add(v_vid)
                            self.test_cluster[w_cid].remove(v_vid)
                            self.test_vertex[v_vid]._cid = v_cid

                        
                if len(cid_Q)>0:
                    cid, max_delta_Q = sorted(cid_Q.items(), key=lambda item: item[1], reverse=True)[0]

                if max_delta_Q > 0.0 and cid != v_cid:
                    self._vid_vertex[v_vid]._cid = cid
                    self.test_vertex[v_vid]._cid = cid
                    self.test_cluster[cid].add(v_vid)
                    self.cluster_dic[cid].add(v_vid)
                    self.cluster_dic[v_cid].remove(v_vid)
                    self.test_cluster[v_cid].remove(v_vid)
                    
                    if len(self.cluster_dic[v_cid]) == 0:

                        del self.cluster_dic[v_cid]
                        del self.test_cluster[v_cid]

                    can_stop = False
                    mod_inc = True

            if can_stop: 
                break
        return mod_inc, self.cluster_dic 

    def get_communities(self):
        communities = []
        for vertices in self.cluster_dic.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(c)
        return communities
        
    def execute(self):
        iter_time = 1
        result = {}
        mod_inc, cluster_result = self.first_stage()
        result[iter_time] = dict(cluster_result)
        print('iter_time',iter_time, 'len(cluster_result)',len(cluster_result))

        return self.get_communities(),result
```

```python
louvain_sc_results = {}
for i in range(1,11):
    consensus_list = random.sample(list(enhanced_gene.index), int(sample_select_number))
    consensus_ge = enhanced_gene.loc[consensus_list]
    consensus_sc_matrix = sr_matrix.loc[consensus_list,consensus_list]
    consensus_input_graphm = enhanced_input_graphm.loc[consensus_list,consensus_list]
    consensus_input_corr = enhanced_input_corr.loc[consensus_list]
    sc_enhanced_G = load_graph(consensus_input_graphm,consensus_sc_matrix)
    algorithm_sc = Louvain_sc(sc_enhanced_G)
    communities_sc,result_sc = algorithm_sc.execute()
    result_sc_df = pd.DataFrame(index = consensus_input_graphm.index)
    result_sc_df = get_result(result_sc_df, result_sc)
    
    sc_matrix_df = pd.DataFrame(index = enhanced_input_graphm.index, 
                                columns = enhanced_input_graphm.index)
    for j in result_sc_df[1].unique():
        cluster_list = list(result_sc_df[result_sc_df[1]==j].index)
        sc_matrix_df.loc[cluster_list,cluster_list] = 1
    sc_matrix_df.fillna(0, inplace = True)
    louvain_sc_results[i] = sc_matrix_df

sc_confidence = pd.DataFrame(index = enhanced_input_graphm.index, columns = enhanced_input_graphm.index)
sc_confidence.fillna(0, inplace = True)
for value in louvain_sc_results.values():
    sc_confidence += value 
diagonal_values = np.diag(sc_confidence.values)
sc_confidence_dia = sc_confidence.div(diagonal_values, axis=1)
sc_confidence_value = sc_confidence_dia.values
np.fill_diagonal(sc_confidence_value, 0)
sc_confidence_score = pd.DataFrame(sc_confidence_value, 
                                    columns=sc_confidence_dia.columns, 
                                    index=sc_confidence_dia.index)

```
```python
# parameter: weight_sc = 0.8
confidence_score = weight_sc *sc_confidence_score + (1-weight_sc)*se_confidence_score
symmetric_confidence_score = (confidence_score+confidence_score.T)/2
```
## Iterative Refine Spatial-Temporal Domain
```python
def consensus(neighbour, chcid, result_df, max):

    consensus_result = result_df.copy()
    for i in result_df.index:
        neighbour_cluster_list = []
        for j in result_df.index:
            
            if neighbour[i][j] == 1:
                neighbour_cluster_list.append(result_df[chcid][j])
                counter = collections.Counter(neighbour_cluster_list)
                # 80%投票
                # print(counter)
                consensus_cluster, max_num = sorted(counter.items(), key=lambda item: item[1], reverse=True)[0]
                if max_num >= max:
                    consensus_result[chcid][i] = consensus_cluster
    return consensus_result

```
```python
else_index = enhanced_input_graphm.index
louvain_iter_num = 0
result  = pd.DataFrame(index = coor.index)
result['result'] = 0

while True:
    louvain_iter_num +=1
    input_enhanced_ge = enhanced_gene.loc[else_index]
    if len(else_index) < 30:
        input_graphm = get_affinity(input_enhanced_ge, len(else_index)-1)
    else:
        input_graphm = get_affinity(input_enhanced_ge, 30)
    input_graphm.index =  input_enhanced_ge.index
    input_graphm.columns = input_enhanced_ge.index
    
    input_matrix = sr_matrix.loc[else_index, else_index]
    confidence_score = symmetric_confidence_score.loc[else_index, else_index]
    input_coor = coor.loc[else_index]

    G = load_graph(input_graphm,confidence_score)
    
    enhanced_algorithm = Louvain_se(G,confidence_score)
    enhanced_communities,enhanced_result,iter_time  = enhanced_algorithm.execute()

    enhanced_communities = sorted(enhanced_communities, key=lambda b: -len(b)) 
    enhanced_count = 0
    for enhanced_communitie in enhanced_communities:
        enhanced_count += 1

    enhanced_result_df = pd.DataFrame(index = input_graphm.index)
    enhanced_result_df = get_result(enhanced_result_df, enhanced_result)
    
    if len(else_index) < 30:
        neighbour1 = get_affinity(input_coor, len(else_index)-1)
    else:
        neighbour1 = get_affinity(input_coor, 30)
    neighbour1.index = input_coor.index
    neighbour1.columns = input_coor.index
    consensus_se1 = consensus(neighbour1,1,enhanced_result_df,24)
    if len(else_index) < 10:
        neighbour2 = get_affinity(input_coor, len(else_index)-1)
    else:
        neighbour2 = get_affinity(input_coor, 10)
    neighbour2.index = input_coor.index
    neighbour2.columns = input_coor.index
    consensus_se2 = consensus(neighbour2,1,consensus_se1,8)

    if len(consensus_se2[1].unique()) == 1:
        sp_cluster_num = consensus_se2[1].unique()
        spnode_index =consensus_se2[consensus_se2[1]==int(sp_cluster_num)].index
    else:
        consensus_se2['sc'] = silhouette_samples(input_matrix, consensus_se2[1], metric='precomputed')
        sp_cluster_num = consensus_se2.groupby(1)['sc'].mean().idxmax()
        spnode_index = consensus_se2[consensus_se2[1]==sp_cluster_num].index
        
    for i in spnode_index:
        result.loc[i]['result'] = louvain_iter_num
    plt.scatter(coor['imagecol'], coor['imagerow'],c = result['result'],s=3,label = result['result'])

    filename = f"weight_{weight}_iter_{louvain_iter_num}.png"
    
    
    plot_folder_path=os.path.join(data_path, 'results_plot')
    if not os.path.exists(plot_folder_path):
        os.mkdir(plot_folder_path)
        print(f"Directory '{plot_folder_path}' created successfully")
        
    save_plot_path = os.path.join(plot_folder_path, filename)
    plt.savefig(save_plot_path)

    print('iter time:',louvain_iter_num,'finished')
    else_index = consensus_se2[consensus_se2[1]!=int(sp_cluster_num)].index
    if len(else_index) == 0:
        print('Finished!')
        break
```
## Reconstruction of developmental path
```python
pca = PCA(n_components=1)
pcage = pd.DataFrame(columns=enhanced_gene.columns)
for i in result['result'].unique():
    srge = enhanced_gene.loc[result[result['result']==i].index]

    reduced_data = pca.fit_transform(srge.T).T
    result_ge = pd.DataFrame(reduced_data, index=[i], columns=enhanced_gene.columns)
    pcage = pd.concat([pcage, result_ge], axis=0)

correlation_pcage = 1 - pairwise_distances(pcage, metric = 'cosine')
correlation_pcage = pd.DataFrame(correlation_pcage,index = pcage.index, columns=pcage.index).sort_index(axis=0).sort_index(axis=1)
correlation_pcage
```
```python
sr.index = coor.index
result['imagecol'] = coor['imagecol']
result['imagerow'] = coor['imagerow']
result['SR'] = sr['SR']
average_sr_by_superid = pd.DataFrame(result.groupby('result')['SR'].mean())
```
```python
coordinates = result[['imagecol', 'imagerow']].values  
labels = result['result'].values 
unique_labels = np.unique(labels)

distance_matrix = np.zeros((len(unique_labels), len(unique_labels)))

for i, label_i in enumerate(unique_labels):
    points_i = coordinates[labels == label_i] 
    for j, label_j in enumerate(unique_labels):
        if i == j:
            distance_matrix[i, j] = 0  
        else:
            points_j = coordinates[labels == label_j] 
            min_distance = np.min(distance.cdist(points_i, points_j))
            distance_matrix[i, j] = min_distance
distance_matrix_df = pd.DataFrame(distance_matrix, index=unique_labels, columns=unique_labels)
```
```python
average_sr_matrix = pd.DataFrame(index = average_sr_by_superid.index, columns = average_sr_by_superid.index)
for i in average_sr_matrix.index:
    average_sr_matrix[i] = abs(average_sr_by_superid['SR'] - average_sr_by_superid['SR'][i])
```
```python
def zscore(distance_matrix):
    upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool))
    
    mean = upper_triangle.stack().mean()
    std = upper_triangle.stack().std()
    

    zscore_matrix = (distance_matrix - mean) / std

    zscore_matrix = (zscore_matrix + zscore_matrix.T) / 2
    
    return zscore_matrix
```
```python
distance_zscore = zscore(distance_matrix_df)
average_sr_zscore = zscore(average_sr_matrix)
correlation_zscore= zscore(correlation_pcage)
```
```python
combined_df = weight_sr*average_sr_zscore + weight_dis*distance_zscore + weight_corr*correlation_zscore
for i in range(len(combined_df)):
    combined_df.iat[i, i] = 0
combined_df

```
```python
combined_df = weight_sr*average_sr_zscore + weight_dis*distance_zscore + weight_corr*correlation_zscore
for i in range(len(combined_df)):
    combined_df.iat[i, i] = 0
combined_df
```
```python
linkage_result = pd.DataFrame(clustering2.children_+1)
new_node_index = average_sr_matrix.index.max()+1
new_index = range(new_node_index, new_node_index + len(linkage_result))

linkage_result.index = new_index
cluster_cell = {}
for i in result['result'].unique():
    cluster_cell[i] = set(result[result['result'] == i].index)
for i in linkage_result.index:
    cluster_cell[i] = cluster_cell[linkage_result.loc[i,0]].union(cluster_cell[linkage_result.loc[i,1]])

```
```python
for i in linkage_result.index:
    plt.scatter(coor['imagecol'], coor['imagerow'],c = result['result'],alpha=0.5,s=1,label = result['result'])
    plt.scatter(coor['imagecol'][list(cluster_cell[linkage_result.loc[i,0]])], 
                coor['imagerow'][list(cluster_cell[linkage_result.loc[i,0]])],
                s = 10)

    plt.scatter(coor['imagecol'][list(cluster_cell[linkage_result.loc[i,1]])], 
                coor['imagerow'][list(cluster_cell[linkage_result.loc[i,1]])],
                s = 10)

    
    filename = f"{linkage_result.loc[i,0]}_{linkage_result.loc[i,1]}_{i}.png"
    linkage_plot_folder_path=os.path.join(data_path, 'linkage_plot')
    if not os.path.exists(linkage_plot_folder_path):
        os.mkdir(linkage_plot_folder_path)
        print(f"Directory '{linkage_plot_folder_path}' created successfully")
        
    save_plot_path = os.path.join(linkage_plot_folder_path, filename)

    plt.savefig(save_plot_path,dpi=300)
    plt.title((i,linkage_result.loc[i,0],linkage_result.loc[i,1]))
    plt.show()
```
```python
for i in result['result'].unique():
    plt.scatter(coor['imagecol'], coor['imagerow'],c = result['result'],alpha=0.5,s=0.3,label = sr_re['re'])
    plt.scatter(coor['imagecol'][list(cluster_cell[i])], 
                coor['imagerow'][list(cluster_cell[i])],
                s = 3)

    filename = f"{i}.png"
    linkage_node_folder_path=os.path.join(data_path, 'linkage_node_plot')
    if not os.path.exists(linkage_node_folder_path):
        os.mkdir(linkage_node_folder_path)
        print(f"Directory '{linkage_node_folder_path}' created successfully")
    
    save_node_path = os.path.join(linkage_node_folder_path, filename)

    plt.savefig(save_path,dpi=300)
    plt.title(i)
    plt.show()
```
