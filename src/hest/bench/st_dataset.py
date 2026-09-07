from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import Dataset
import h5py
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from PIL import Image


class H5PatchDataset(Dataset):
    """Dataset for patch HDF5 files written by HEST/TRIDENT-style patching."""

    def __init__(self, h5_path, img_transform=None):
        self.h5_path = h5_path
        self.img_transform = img_transform
        with h5py.File(self.h5_path, "r") as f:
            self.img_key = "img" if "img" in f else ("imgs" if "imgs" in f else "images")
            self.coords_key = "coords"
            self.barcodes_key = "barcodes" if "barcodes" in f else ("barcode" if "barcode" in f else None)
            self.length = int(f[self.img_key].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            img = f[self.img_key][idx]
            coords = f[self.coords_key][idx]
            if self.barcodes_key is not None:
                barcode = f[self.barcodes_key][idx]
            else:
                barcode = b""

        if isinstance(barcode, np.ndarray) and barcode.shape:
            barcode = barcode[0]
        if isinstance(barcode, bytes):
            barcode = barcode.decode("utf-8")
        else:
            barcode = str(barcode)

        if self.img_transform is not None:
            img_out = self.img_transform(Image.fromarray(img.astype(np.uint8)))
        else:
            img_out = img

        return {
            "imgs": img_out,
            "coords": coords,
            "barcodes": barcode,
        }


def normalize_adata(adata: sc.AnnData, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)
    #print(adata.obs)
    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])
            neighbors_index = adata.obs[((adata.obs['array_row'] >= row - 1) & (adata.obs['array_row'] <= row + 1)) & \
                ((adata.obs['array_col'] >= col - 1) & (adata.obs['array_col'] <= col + 1))].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)
            
            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg
    
    
    # Logarithm of the expression
    sc.pp.log1p(filtered_adata)

    return filtered_adata


def apply_spatial_smoothing(adata: sc.AnnData) -> sc.AnnData:
    """Helper function to run fast, vectorized neighbor averaging."""
    X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.astype(np.float64)
    smoothed_X = np.zeros_like(X_dense)
    
    rows = adata.obs['array_row'].values
    cols = adata.obs['array_col'].values
    
    for i in range(len(adata)):
        r, c = rows[i], cols[i]
        neighbor_mask = (rows >= r - 1) & (rows <= r + 1) & (cols >= c - 1) & (cols <= c + 1)
        smoothed_X[i] = X_dense[neighbor_mask].mean(axis=0)
        
    adata.X = smoothed_X
    return adata


def load_adata(expr_path, genes = None, barcodes = None, normalize=False, feature_type=None, smooth=False):
    adata = sc.read_h5ad(expr_path)
    adata.var_names_make_unique() # TODO: DEBUG: GBMSpace debug
    if barcodes is not None:
        adata = adata[barcodes]

    if feature_type is not None:
        print(f"Restricting  features to {feature_type} type")
        # this is GBMSpace specific as different feature types are stored within the same matrix X 
        # ['Cell state abundances', 'Gene Expression', 'Histopath annotation overlap', 'Spatial niche abundances']
        adata = adata[:, adata.var.feature_types == feature_type]

    # 2. Total Count Normalization
    # Must happen BEFORE gene subsetting so size factors use the entire transcriptome
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)

    # 3. Gene Subsetting
    # Mathematically commutes with row-wise smoothing, so doing it early saves memory and compute
    if genes is not None:
        adata = adata[:, genes]

    # 4. Spatial Smoothing
    # Performed on linear scale (raw or normalized) before log transformation
    if smooth:
        adata = apply_spatial_smoothing(adata)

    # 5. Log Transformation
    if normalize:
        sc.pp.log1p(adata)

    return adata.to_df()
