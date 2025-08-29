from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import joblib
from scipy import signal
import numpy as np
import cv2 as cv
from skimage.transform import rescale
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2 as cv
from skimage.feature import hog, blob_log, blob_dog, blob_doh, canny
from skimage import data, exposure
from skimage.filters import difference_of_gaussians
from skimage.filters import laplace
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, rotate
from skimage.color import rgb2gray
from sklearn.metrics import mean_squared_error, roc_curve, auc
from scipy import stats
from skimage.feature import hessian_matrix, hessian_matrix_det
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn import metrics
import time
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import ast  
import re
from scipy.stats import mode



########################
# Image pre-processing #
########################

""" create a 2-D gaussian blurr filter for a given mean and std """
def create_2d_gaussian(size=9, std=1.5):
    gaussian_1d = signal.windows.gaussian(size,std=std)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d/(gaussian_2d.sum())
    return gaussian_2d


""" normalize teh image between 0 and 1 """
def normalize_img(img):
    normalized = (img - img.min())/(img.max() - img.min())   
    return normalized

""" 
convert to grayscale
normalize teh image between 0 and 1
resize image to im_size """

def preprocess_image(img_path, im_size=512):
    img = cv.imread(img_path)
    img_grayscale = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    normalized = normalize_img(img_grayscale)
    rescaled = rescale_img(img=normalized, standard=im_size)
    return rescaled

def rescale_img(img, standard=256):
  # rescale short side to standard size, then crop center
  scale = standard / min(img.shape[:2])
  img = rescale(img, scale, anti_aliasing=True)
  img = img[int(img.shape[0]/2 - standard/2) : int(img.shape[0]/2 + standard/2),
            int(img.shape[1]/2 - standard/2) : int(img.shape[1]/2 + standard/2)]
  return img

def summarize_blobs(blobs, image_shape):
    if len(blobs) == 0:
        return np.zeros(14)
    
    y, x, r = blobs[:, 0], blobs[:, 1], blobs[:, 2]
    h, w = image_shape

    # Normalize coordinates to [0, 1]
    y_norm = y / h
    x_norm = x / w
    r_norm = r / max(h, w)

    features = [
        len(blobs),                        # total number of blobs
        len(blobs) / (h * w),              # blob density
        np.mean(y_norm), np.std(y_norm), np.min(y_norm), np.max(y_norm),
        np.mean(x_norm), np.std(x_norm), np.min(x_norm), np.max(x_norm),
        np.mean(r_norm), np.std(r_norm), np.min(r_norm), np.max(r_norm)
    ]
    return np.array(features)

# contrast stretching
def contrast_stretching(in_img):
  # get min and max
  min_v = np.min(in_img)
  max_v = np.max(in_img)

  # rescale intensity to min/max range
  out_img = exposure.rescale_intensity(in_img, in_range=(min_v, max_v))
  return out_img

# gamma correction
def gamma_correction(in_img, gamma, with_sigmoid=False):
  # convert img to float64
  f_img = in_img.astype(np.float64)/255.0

  # apply gamma
  if with_sigmoid:
    g_img = 1/(1+np.exp(-gamma*(f_img-0.5)))
  else: 
    g_img = f_img ** gamma

  # convert img back to uint8
  out_img = (g_img * 255).astype(np.uint8)
  return out_img

# histogram equalization
def histogram_equalization(in_img):
  # compute cdf
  img_cdf, bins = exposure.cumulative_distribution(in_img, 256)

  # create empty array for all possible pixel values
  new_cdf = np.zeros(256)

  # populate array with values from cdf
  # use bins as the index into the array
  new_cdf[bins] = img_cdf

  # create empty array the same size as the image
  out_img = np.zeros(in_img.shape, dtype=in_img.dtype)

  # for each pixel, look up the value from the cdf
  for i in range(out_img.shape[0]):
    for j in range(out_img.shape[1]):
      out_img[i, j] = (new_cdf[ in_img[i, j] ] * 255)

  return out_img

#######################
# Feature engineering #
#######################

#============== Complex feature ====================================
''' Complex Features '''


''' Container returned by 'complex' loader 
    Ouput: featureBundle thats fully populated with all below and prints summary'''
@dataclass
class FeatureBundle:
    X: np.ndarray            # (N, D) float32
    y: Optional[np.ndarray]  # (N,) int64 or None
    paths: List[str] #original sample paths
    label2id: Dict[str, int] # mapping from string lbl -> class ID
    id2label: Dict[int, str] # inverse mapping from class ID -> string label
    path2label: Dict[str, str] #mapping from path -> string lable
    mismatches: List[Tuple[int, str, str, int, str]] #rows where y[i] mismatch with path2label[paths[i]]
    class_counts: Dict[str, int] # per class sample counts
    target_label_indices: List[int] #indices in paths belong to target_label
    target_label_ok: bool # Sanity check


def _load_features_from_joblib(
    joblib_path: str,
    label2id: Optional[Dict[str, int]],
    sort_within_label: bool,
    global_sort: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[int, str], Dict[str, int]]:
    data: Any = joblib.load(joblib_path)

    # Packed dict format
    if isinstance(data, dict) and "X" in data and "y" in data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        paths = list(data.get("paths", [f"sample_{i}" for i in range(len(X))]))
        if "label2id" in data and isinstance(data["label2id"], dict):
            label2id_local = {str(k): int(v) for k, v in data["label2id"].items()}
        elif "id2label" in data and isinstance(data["id2label"], dict):
            id2label_local = {int(k): str(v) for k, v in data["id2label"].items()}
            label2id_local = {v: k for k, v in id2label_local.items()}
        else:
            if label2id is None:
                uniq = sorted(set(int(i) for i in y))
                label2id_local = {str(i): i for i in uniq}
            else:
                label2id_local = dict(label2id)
        id2label_local = {i: lab for lab, i in label2id_local.items()}
        return X, y, paths, id2label_local, label2id_local

    # Nested dict {label: {path: vec}}
    if not (isinstance(data, dict) and all(isinstance(v, dict) for v in data.values())):
        raise ValueError("Unsupported joblib format. Expected packed dict or nested {label:{path:vec}}.")

    labels_in_file = sorted(data.keys())
    label2id_local = dict(label2id) if label2id is not None else {lab: i for i, lab in enumerate(labels_in_file)}
    id2label_local = {i: lab for lab, i in label2id_local.items()}

    feats_list: List[np.ndarray] = []
    y_list: List[int] = []
    paths_list: List[str] = []

    if not global_sort:
        for lab in labels_in_file:
            items = data[lab].items()
            if sort_within_label:
                items = sorted(items, key=lambda kv: kv[0])
            for path, vec in items:
                arr = np.asarray(vec, dtype=np.float32)
                feats_list.append(arr)
                y_list.append(label2id_local[lab])
                paths_list.append(path)
    else:
        all_items = []
        for lab, d in data.items():
            for path, vec in d.items():
                all_items.append((path, lab, vec))
        all_items.sort(key=lambda x: x[0])
        for path, lab, vec in all_items:
            arr = np.asarray(vec, dtype=np.float32)
            feats_list.append(arr)
            y_list.append(label2id_local[lab])
            paths_list.append(path)

    if not feats_list:
        raise ValueError("No features found in joblib file.")
    D = feats_list[0].shape[0]
    for i, a in enumerate(feats_list):
        if a.shape[0] != D:
            raise ValueError(f"Feature length mismatch at index {i}: got {a.shape[0]}, expected {D}")

    X = np.vstack(feats_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, paths_list, id2label_local, label2id_local


def _build_path2label(joblib_path: str) -> Dict[str, str]:
    obj = joblib.load(joblib_path)
    # Nested
    if isinstance(obj, dict) and "X" not in obj and all(isinstance(v, dict) for v in obj.values()):
        return {p: lab for lab, dd in obj.items() for p in dd.keys()}
    # Packed
    if isinstance(obj, dict) and "paths" in obj and "y" in obj:
        paths = list(obj["paths"])
        y     = np.asarray(obj["y"]).astype(int)
        if "id2label" in obj and isinstance(obj["id2label"], dict):
            id2label_local = {int(k): v for k, v in obj["id2label"].items()}
        elif "label2id" in obj and isinstance(obj["label2id"], dict):
            id2label_local = {int(v): k for k, v in obj["label2id"].items()}
        else:
            ids = sorted(set(int(i) for i in y))
            id2label_local = {i: str(i) for i in ids}
        return {p: id2label_local[int(cls)] for p, cls in zip(paths, y)}
    raise ValueError("Unsupported joblib format.")

def combine_load_and_validate_joblib(
    joblib_path: str,
    label2id: Optional[Dict[str, int]] = None,
    sort_within_label: bool = True,
    global_sort: bool = False,
    target_label: str = "glioma",
    assert_on_mismatch: bool = False,
) -> FeatureBundle:
    X, y, paths, id2label_out, label2id_out = _load_features_from_joblib(
        joblib_path, label2id, sort_within_label, global_sort
    )
    path2label = _build_path2label(joblib_path)

    mismatches: List[Tuple[int, str, str, int, str]] = []
    for i, p in enumerate(paths):
        lab = path2label.get(p)
        if lab is None:
            mismatches.append((i, p, "<missing in joblib>", int(y[i]), id2label_out[int(y[i])]))
        else:
            expected = int(label2id_out[lab])
            if int(y[i]) != expected:
                mismatches.append((i, p, lab, int(y[i]), id2label_out[int(y[i])]))

    target_label_indices: List[int] = []
    target_label_ok = True
    if target_label in label2id_out:
        t_id = int(label2id_out[target_label])
        target_label_indices = [i for i, p in enumerate(paths) if path2label.get(p) == target_label]
        target_label_ok = all(int(y[i]) == t_id for i in target_label_indices)

    uniq, cnt = np.unique(y.astype(int), return_counts=True)
    class_counts = {id2label_out[int(i)]: int(c) for i, c in zip(uniq, cnt)}

    print(f"Loaded: X={X.shape}, y={y.shape}, paths={len(paths)}")
    print("Class counts:", class_counts)
    print("Mismatches:", len(mismatches))
    if target_label in label2id_out:
        print(f"{target_label!r} indices sample:", target_label_indices[:5])
        if not target_label_ok:
            print(f"Some '{target_label}' rows have mismatched ids.")
    if assert_on_mismatch and mismatches:
        raise AssertionError(f"Found {len(mismatches)} label mismatches.")

    return FeatureBundle(
        X=X, y=y, paths=paths, label2id=label2id_out, id2label=id2label_out,
        path2label=path2label, mismatches=mismatches, class_counts=class_counts,
        target_label_indices=target_label_indices, target_label_ok=target_label_ok
    )

# ================================== End of Complex feat ======================================

def preprocess_image_dog(image):
   # contrast equalization
  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  clahe_img = clahe.apply(image)

  # apply vignette on images so that edges are less emphasized for DoG computation
  clahe_img = clahe_img.astype(np.float32)/255.0
  mean_pixel_intensity = np.mean(clahe_img)
  h, w = clahe_img.shape
  center_x, center_y = w // 2, h // 2
  # Step 2: Create radial mask centered in image
  Y, X = np.ogrid[:h, :w]
  dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
  max_dist = np.sqrt(center_x**2 + center_y**2)
  mask = 1 - (dist / max_dist)
  mask = np.clip(mask, 0, 1)
  if (mean_pixel_intensity > 0.3):
    mask = mask**1.6  # steeper falloff
  elif (mean_pixel_intensity < 0.15):
    mask = mask**1.5
  # Step 3: Apply mask
  img_masked = clahe_img * mask
  return img_masked

def preprocess_image_doh(image):
  # contrast equalization
  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  clahe_img = clahe.apply(image)

  # apply vignette on images so that edges are less emphasized for DoG computation
  clahe_img = clahe_img.astype(np.float32)/255.0
  mean_pixel_intensity = np.mean(clahe_img)
  h, w = clahe_img.shape
  center_x, center_y = w // 2, h // 2
  # Step 2: Create radial mask centered in image
  Y, X = np.ogrid[:h, :w]
  dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
  max_dist = np.sqrt(center_x**2 + center_y**2)
  mask = 1 - (dist / max_dist)
  mask = np.clip(mask, 0, 1)
  if (mean_pixel_intensity > 0.3):
    mask = mask**1.6  # steeper falloff
  elif (mean_pixel_intensity < 0.15):
    mask = mask**1.5
  # Step 3: Apply mask
  img_masked = clahe_img * mask
  return img_masked

def get_features(in_imgs: Optional[np.ndarray], 
                 feat_name='canny',
                 *,
                 joblib_path: Optional[str] = None,
                 label2id: Optional[Dict[str, int]] = None,
                 sort_within_label: bool = True,
                 global_sort: bool = False,
                 target_label: str = "glioma",
                 return_bundle: bool = False):
    features = []
    if feat_name == 'canny':
        for i in tqdm(range(in_imgs.shape[0]), desc = 'Canny Edge Images'):
            image = in_imgs[i]
            img_uint8 = (image * 255).astype(np.uint8)
            edges = cv.Canny(img_uint8, threshold1=50, threshold2=150)
            edges_norm = edges/255.0
            edges_norm_flatten = edges_norm.flatten()
            features.append(edges_norm_flatten)
        features = np.array(features)

        return features

    if feat_name == 'blob_dog':
        max_features = 0
        for i in tqdm(range(in_imgs.shape[0]), desc = 'Blob Dog Images'):
            #print("Blob DoG Image:" + str(i))
            image = (in_imgs[i]*255).astype(np.uint8)
            
            img_masked = preprocess_image_dog(image)

            # Apply DoG
            large_low_sigma, large_high_sigma = 24, 40
            small_low_sigma, small_high_sigma = 20, 30
            dog_image_large_blobs = difference_of_gaussians(img_masked, large_low_sigma, large_high_sigma)
            dog_image_small_blobs = difference_of_gaussians(img_masked, small_low_sigma, small_high_sigma)
            blobs_dog_small = blob_dog(img_masked, min_sigma=small_low_sigma, max_sigma=small_high_sigma, threshold=0.078)
            blobs_dog_small[:, 2] = blobs_dog_small[:, 2] * np.sqrt(2)
            blobs_dog_large = blob_dog(img_masked, min_sigma=large_low_sigma, max_sigma=large_high_sigma, threshold=0.056)
            blobs_dog_large[:, 2] = blobs_dog_large[:, 2] * np.sqrt(2)

            # blob_dog_final = blobs_dog_large.flatten()
            # if blob_dog_final.shape[0] > max_features:
            #     max_features = blob_dog_final.shape[0]
            summary_feature = summarize_blobs(np.vstack((blobs_dog_large, blobs_dog_small)), image.shape)
            features.append(summary_feature)
            
        # pbar = tqdm(range(len(features)), desc="Padding Features")
        # for blob_index in pbar:
        #     if features[blob_index].shape[0] < max_features:
        #         features[blob_index] = np.pad(features[blob_index], pad_width=(0,max_features-features[blob_index].shape[0]))
        features = np.array(features)
        return features
    
    if feat_name == 'blob_doh':
        max_features = 0
        for i in tqdm(range(in_imgs.shape[0]), desc = 'Blob DoH images'):
            #print("DoH Image: " + str(i))
            image = (in_imgs[i]*255).astype(np.uint8)
            
            img_masked = preprocess_image_doh(image)

            # Apply DoH
            large_low_sigma, large_high_sigma = 30, 45
            small_low_sigma, small_high_sigma = 10, 30
            doh_image_large_blobs = hessian_matrix_det(img_masked, large_low_sigma)
            doh_image_small_blobs = hessian_matrix_det(img_masked, small_low_sigma)
            blobs_doh_small = blob_doh(img_masked, min_sigma=small_low_sigma, max_sigma=small_high_sigma, threshold=0.0025)
            blobs_doh_small[:, 2] = blobs_doh_small[:, 2] * np.sqrt(2)
            blobs_doh_large = blob_doh(img_masked, min_sigma=large_low_sigma, max_sigma=large_high_sigma, threshold=0.0015)
            blobs_doh_large[:, 2] = blobs_doh_large[:, 2] * np.sqrt(2)

            # blobs_doh_final = blobs_doh_large.flatten()
            # if blobs_doh_final.shape[0] > max_features:
            #     max_features = blobs_doh_final.shape[0]
            summary_feature = summarize_blobs(np.vstack((blobs_doh_large,blobs_doh_small)), image.shape)
            features.append(summary_feature)
        
        # pbar = tqdm(range(len(features)), desc="Padding Features")
        # for blob_index in pbar:
        #     if features[blob_index].shape[0] < max_features:
        #         features[blob_index] = np.pad(features[blob_index], pad_width=(0,max_features-features[blob_index].shape[0]))

        features = np.array(features)
        return features
    
    if feat_name == 'complex':
        """ ignores in_imgs, requires joblib_path 
        returns all components from dataclass
        
        ex call : bundle = get_features(None, "complex", joblib_path="/content/features.joblib", return_bundle=True)
        X, y = bundle.X, bundle.y"""
        
        if not joblib_path:
            raise ValueError("joblib_path is required when feat_name='complex'.")
        bundle = combine_load_and_validate_joblib(
            joblib_path=joblib_path,
            label2id=label2id,
            sort_within_label=sort_within_label,
            global_sort=global_sort,
            target_label=target_label,
        )
        return bundle if return_bundle else bundle.X
        

    return None

##########################
# Feature saving/loading #
##########################

def get_path(image_file_path):
    """
    Args: 
        image_file_path: full image file path
    Returns:
        portion of path up until the /Training or /Testing folder (e.g. '/Training/glioma/Tr-gl_0952.jpg')
    """
    file_name = os.path.basename(image_file_path)
    parent_dir = os.path.dirname(image_file_path)
    parent_dir_name = os.path.basename(parent_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    grandparent_dir_name = os.path.basename(grandparent_dir)
    return os.path.join("/",grandparent_dir_name, parent_dir_name, file_name)

def get_paths(image_file_paths):
    """
    Applies get_path (see above) for list of full image file paths.
    """
    paths = []
    for file_path in image_file_paths:
        paths.append(get_path(file_path))
    paths = np.array(paths).reshape(-1,1)
    return paths

def get_updated_aligned_features(aligned_features_path, new_features_df):
    """
    Reads in pandas dataframe from aligned_features_path and updates it with new_features_df. Displays result.
    (Note: does not save the updated df. This must be done separately.)
    Args:
        aligned_features_path: file path of aligned features CSV (e.g. 'aligned_training_features.csv')
        new_features_df: pandas dataframe with columns for 'filepath' (e.g. contains paths like '/Training/glioma/Tr-gl_0952.jpg') and new features to update
    Returns:
        aligned features pandas dataframe updated with new features
    """
    aligned_features_df = pd.read_csv(aligned_features_path)
    new_features_df_columns = new_features_df.columns.tolist()
    new_features_names = [col for col in new_features_df_columns if col != 'filepath']
    aligned_features_df = pd.merge(aligned_features_df, new_features_df, on='filepath')
    aligned_features_df = aligned_features_df.rename(columns={col+"_y": col for col in new_features_names})
    aligned_features_df.drop(columns=[col+'_x' for col in new_features_names], inplace=True, errors='ignore')
    display(aligned_features_df.head())
    return aligned_features_df

def clean_df(
    df: pd.DataFrame,
    label_col: str,
    feat_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int], Optional[LabelEncoder]]:

    """ clean_df, dims, enc = clean_df (
    df=df,
    label_col="target",
    feat_cols=["feat_canny", "feat_vec", "feat_dog", "feat_doh"]
    )"""

    def _to_array(x: Union[str, list, tuple, np.ndarray]) -> np.ndarray:


        if isinstance(x, (list, tuple, np.ndarray)):
            return np.asarray(x, dtype=np.float32)

        if isinstance(x, str):
            s = x.strip()
            try:
                parsed = ast.literal_eval(s)
                return np.asarray(parsed, dtype=np.float32)
            except Exception:
                pass 
            s = s.lstrip("[").rstrip("]")
            s = re.sub(r"[,'\"\[\]]", " ", s)        # drop quotes & commas
            return np.fromstring(s, sep=" ", dtype=np.float32)
        return np.asarray([x], dtype=np.float32)


    missing = [c for c in feat_cols + [label_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    cleaned = df.copy(deep=True)
    dims: Dict[str, int] = {}

    for col in feat_cols:
        arrs    = cleaned[col].apply(_to_array)
        lengths = arrs.map(len)
        if lengths.nunique() != 1:
            raise ValueError(
                f"Inconsistent vector lengths in '{col}':\n{lengths.value_counts()}"
            )
        cleaned[col] = arrs
        dims[col]    = lengths.iloc[0]

    def _to_scalar(v):
        if isinstance(v, np.ndarray):
            return v.item()
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return v

    y_series  = cleaned[label_col].apply(_to_scalar)
    y_numeric = pd.to_numeric(y_series, errors="coerce")

    numeric_ok = y_numeric.notna().all() and np.allclose(
        y_numeric, np.round(y_numeric)
    )

    encoder: Optional[LabelEncoder] = None
    if numeric_ok:
        cleaned[label_col] = y_numeric.astype(np.int32)
    else:
        encoder = LabelEncoder()
        cleaned[label_col] = (
            encoder.fit_transform(y_series.astype(str)).astype(np.int32)
        )

    return cleaned, dims, encoder

def parse_aligned_features(aligned_df, features):
    """
    Args:
        aligned_df: output pandas dataframe from clean_df()
        features: list of feature names to parse numpy arrays for
    Returns:
        dictionary mapping column names from pandas aligned_df to corresponding numpy arrays/matrices
        i.e. {"filepaths": <numpy array>, "target": <numpy array>, "feat_vec": <numpy matrix>, ...}
        
    """
    result_dict = {}
    result_dict['filepath'] = aligned_df['filepath'].to_numpy()
    result_dict['target'] = aligned_df['target'].to_numpy()
    for feature in features:
        result_dict[feature] = np.vstack(aligned_df[feature].values)
    return result_dict

##############################
# Feature dimensionality EDA #
##############################

def get_PCA(X_list, n_components=[14,14,100,100]):
    pca_list = []
    xpca_list = []
    for index, X in enumerate(X_list):
        pca = PCA(n_components=n_components[index], svd_solver="randomized", whiten=True).fit(X)
        X_pca = pca.transform(X)
        pca_list.append(pca)
        xpca_list.append(X_pca)
    return pca_list, xpca_list

def plot_PCA(X_list, n_components=[14,14,100,100], labels=['dog features', 'doh features','canny features','complex features'], colors=['r-', 'b-','g-','p-']):
    pca_list, xpca_list = get_PCA(X_list, n_components=n_components)

    plt.figure(figsize=(15,5))
    for i in range(len(X_list)):
        plt.plot(np.cumsum(pca_list[i].explained_variance_ratio_), colors[i], label=labels[i])
        plt.grid(True)
        plt.xlabel('Number of components')
        plt.ylabel('Explained Variances')
        plt.legend()
    
    plt.show()

def get_tsne(X_list, n_components=2):
  xtsne_list = []
  for X in X_list:
    tsne = TSNE(n_components=n_components, random_state=0)
    X_tsne = tsne.fit_transform(X)
    xtsne_list.append(X_tsne)
  return xtsne_list


def plot_classes(X, y, title):
    """
    Plots 2D or 3D scatter plot of X, color-differentiated according to y.
    """
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)

    fig = plt.figure(figsize=(10,10))
    if X.shape[1] > 2:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    # color code each cluster (person ID)
    colormap = plt.cm.tab20
    colorst = [colormap(i) for i in np.linspace(0, 1.0, len(np.unique(y_enc)))]

    # project the features into 2 or 3 dimensions
    for k in np.unique(y_enc):
        label = le.classes_[k]
        if X.shape[1] > 2:
            ax.scatter(X[y_enc==k, 0], X[y_enc==k, 1], X[y_enc==k, 2], alpha=0.5, facecolors=colorst[k], label=label)
        else:
            ax.scatter(X[y_enc==k, 0], X[y_enc==k, 1], alpha=0.5, facecolors=colorst[k], label=label)

    ax.set_title(title)
    ax.legend()


##########################
# Model training/testing #
##########################


def train_model(X_train, y_train, classes, model_type='logistic', feature='canny'):

    if model_type =='logistic':
        params = {
        'solver':['lbfgs','newton-cg'],
        'max_iter':[1000, 1500]
        }

        logistic_model = LogisticRegression()
        model = GridSearchCV(logistic_model, params, scoring='accuracy', cv=5, verbose=1)
        start_time = time.perf_counter()
        model.fit(X_train,y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

    elif model_type =='svm':
        svm_param_grid = {
            'C': [0.1],
            'gamma': [0.001,0.1,1],
            'kernel': ['rbf','linear']
        }
        svc = svm.SVC(probability=True)
        model = GridSearchCV(svc, svm_param_grid, scoring='accuracy', cv=5, verbose=2)
        start_time = time.perf_counter()
        model.fit(X_train,y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

    elif model_type =='gbm':
        gbm_param_grid = {'loss':['log_loss'],
                'learning_rate':[0.1],
                'n_estimators':[80,100,120],
                'max_depth':[2,3,4]}
        
        gbm = GradientBoostingClassifier()
        model = GridSearchCV(gbm, gbm_param_grid, scoring='accuracy',cv=5, verbose=1)
        start_time = time.perf_counter()
        model.fit(X_train,y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

        print(str(model.best_params_))
        print(str(model.score(X_train,y_train)))
        
    elif model_type =='xgboost':
        params = {'learning_rate':[0.05, 0.1,0.15],
                'n_estimators':[80,90,100,110],
                'max_depth':[2,3,4]}

        xgb_model = XGBClassifier()

        # Fit model
        # Use GridSearch to find the best parameters
        model = GridSearchCV(xgb_model, params, cv=5, scoring='accuracy',verbose=1)
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

    elif model_type == 'rf':
        params = {
        'n_estimators':[80,90,100,110],
        'max_depth':[2,3,4],
        'max_features':['sqrt','log2']
        }

        rf_model = RandomForestClassifier()

        model = GridSearchCV(rf_model, params, cv=5, scoring='accuracy',verbose=1)
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

    elif model_type == 'lda':
        params = {
        'solver':['svd','lsqr','eigen'],
        'shrinkage':['auto']
        }

        lda_model = LinearDiscriminantAnalysis()

        model = GridSearchCV(lda_model, params, cv=5, scoring='accuracy',verbose=1)
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)

    elif model_type == 'qda':
        params = {
        'reg_param':np.linspace(start=0.5,stop=1,num=3)
        }

        qda_model = QuadraticDiscriminantAnalysis()

        model = GridSearchCV(qda_model, params, cv=5, scoring='accuracy',verbose=2)
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        y_model_pred = model.best_estimator_.predict(X_train)
        y_model_pred_proba = model.best_estimator_.predict_proba(X_train)


    # Generate Confusion Matrix for Logistic Regression
    confusion_matrix = metrics.confusion_matrix(y_train, y_model_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classes)
    cm_display.plot()
    plt.title('Training CM Feature:' + str(feature) + ' Model: ' + str(model_type))
    save_dir = 'results_images'
    save_path = os.path.join(save_dir, str(feature) + '_' + str(model_type)+'_training_confusion_matrix.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    # # Handle classifier output format
    # if isinstance(y_model_pred_proba, list):
    #     # Convert list of class-wise predictions to proper array
    y_train_binarized = label_binarize(y_train, classes=[0,1,2,3])

    if isinstance(y_model_pred_proba, list):
        y_model_pred_proba = np.stack([score[:, 1] for score in y_model_pred_proba], axis=1)
    print(y_model_pred_proba.shape)

    # Generate ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_train_binarized[:, i], y_model_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red','brown']
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Feature:' + str(feature) + ' Model: ' + str(model_type))
    plt.legend(loc='lower right')
    plt.grid(True)
    save_dir = 'results_images'
    save_path = os.path.join(save_dir, str(feature) + '_' + str(model_type)+'_training_roc.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    accuracy_score = metrics.accuracy_score(y_train, y_model_pred)
    macro_precision = metrics.precision_score(y_train, y_model_pred,average ='macro')
    macro_recall = metrics.recall_score(y_train, y_model_pred,average='macro')
    macro_f1 = metrics.f1_score(y_train, y_model_pred,average='macro')
    micro_precision = metrics.precision_score(y_train, y_model_pred,average='micro')
    micro_recall = metrics.recall_score(y_train, y_model_pred,average='micro')
    micro_f1 = metrics.f1_score(y_train, y_model_pred,average='micro')
    # fpr, tpr, thresholds = metrics.roc_curve(y_train, y_logistic_pred_dog)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    #                                   name='dog estimator')

    # print("==================" + str(feature) + " TRAINING METRICS ===================")
    # print("Accuracy Score: " + str(accuracy_score))
    # print("Macro Precision: " + str(macro_precision))
    # print("Macro Recall: " + str(macro_recall))
    # print("Macro F1: " + str(macro_f1))
    # print("Micro Precision: " + str(micro_precision))
    # print("Micro Recall: " + str(micro_recall))
    # print("Micro F1: " + str(micro_f1))

    results_dict = {}
    results_dict['feature'] = feature
    results_dict['model_type'] = model_type
    results_dict['accuracy_score'] = accuracy_score
    results_dict['macro_precision'] = macro_precision
    results_dict['macro_recall'] = macro_recall
    results_dict['macro_f1'] = macro_f1
    results_dict['micro_precision'] = micro_precision
    results_dict['micro_recall'] = micro_recall
    results_dict['micro_f1'] = micro_f1
    results_dict['training_time'] = elapsed_time


    return model.best_estimator_, results_dict

def ensemble_model(
    model_specs: Dict[str, Tuple[BaseEstimator, np.ndarray, np.ndarray]],
    X_test_dict: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
    meta_model: BaseEstimator = None
) -> Dict[str, Any]:
    """
    Train base learners on (train, val) matrices in `model_specs`,
    evaluate on a test-set per model in `X_test_dict`, then stack via a meta-classifier.

    Parameters:
    model_specs  : dict
        name -> (estimator, X_train, X_val)
    X_test_dict  : dict
        name -> X_test   (must have the same keys as model_specs)
    y_*          : label vectors, one per split
    verbose      : print per-model val accuracy and final test accuracy

    Returns:
    dict with trained objects, predictions, and stacked matrices.
    
    
    Example call:
    y = clean_df["target"].to_numpy()
    X_canny_all = np.vstack(clean_df["feat_canny"].values)   # (n_samples, n_canny_feats)
    X_vec_all   = np.vstack(clean_df["feat_vec"].values)     # (n_samples, n_vec_feats)

    X_canny_train, X_canny_temp, X_vec_train, X_vec_temp, y_train, y_temp = train_test_split(
        X_canny_all, X_vec_all, y,
        test_size=0.20, random_state=42, stratify=y
    )
    X_canny_val, X_canny_test, X_vec_val, X_vec_test, y_val, y_test = train_test_split(
        X_canny_temp, X_vec_temp, y_temp,
        test_size=0.50, random_state=42, stratify=y_temp
    )

    model_specs = {
        "canny": (
            LogisticRegression(max_iter=1000),   # estimator
            X_canny_train,                       # train matrix
            X_canny_val                          # val matrix
        ),
        "vec": (
            SVC(kernel="linear", probability=True, max_iter=1000, random_state=42),
            X_vec_train,
            X_vec_val
        )
    }
    # Separate dict holding the per-model test matrices
    X_test_dict = {
        "canny": X_canny_test,
        "vec":   X_vec_test
    }
    results = ensemble_model(
        model_specs=model_specs,
        X_test_dict=X_test_dict,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        meta_model=LogisticRegression(max_iter=1000)
    )
    print("Ensemble test accuracy:", (results["y_pred"] == y_test).mean())

    
    """

    base_models, val_columns, test_columns = {}, [], []

    # Fit base models & get probs 
    for name, (model, X_tr, X_val) in model_specs.items():
        model.fit(X_tr, y_train)
        base_models[name] = model

        val_prob  = model.predict_proba(X_val)
        test_prob = model.predict_proba(X_test_dict[name])

        val_columns.append(val_prob)
        test_columns.append(test_prob)

        acc = accuracy_score(y_val, np.argmax(val_prob, axis=1))
        print(f"[{name}] val-acc: {acc:0.4f}")

    # Stack prob matrices
    val_stack  = np.hstack(val_columns)
    test_stack = np.hstack(test_columns)

    # Train meta-model on stacked val outputs
    meta_model.fit(val_stack, y_val)

    # Final test-set preds
    y_pred  = meta_model.predict(test_stack)
    y_score = meta_model.predict_proba(test_stack)


    acc = accuracy_score(y_test, y_pred)
    print(f"[Ensemble] test-acc: {acc:0.4f}")

    return {
        "meta_model": meta_model,
        "base_models": base_models,
        "y_pred": y_pred,
        "y_score": y_score,
        "val_stack": val_stack,
        "test_stack": test_stack,
    }

class FeatureVectorEnsemble:
    def __init__(self, models, feature_sets, use_proba=False):
        self.models = models
        self.feature_sets = feature_sets
        self.use_proba = use_proba  # True = probability averaging, False = majority vote

    def predict(self, idx):
        """
        Predicts label for image at index `idx` using the ensemble.
        """
        preds = []

        for model, X in zip(self.models, self.feature_sets):
            x = X[idx].reshape(1, -1)
            if self.use_proba:
                proba = model.predict_proba(x)
                preds.append(proba)
            else:
                pred = model.predict(x)[0]
                preds.append(pred)

        if self.use_proba:
            avg_proba = np.mean(preds, axis=0)
            return np.argmax(avg_proba)
        else:
            majority_vote, _ = mode(preds)
            return majority_vote[0]
        

def test_model(model, X_test_feature, Y_test, classes, model_type='logistic', feature='canny'):
    start_time = time.perf_counter()
    Y_test_predict = model.predict(X_test_feature)
    Y_test_predict_proba = model.predict_proba(X_test_feature)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_test_predict)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classes)
    cm_display.plot()
    plt.title("CM Testing Model: " + str(model_type) + " Feature: " + str(feature))
    save_dir = 'results_images'
    save_path = os.path.join(save_dir, str(feature) + '_' + str(model_type)+'_testing_confusion_matrix.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.show()


    y_test_binarized = label_binarize(Y_test, classes=[0,1,2,3])

    if isinstance(Y_test_predict_proba, list):
        Y_test_predict_proba = np.stack([score[:, 1] for score in Y_test_predict_proba], axis=1)
    print(Y_test_predict_proba.shape)

    # Generate ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], Y_test_predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red','brown']
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Testing Curve Feature: ' + str(feature) + ' Model: ' + str(model_type))
    plt.legend(loc='lower right')
    plt.grid(True)
    save_dir = 'results_images'
    save_path = os.path.join(save_dir, str(feature) + '_' + str(model_type)+'_testing_roc.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)

    plt.show()



    accuracy_score = metrics.accuracy_score(Y_test, Y_test_predict)
    macro_precision = metrics.precision_score(Y_test, Y_test_predict,average ='macro')
    macro_recall = metrics.recall_score(Y_test, Y_test_predict,average='macro')
    macro_f1 = metrics.f1_score(Y_test, Y_test_predict,average='macro')
    micro_precision = metrics.precision_score(Y_test, Y_test_predict,average='micro')
    micro_recall = metrics.recall_score(Y_test, Y_test_predict,average='micro')
    micro_f1 = metrics.f1_score(Y_test, Y_test_predict,average='micro')

    results_dict = {}
    results_dict['feature'] = feature
    results_dict['model_type'] = model_type
    results_dict['accuracy_score'] = accuracy_score
    results_dict['macro_precision'] = macro_precision
    results_dict['macro_recall'] = macro_recall
    results_dict['macro_f1'] = macro_f1
    results_dict['micro_precision'] = micro_precision
    results_dict['micro_recall'] = micro_recall
    results_dict['micro_f1'] = micro_f1
    results_dict['inference_time'] = elapsed_time

    return results_dict

########################
# Model saving/loading #
########################

def save_models(feature_model_dict):
    """
    Saves sklearn model(s) from feature_model_dict to path(s) with specified model_path_prefix.
    Args:
        feature_model_dict: dictionary of feature types to sklearn models. e.g. {'canny', canny_logistic_model, 'complex': complex_logistic_model}
        model_path_prefix: file path prefix to identify model(s) being saved. e.g. "logistic_model". 
    """
    for feature in feature_model_dict.keys():
        model = feature_model_dict[feature]
        os.makedirs("models", exist_ok=True)
        path = f"models/{feature}.joblib"
        joblib.dump(model, path)
        print(f"saved model={model} for feature={feature} to path={path}")


def load_models(feature_list, model_path_prefix):
    """
    Loads sklearn model(s) to feature_model_dict from path(s) with specified model_path_prefix.
    Args:
        feature_list: list of feature type(s) to load model(s) for
        model_path_prefix: file path prefix to identify model(s) being loaded. e.g. "logistic_model".
    Returns:
        feature_model_dict: dictionary of feature types to sklearn models. e.g. {'canny': canny_logistic_model, 'complex': complex_logistic_model}
    """
    feature_model_dict = {}
    for feature in feature_list:
        path = f"models/{model_path_prefix}_{feature}.joblib"
        feature_model_dict[feature] = joblib.load(path)
        print(f"loaded model={feature_model_dict[feature]} for feature={feature} from path={path}")
    return feature_model_dict