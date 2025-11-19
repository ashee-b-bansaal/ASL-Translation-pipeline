import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import cv2
import os
import re
import logging
import random
from copy import deepcopy
from math import ceil
import pickle

import pandas as pd
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from collections import Counter

from sklearn.model_selection import StratifiedKFold

def say_hello(name):
    return f"Hello, {name}!"


def load_acoustic_timestamps(timestamp_file_path, verbose=False):
    """
    Load acoustic data timestamps from pickle file.
    Can handle pandas DataFrames (like in create_sidebyside_timestamp1.py) or simple arrays.
    
    Args:
        timestamp_file_path: Path to the timestamp pickle file
        verbose: If True, print detailed error messages (for debugging)
    
    Returns:
        timestamps: Array of timestamps for acoustic frames, or None if file doesn't exist/error
    """
    if not os.path.exists(timestamp_file_path):
        return None
    
    error_msg = None
    timestamp_data = None
    
    try:
        # Try using pandas first (like in create_sidebyside_timestamp1.py)
        try:
            timestamp_df = pd.read_pickle(timestamp_file_path)
            # If it's a DataFrame, extract timestamps column
            if isinstance(timestamp_df, pd.DataFrame):
                # Look for timestamp column (most common case)
                if 'timestamp' in timestamp_df.columns:
                    result = timestamp_df['timestamp'].values
                    if verbose:
                        print(f"      Loaded from DataFrame['timestamp']: shape={result.shape}")
                    return result
                elif 'acoustic_timestamp' in timestamp_df.columns:
                    result = timestamp_df['acoustic_timestamp'].values
                    if verbose:
                        print(f"      Loaded from DataFrame['acoustic_timestamp']: shape={result.shape}")
                    return result
                elif 'time' in timestamp_df.columns:
                    result = timestamp_df['time'].values
                    if verbose:
                        print(f"      Loaded from DataFrame['time']: shape={result.shape}")
                    return result
                else:
                    # Try first numeric column
                    for col in timestamp_df.columns:
                        if timestamp_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                            result = timestamp_df[col].values
                            if verbose:
                                print(f"      Loaded from DataFrame column '{col}': shape={result.shape}")
                            return result
                    # If no numeric column found, convert first column
                    result = timestamp_df.iloc[:, 0].values
                    if verbose:
                        print(f"      Loaded from DataFrame first column: shape={result.shape}, columns={list(timestamp_df.columns)}")
                    return result
            # If it's a Series, convert to array
            elif isinstance(timestamp_df, pd.Series):
                result = timestamp_df.values
                if verbose:
                    print(f"      Loaded from Series: shape={result.shape}")
                return result
            else:
                error_msg = f"Pandas loaded unexpected type: {type(timestamp_df)}"
                if verbose:
                    print(f"      {error_msg}")
        except Exception as e:
            # Check if it's a protocol 5 error
            if "pickle protocol" in str(e) or "unsupported pickle protocol" in str(e):
                # Protocol 5 detected - use pickle5.load() directly
                error_msg = f"Pandas read_pickle failed with protocol 5: {str(e)}"
                if verbose:
                    print(f"      {error_msg}, trying pickle5.load() directly...")
                try:
                    import pickle5
                    with open(timestamp_file_path, 'rb') as f:
                        timestamp_data = pickle5.load(f)
                    # Process the loaded DataFrame/Series immediately
                    if isinstance(timestamp_data, pd.DataFrame):
                        if 'timestamp' in timestamp_data.columns:
                            result = timestamp_data['timestamp'].values
                            if verbose:
                                print(f"      Loaded with pickle5.load() from DataFrame['timestamp']: shape={result.shape}")
                            return result
                        elif 'acoustic_timestamp' in timestamp_data.columns:
                            result = timestamp_data['acoustic_timestamp'].values
                            if verbose:
                                print(f"      Loaded with pickle5.load() from DataFrame['acoustic_timestamp']: shape={result.shape}")
                            return result
                        elif 'time' in timestamp_data.columns:
                            result = timestamp_data['time'].values
                            if verbose:
                                print(f"      Loaded with pickle5.load() from DataFrame['time']: shape={result.shape}")
                            return result
                        else:
                            for col in timestamp_data.columns:
                                if timestamp_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                                    result = timestamp_data[col].values
                                    if verbose:
                                        print(f"      Loaded with pickle5.load() from DataFrame column '{col}': shape={result.shape}")
                                    return result
                            result = timestamp_data.iloc[:, 0].values
                            if verbose:
                                print(f"      Loaded with pickle5.load() from DataFrame first column: shape={result.shape}, columns={list(timestamp_data.columns)}")
                            return result
                    elif isinstance(timestamp_data, pd.Series):
                        result = timestamp_data.values
                        if verbose:
                            print(f"      Loaded with pickle5.load() from Series: shape={result.shape}")
                        return result
                    else:
                        # Unknown type, fall through to bottom processing logic
                        pass
                except ImportError:
                    if verbose:
                        print(f"      pickle5 module not available, cannot load protocol 5 pickle")
                    return None
                except (AttributeError, ImportError, TypeError) as e2:
                    # pandas version incompatibility - the pickle file was created with newer pandas
                    if "_unpickle_block" in str(e2) or "pandas._libs.internals" in str(e2):
                        if verbose:
                            print(f"      pickle5.load() failed: pandas version incompatibility")
                            print(f"      Attempting alternative: try loading with dill (if available)...")
                        # Try using dill as alternative (sometimes handles version mismatches better)
                        try:
                            import dill
                            with open(timestamp_file_path, 'rb') as f:
                                timestamp_data = dill.load(f)
                            if isinstance(timestamp_data, pd.DataFrame):
                                if 'timestamp' in timestamp_data.columns:
                                    result = timestamp_data['timestamp'].values
                                    if verbose:
                                        print(f"      Successfully loaded with dill from DataFrame['timestamp']: shape={result.shape}")
                                    return result
                                elif 'acoustic_timestamp' in timestamp_data.columns:
                                    result = timestamp_data['acoustic_timestamp'].values
                                    if verbose:
                                        print(f"      Successfully loaded with dill from DataFrame['acoustic_timestamp']: shape={result.shape}")
                                    return result
                                elif 'time' in timestamp_data.columns:
                                    result = timestamp_data['time'].values
                                    if verbose:
                                        print(f"      Successfully loaded with dill from DataFrame['time']: shape={result.shape}")
                                    return result
                                else:
                                    result = timestamp_data.iloc[:, 0].values
                                    if verbose:
                                        print(f"      Successfully loaded with dill from DataFrame first column: shape={result.shape}")
                                    return result
                            elif isinstance(timestamp_data, pd.Series):
                                result = timestamp_data.values
                                if verbose:
                                    print(f"      Successfully loaded with dill from Series: shape={result.shape}")
                                return result
                        except ImportError:
                            if verbose:
                                print(f"      dill not available. pandas version incompatibility detected.")
                                print(f"      Options: 1) Install dill: pip install dill, 2) Upgrade pandas, 3) Re-save timestamps, or 4) Use frame-based alignment")
                        except Exception as e3:
                            if verbose:
                                print(f"      dill.load() also failed: {str(e3)}")
                                print(f"      pandas version incompatibility - cannot load timestamp file.")
                                print(f"      Falling back to frame-based alignment for this sample.")
                        return None
                    else:
                        if verbose:
                            print(f"      pickle5.load() failed: {str(e2)}")
                    return None
                except Exception as e2:
                    if verbose:
                        print(f"      pickle5.load() failed: {str(e2)}")
                    return None
            else:
                # Other pandas error - try standard pickle
                error_msg = f"Pandas read_pickle failed: {str(e)}"
                if verbose:
                    print(f"      {error_msg}, trying standard pickle...")
                # Continue to standard pickle loading
        
        # Try standard pickle loading (if not already loaded with pickle5)
        if timestamp_data is None:
            try:
                with open(timestamp_file_path, 'rb') as f:
                    timestamp_data = pickle.load(f)
            except (ValueError, TypeError) as pe:
                # Standard pickle also failed, but we've already tried pickle5 if it was protocol 5
                raise pe
    except (ValueError, TypeError) as e:
        if "pickle protocol" in str(e) or "unsupported pickle protocol" in str(e):
            error_msg = f"Pickle protocol error: {str(e)}"
            if verbose:
                print(f"      {error_msg}, trying pickle5.load() directly...")
            # Try using pickle5.load() directly (pd.read_pickle doesn't work with pickle5 for protocol 5)
            try:
                import pickle5
                # Use pickle5.load() directly instead of pd.read_pickle()
                with open(timestamp_file_path, 'rb') as f:
                    timestamp_data = pickle5.load(f)
                # Now process the loaded data (should be a DataFrame/Series)
                if isinstance(timestamp_data, pd.DataFrame):
                    if 'timestamp' in timestamp_data.columns:
                        result = timestamp_data['timestamp'].values
                        if verbose:
                            print(f"      Loaded with pickle5.load() from DataFrame['timestamp']: shape={result.shape}")
                        return result
                    elif 'acoustic_timestamp' in timestamp_data.columns:
                        result = timestamp_data['acoustic_timestamp'].values
                        if verbose:
                            print(f"      Loaded with pickle5.load() from DataFrame['acoustic_timestamp']: shape={result.shape}")
                        return result
                    elif 'time' in timestamp_data.columns:
                        result = timestamp_data['time'].values
                        if verbose:
                            print(f"      Loaded with pickle5.load() from DataFrame['time']: shape={result.shape}")
                        return result
                    else:
                        for col in timestamp_data.columns:
                            if timestamp_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                                result = timestamp_data[col].values
                                if verbose:
                                    print(f"      Loaded with pickle5.load() from DataFrame column '{col}': shape={result.shape}")
                                return result
                        result = timestamp_data.iloc[:, 0].values
                        if verbose:
                            print(f"      Loaded with pickle5.load() from DataFrame first column: shape={result.shape}, columns={list(timestamp_data.columns)}")
                        return result
                elif isinstance(timestamp_data, pd.Series):
                    result = timestamp_data.values
                    if verbose:
                        print(f"      Loaded with pickle5.load() from Series: shape={result.shape}")
                    return result
                else:
                    error_msg = f"pickle5.load() returned unexpected type: {type(timestamp_data)}"
                    if verbose:
                        print(f"      {error_msg}")
            except ImportError:
                if verbose:
                    print(f"      pickle5 module not available, cannot load protocol 5 pickle")
                return None
            except Exception as e2:
                error_msg = f"pickle5.load() failed: {str(e2)}"
                if verbose:
                    print(f"      {error_msg}")
                return None
        else:
            error_msg = f"Pickle ValueError/TypeError: {str(e)}"
            if verbose:
                print(f"      {error_msg}")
            return None
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if verbose:
            print(f"      {error_msg}")
        return None
    
    # Handle different possible formats (for non-DataFrame pickles)
    if timestamp_data is not None:
        if isinstance(timestamp_data, pd.DataFrame):
            if 'timestamp' in timestamp_data.columns:
                result = timestamp_data['timestamp'].values
                if verbose:
                    print(f"      Loaded from pickle DataFrame['timestamp']: shape={result.shape}")
                return result
            elif 'acoustic_timestamp' in timestamp_data.columns:
                result = timestamp_data['acoustic_timestamp'].values
                if verbose:
                    print(f"      Loaded from pickle DataFrame['acoustic_timestamp']: shape={result.shape}")
                return result
            elif 'time' in timestamp_data.columns:
                result = timestamp_data['time'].values
                if verbose:
                    print(f"      Loaded from pickle DataFrame['time']: shape={result.shape}")
                return result
            else:
                result = timestamp_data.iloc[:, 0].values
                if verbose:
                    print(f"      Loaded from pickle DataFrame first column: shape={result.shape}, columns={list(timestamp_data.columns)}")
                return result
        elif isinstance(timestamp_data, pd.Series):
            result = timestamp_data.values
            if verbose:
                print(f"      Loaded from pickle Series: shape={result.shape}")
            return result
        elif isinstance(timestamp_data, np.ndarray):
            result = timestamp_data
            if verbose:
                print(f"      Loaded from pickle numpy array: shape={result.shape}")
            return result
        elif isinstance(timestamp_data, dict):
            # Try common keys
            for key in ['timestamps', 'time', 'timestamp', 'acoustic_timestamps']:
                if key in timestamp_data:
                    val = timestamp_data[key]
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        result = val.values if isinstance(val, pd.Series) else val.iloc[:, 0].values
                        if verbose:
                            print(f"      Loaded from pickle dict['{key}']: shape={result.shape}")
                        return result
                    result = np.array(val)
                    if verbose:
                        print(f"      Loaded from pickle dict['{key}']: shape={result.shape}")
                    return result
            # If no standard key, try first array-like value
            for value in timestamp_data.values():
                if isinstance(value, pd.DataFrame):
                    result = value.iloc[:, 0].values
                    if verbose:
                        print(f"      Loaded from pickle dict first DataFrame value: shape={result.shape}")
                    return result
                elif isinstance(value, pd.Series):
                    result = value.values
                    if verbose:
                        print(f"      Loaded from pickle dict first Series value: shape={result.shape}")
                    return result
                elif isinstance(value, (np.ndarray, list)):
                    result = np.array(value)
                    if verbose:
                        print(f"      Loaded from pickle dict first array value: shape={result.shape}")
                    return result
        elif isinstance(timestamp_data, list):
            result = np.array(timestamp_data)
            if verbose:
                print(f"      Loaded from pickle list: shape={result.shape}")
            return result
        else:
            error_msg = f"Unsupported timestamp data type: {type(timestamp_data)}"
            if verbose:
                print(f"      {error_msg}")
    
    return None


def align_imu_to_acoustic_timestamps(imu_data, imu_timestamps, acoustic_timestamps, target_length):
    """
    Align IMU data to acoustic timestamps using interpolation.
    
    Args:
        imu_data: IMU data [N_imu, 3]
        imu_timestamps: IMU timestamps [N_imu]
        acoustic_timestamps: Acoustic timestamps [N_acoustic] 
        target_length: Target length for output (frame count after removing bad frames)
    
    Returns:
        aligned_imu_data: Aligned IMU data [target_length, 3]
    """
    if acoustic_timestamps is None or len(acoustic_timestamps) == 0:
        # Fall back to simple upsampling
        _, upsampled_imu = upsample_imu_data(imu_timestamps, imu_data, target_length)
        return upsampled_imu
    
    # Find overlapping time range
    imu_time_min, imu_time_max = imu_timestamps.min(), imu_timestamps.max()
    acoustic_time_min, acoustic_time_max = acoustic_timestamps.min(), acoustic_timestamps.max()
    
    common_start = max(imu_time_min, acoustic_time_min)
    common_end = min(imu_time_max, acoustic_time_max)
    
    if common_start >= common_end:
        # No overlap, use simple upsampling
        _, upsampled_imu = upsample_imu_data(imu_timestamps, imu_data, target_length)
        return upsampled_imu
    
    # Filter IMU data to overlapping range
    imu_mask = (imu_timestamps >= common_start) & (imu_timestamps <= common_end)
    filtered_imu = imu_data[imu_mask]
    filtered_imu_time = imu_timestamps[imu_mask]
    
    if len(filtered_imu_time) == 0:
        # No overlap, use simple upsampling
        _, upsampled_imu = upsample_imu_data(imu_timestamps, imu_data, target_length)
        return upsampled_imu
    
    # Filter acoustic timestamps to overlapping range
    acoustic_mask = (acoustic_timestamps >= common_start) & (acoustic_timestamps <= common_end)
    valid_acoustic_times = acoustic_timestamps[acoustic_mask]
    
    if len(valid_acoustic_times) == 0:
        # No overlap, use simple upsampling
        _, upsampled_imu = upsample_imu_data(imu_timestamps, imu_data, target_length)
        return upsampled_imu
    
    # Interpolate IMU to acoustic timestamps
    try:
        interp_functions = [CubicSpline(filtered_imu_time, filtered_imu[:, i]) 
                          for i in range(3)]
        upsampled_imu = np.column_stack([f(valid_acoustic_times) for f in interp_functions])
        
        # Create full length array
        full_imu = np.zeros((target_length, 3))
        valid_idx = np.where(acoustic_mask)[0]
        valid_count = min(len(valid_idx), target_length, len(upsampled_imu))
        
        if valid_count > 0:
            full_imu[:valid_count] = upsampled_imu[:valid_count]
        
        return full_imu
    except Exception:
        # If interpolation fails, use simple upsampling
        _, upsampled_imu = upsample_imu_data(imu_timestamps, imu_data, target_length)
        return upsampled_imu


def upsample_imu_data(time, imu_data, target_num_samples):
    """
    Upsample IMU data to a target number of samples.

    Parameters:
    - time: 1D array, timestamps of the original IMU data.
    - imu_data: 2D array, IMU data (e.g., acceleration, angular velocity).
    - target_num_samples: desired number of samples after upsampling.

    Returns:
    - upsampled_time: 1D array, timestamps of the upsampled data.
    - upsampled_imu_data: 2D array, upsampled IMU data.
    """
    # Ensure time values are strictly increasing and remove duplicates
    unique_time, unique_idx = np.unique(time, return_index=True)
    sorted_idx = np.argsort(unique_time)
    unique_time = unique_time[sorted_idx]
    unique_idx = unique_idx[sorted_idx]

    # Sort imu_data based on unique_time
    sorted_imu_data = imu_data[unique_idx]

    # Create an interpolation function for each dimension of the IMU data
    interp_functions = [CubicSpline(unique_time, sorted_imu_data[:, i]) for i in range(sorted_imu_data.shape[1])]
    #interp_functions = [CubicSpline(unique_time, sorted_imu_data[:, i]) for i in range(sorted_imu_data.shape[1])]

    # Create upsampled time array
    upsampled_time = np.linspace(unique_time[0], unique_time[-1], target_num_samples)

    # Interpolate IMU data at upsampled time points
    upsampled_imu_data = np.column_stack([f(upsampled_time) for f in interp_functions])

    return upsampled_time, upsampled_imu_data

def normalize_imu_data(upsampled_imu_data):
    """
    Normalize upsampled IMU data.

    Parameters:
    - upsampled_imu_data: 2D array, upsampled IMU data.

    Returns:
    - normalized_imu_data: 2D array, normalized IMU data.
    - means: 1D array, means of each axis before normalization.
    - stds: 1D array, standard deviations of each axis before normalization.
    """
    means = np.mean(upsampled_imu_data, axis=0)
    stds = np.std(upsampled_imu_data, axis=0)

    normalized_imu_data = (upsampled_imu_data - means) / stds

    return normalized_imu_data, means, stds

def normal_0_1(imu_data):
    # Normalize each column between 0 and 1
    imu_min = imu_data.min(axis=0)  # Minimum value for each column
    imu_max = imu_data.max(axis=0)  # Maximum value for each column

    imu_normalized = (imu_data - imu_min) / (imu_max - imu_min)
    return imu_normalized
    
def plot_profiles(profiles, max_val=None, min_val=None):
    max_h = 0       # red
    min_h = 120     # blue
    if not max_val:
        max_val = np.max(profiles)
    if not min_val:
        min_val = np.min(profiles)
    #print(max_val, min_val)
    heat_map_val = np.clip(profiles, min_val, max_val)
    heat_map = np.zeros(
        (heat_map_val.shape[0], heat_map_val.shape[1], 3), dtype=np.uint8)
    # print(heat_map_val.shape)
    heat_map[:, :, 0] = heat_map_val / \
        (max_val + 1e-6) * (max_h - min_h) + min_h
    heat_map[:, :, 1] = np.ones(heat_map_val.shape) * 255
    heat_map[:, :, 2] = np.ones(heat_map_val.shape) * 255
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_HSV2BGR)
    return heat_map


def plot_profiles_split_channels(profiles, n_channels, maxval=None, minval=None):
    channel_width = profiles.shape[0] // n_channels

    profiles_img = np.zeros(
        ((channel_width + 5) * n_channels, profiles.shape[1], 3))

    for n in range(n_channels):
        channel_profiles = profiles[n * channel_width: (n + 1) * channel_width]
        profiles_img[n * (channel_width + 5): (n + 1) * (channel_width + 5) - 5,
                     :, :] = plot_profiles(channel_profiles, maxval, minval)

    return profiles_img

def vis(input):
    img_input = input.copy()
    diff_profiles_img = plot_profiles_split_channels(img_input.T, 1, 20000000, -20000000)
    #profiles11 = plot_profiles(profiles1, 20000000, -20000000)
    acous_npy_img = cv2.cvtColor(np.float32(diff_profiles_img), cv2.COLOR_BGR2RGB)
    plt.imshow(acous_npy_img.astype(np.uint16), aspect = 'auto')
    plt.savefig('./test111.png')
    print('saved!')



def read_from_folder(session_num, data_path, is_train=False, is_shuffle = True, seed_num =43, use_timestamps=True, gap_threshold_factor=None, prefer_legacy_dirs=False):
    """
    Read data from session folder with optional timestamp-based alignment.
    
    Args:
        session_num: Session number
        data_path: Base path to session data
        is_train: Whether this is training data
        is_shuffle: Whether to shuffle file order
        seed_num: Random seed for shuffling
        use_timestamps: Whether to use acoustic timestamps for alignment (default: True)
        gap_threshold_factor: Gap threshold factor (e.g., 1.0, 1.2, 1.5, 1.7, 2.0). If provided, uses diff_new_{factor} and non_diff_new_{factor} folders
    """
    file_path = data_path + '%s'%str(session_num)
    
    # Determine which diff directory to use
    if prefer_legacy_dirs:
        file_echo_diff = os.path.join(file_path, 'acoustic/diff')
        diff_new_path = os.path.join(file_path, 'acoustic/diff_new')
        diff_old_path = file_echo_diff
    elif gap_threshold_factor is not None:
        # Use thresholded version: diff_new_{factor}
        diff_thresholded_path = os.path.join(file_path, f'acoustic/diff_new_{gap_threshold_factor}')
        diff_old_path = os.path.join(file_path, 'acoustic/diff')
        file_echo_diff = diff_thresholded_path if os.path.exists(diff_thresholded_path) else diff_old_path
    else:
        # Prefer new directories if available, fallback to legacy structure
        diff_new_path = os.path.join(file_path, 'acoustic/diff_new')
        diff_old_path = os.path.join(file_path, 'acoustic/diff')
        file_echo_diff = diff_new_path if os.path.exists(diff_new_path) else diff_old_path
    
    if not os.path.exists(file_echo_diff):
        if gap_threshold_factor is not None:
            raise FileNotFoundError(f"Diff directory not found: {os.path.join(file_path, f'acoustic/diff_new_{gap_threshold_factor}')} or {diff_old_path}")
        else:
            raise FileNotFoundError(f"Neither new nor old diff directory found: {diff_new_path} or {diff_old_path}")

    # Determine which non-diff directory to use
    if prefer_legacy_dirs:
        file_echo_org = os.path.join(file_path, 'acoustic/non_diff')
    elif gap_threshold_factor is not None:
        # Use thresholded version: non_diff_new_{factor}
        org_thresholded_path = os.path.join(file_path, f'acoustic/non_diff_new_{gap_threshold_factor}')
        org_old_path = os.path.join(file_path, 'acoustic/non_diff')
        file_echo_org = org_thresholded_path if os.path.exists(org_thresholded_path) else org_old_path
    else:
        org_new_path = os.path.join(file_path, 'acoustic/non_diff_new')
        org_old_path = os.path.join(file_path, 'acoustic/non_diff')
        file_echo_org = org_new_path if os.path.exists(org_new_path) else org_old_path
    file_timestamps = file_path +  "/" + 'acoustic/timestamp'  # Timestamp directory
    file_imus = file_path +  "/"  + 'imu'
    file_gnds = file_path +  "/" + 'gnd_truth.txt'
    
    # Check if directories exist
    file_echo_diff_list = sorted([f for f in os.listdir(file_echo_diff)])
    file_imus_list = sorted([f for f in os.listdir(file_imus)])
    
    # Non-diff directory handling: if prefer_legacy_dirs, expect it to exist (like original)
    # Otherwise, make it optional
    if prefer_legacy_dirs:
        # Original behavior: expect non_diff to exist
        file_echo_org_list = sorted([f for f in os.listdir(file_echo_org)])
        has_non_diff = True
    else:
        # New behavior: non-diff is optional
        has_non_diff = os.path.exists(file_echo_org)
        if has_non_diff:
            file_echo_org_list = sorted([f for f in os.listdir(file_echo_org)])
        else:
            file_echo_org_list = []

    with open(file_gnds, 'r', encoding='utf-8') as f:
        gt = f.read()

    gt = gt.split("\n")[:-1]

    loaded_gt = []
    data_pairs = []
    n_bad = 0
    bad_signal_remove_length = 5
    timestamp_usage_count = 0
    frame_based_count = 0
    
    file_order = [i for i in range(0, len(file_echo_diff_list))]
    
    if is_shuffle ==True:
        random.seed(seed_num)        
        random.shuffle(file_order) 
        #print(file_order)

    for i in file_order:
        # ground truth
        gnd = int(file_echo_diff_list[i].split('.')[0].split('_')[2])

        truth = gt[gnd].split(';')[3]
        #truth = extract_mouth_morphemes(gt[gnd].split(';')[3])
        #truth = extract_mouth_morphemes_none(gt[gnd].split(';')[3])
        #_, truth, _ = parse_nonmanuals(gt[gnd].split(';')[3])
        #print(i, truth)
        try:
            loaded_gt += [gt[gnd].split(';')]
            #print(loaded_gt)
            # load imu
            File_data = np.loadtxt(file_imus+"/"+file_imus_list[i], dtype=str, delimiter=" ") 
            all_imu = np.array(File_data, dtype=float)[:, :3]
            all_imu_time = np.array(File_data, dtype=float)[:, 3:]
            all_imu_time = np.array([i[0] for i in all_imu_time])
        
            # load echo_diff
            profiles = np.load(file_echo_diff+"/"+file_echo_diff_list[i])
            profile_data_piece = profiles.copy()
            profile_data_piece = profile_data_piece.swapaxes(1, 2) # 

            # load echo_org
            if prefer_legacy_dirs:
                # Original behavior: always load from file_echo_org_list
                profiles_org = np.load(file_echo_org+"/"+file_echo_org_list[i])
                profile_data_piece_org = profiles_org.copy()
                profile_data_piece_org = profile_data_piece_org.swapaxes(1, 2)[:,1:,] # org echo has always +1 frame
            else:
                # New behavior: optional - check if file exists
                profile_data_piece_org = None
                org_filename = file_echo_diff_list[i].replace('acoustic_diff_', 'acoustic_')
                org_file_path = os.path.join(file_echo_org, org_filename)
                
                if os.path.exists(org_file_path):
                    profiles_org = np.load(org_file_path)
                    profile_data_piece_org = profiles_org.copy()
                    profile_data_piece_org = profile_data_piece_org.swapaxes(1, 2)[:,1:,] # org echo has always +1 frame
            
            # Combine echo data
            if prefer_legacy_dirs:
                # Original behavior: always concatenate
                profile_data_piece_all = np.concatenate([profile_data_piece, profile_data_piece_org], axis=0)
            else:
                # New behavior: use both if available, otherwise just diff
                if profile_data_piece_org is not None:
                    profile_data_piece_all = np.concatenate([profile_data_piece, profile_data_piece_org], axis=0)
                else:
                    profile_data_piece_all = profile_data_piece
            
            # Get target length after removing bad frames
            target_length_after_trim = profile_data_piece.shape[1] - bad_signal_remove_length
            
            # Load acoustic timestamps if available and requested
            acoustic_timestamps = None
            if use_timestamps:
                # Construct timestamp filename from diff filename
                timestamp_filename = file_echo_diff_list[i].replace('acoustic_diff_', 'acoustic_timestamp_').replace('.npy', '.pkl')
                timestamp_file_path = os.path.join(file_timestamps, timestamp_filename)
                
                # Debug: Check if timestamp directory exists (only log once per session)
                if i == file_order[0] and not os.path.exists(file_timestamps):
                    print(f'     Warning: Timestamp directory not found: {file_timestamps}')
                    print(f'     Using frame-based alignment for all samples in this session')
                
                # Use verbose mode for the first few failures to diagnose issues
                verbose = (frame_based_count < 3 and os.path.exists(timestamp_file_path))
                acoustic_timestamps = load_acoustic_timestamps(timestamp_file_path, verbose=verbose)
                
                if acoustic_timestamps is not None:
                    # Remove last bad_signal_remove_length frames from timestamps too
                    if len(acoustic_timestamps) > bad_signal_remove_length:
                        acoustic_timestamps = acoustic_timestamps[:-bad_signal_remove_length]
                    timestamp_usage_count += 1
                else:
                    frame_based_count += 1
                    # Debug: Log first few failures to see why timestamps aren't found
                    if frame_based_count <= 3:
                        if not os.path.exists(timestamp_file_path):
                            print(f'     Sample {i}: Timestamp file not found: {timestamp_filename}')
                        else:
                            print(f'     Sample {i}: Timestamp file exists but failed to load: {timestamp_filename}')
                            # The verbose output from load_acoustic_timestamps will show the actual error
            else:
                frame_based_count += 1
            
            # Align IMU data using timestamps if available, otherwise use frame-based
            if acoustic_timestamps is not None and len(acoustic_timestamps) > 0:
                # Use timestamp-based alignment
                upsampled_imu_flat = align_imu_to_acoustic_timestamps(
                    all_imu, all_imu_time, acoustic_timestamps, target_length_after_trim
                )
                # Reshape to [1, time, 3]
                upsampled_imu_data = upsampled_imu_flat.reshape(1, upsampled_imu_flat.shape[0], upsampled_imu_flat.shape[1])
            else:
                # Fall back to frame-based upsampling
                if prefer_legacy_dirs:
                    # Original behavior: upsample to full length, then trim after reshape
                    psampled_time, upsampled_imu_flat = upsample_imu_data(all_imu_time, all_imu, profile_data_piece.shape[1])
                    # Reshape first, then trim (matching original behavior)
                    upsampled_imu_data = upsampled_imu_flat.reshape(1, upsampled_imu_flat.shape[0], upsampled_imu_flat.shape[1])
                else:
                    # New behavior: trim before reshape
                    _, upsampled_imu_flat = upsample_imu_data(all_imu_time, all_imu, profile_data_piece.shape[1])
                    # Remove last bad_signal_remove_length frames to match acoustic data
                    if upsampled_imu_flat.shape[0] > bad_signal_remove_length:
                        upsampled_imu_flat = upsampled_imu_flat[:-bad_signal_remove_length]
                    # Reshape to [1, time, 3]
                    upsampled_imu_data = upsampled_imu_flat.reshape(1, upsampled_imu_flat.shape[0], upsampled_imu_flat.shape[1])

            #normalized_imu_data, means, stds = normalize_imu_data(upsampled_imu_data)
            #normalized_imu_data.shape = 1, normalized_imu_data.shape[0], normalized_imu_data.shape[1]

            # Data quality check: match original behavior when prefer_legacy_dirs=True
            if prefer_legacy_dirs:
                # Original behavior from prev_utils.py: check >300 frames
                if profile_data_piece.shape[1] > 300: # check the data quality 
                    #print("final:",  i,truth, profile_data_piece.shape, normalized_imu_data.shape)
                    data_pairs += [(profile_data_piece_all[:,:-bad_signal_remove_length,:],
                                    upsampled_imu_data[:,:-bad_signal_remove_length,:],
                                    truth
                                   )]
                else:
                    n_bad +=1
            else:
                # New behavior: load all data regardless of length
                #print("final:",  i,truth, profile_data_piece.shape, normalized_imu_data.shape)
                data_pairs += [(profile_data_piece_all[:,:-bad_signal_remove_length,:],
                                upsampled_imu_data,  # Already trimmed to correct length
                                truth
                               )]
        except Exception as e:
            if prefer_legacy_dirs:
                # Original behavior: simpler error message
                print(f"Error - data load: {i, truth}")
            else:
                # New behavior: detailed error message
                print(f"Error - data load: {i, truth}, Error: {e}")

    if prefer_legacy_dirs and n_bad:
        print('     %d bad data pieces' % n_bad, file_echo_diff_list[i])
    
    if use_timestamps and (timestamp_usage_count > 0 or frame_based_count > 0):
        print(f'     Timestamp-based alignment: {timestamp_usage_count}, Frame-based fallback: {frame_based_count}')

    if is_train:
        data_pairs

    return data_pairs, loaded_gt


def round_robin_split(seq, k):
    folds = [[] for _ in range(k)]
    for i, val in enumerate(seq):
        folds[i % k].append(val)
    return folds


def dedup_all_keep_order(s: str) -> str:
    """Remove all duplicates anywhere, keep first occurrence order."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return ",".join(out)

def dedup_consecutive_only(s: str) -> str:
    """Collapse only consecutive duplicates (A,A,B,A -> A,B,A)."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    prev = object()
    for p in parts:
        if p != prev:
            out.append(p)
            prev = p
    return ",".join(out)



def parse_nonmanuals(sentence, mode = None):
    # Split by glosses
    glosses = re.findall(r'(\w+)(\([^)]+\))?', sentence.replace("fs-", ""))
    nms1_list = []
    nms2_list = []
    types = []
    emo_mouth = []
    alls = []
    sign = [re.sub(r"\(.*?\)", "", w) for w in sentence.split(" ")]
    sign = [ 'fs' if i.startswith('fs-') else i for i in sign]


    for gloss, nms in glosses:
        # Extract non-manuals if any
        if nms:
            nms_items = nms.strip("()").split(",")
            nms_items = [n.strip() for n in nms_items]
        else:
            nms_items = []

        # Determine type
        facial_markers = {"raise", "furrow", "shake", "mm", "th", "puff", "cs", "oo", "cha", "pah"}
        emotional_markers = {"happy", "sad", "angry", "scared", "surprised", "disgust", 'none'}
        mouth_morphemes = {"mm", "th", "puff", "cs", "oo", "cha", "pah",'none'}
        grammatical_markers = {"raise", "furrow", "shake",'none'}


        # Assign first and second non-manuals
        if len(nms_items) == 1:
            if nms_items[0].lower() in grammatical_markers:
              nms1 = nms_items[0]
              nms2 = "none"
              nms3 = "grammar"
              nms4 = "none"
              nms5 = nms_items[0]

            elif nms_items[0].lower() in mouth_morphemes:
              nms1 = "none"
              nms2 = nms_items[0]
              nms3 = "mouth"
              nms4 = nms_items[0]
              nms5 = nms_items[0]

            elif nms_items[0].lower() in emotional_markers:
              nms1 = "none"
              nms2 = "none"
              nms3 = nms_items[0]
              nms4 = nms_items[0]
              nms5 = nms_items[0]

            else:
              nms1 = "none"
              nms2 = "none"
              nms3 = "none"
              nms4 = "none"
              nms5 = "none"
              
        elif len(nms_items) >= 2:
            if nms_items[1].lower() in mouth_morphemes:
              nms1 = nms_items[0]
              nms2 = nms_items[1]
              nms3 = 'none'
              nms4 = nms_items[1]
              nms5 = nms_items[1]
                
        else:
            nms1 = "none"
            nms2 = "none"
            nms3 = "none"
            nms4 = "none"
            nms5 = "none"


        #nms_type = "emotion" if (nms3.lower() in emotional_markers) else "grammar"

        # Append results
        nms1_list.append(nms1.lower())
        nms2_list.append(nms2.lower())
        types.append(nms3.lower())
        emo_mouth.append(nms4.lower())
        alls.append(nms5.lower())

    if mode == 'grammar':
        nms1_list = [ i for i in nms1_list if i != 'none' ]
        out =  ",".join(nms1_list)
        #out = [ i if i not in ('mouth', 'grammar') else 'others' for i in out ]
    elif  mode == 'mouth':
        nms2_list = [ i for i in nms2_list if i != 'none' ]
        out =  ",".join(nms2_list)
    elif mode == 'emotion':
        types = [ i for i in types if i != 'none' ]
        out = ",".join(types)
    elif mode == 'signs':
        out = ",".join(sign)
    elif mode == 'emomouth':
        emo_mouth = [ i for i in emo_mouth if i != 'none' ]
        out = ",".join(emo_mouth) 
        #print(out)
    elif mode == 'all':
        alls = [ i for i in alls if i != 'none' ]
        out = ",".join(alls) 

    return  dedup_all_keep_order(out)


def round_robin_split(seq, k):
    folds = [[] for _ in range(k)]
    for i, val in enumerate(seq):
        folds[i % k].append(val)
    return folds


def stratified_kfold_single_label(y, k=5, shuffle=True, seed=42):
    """
    y: array-like of shape (N,) with a single class label per sample
    returns: list of (train_idx, val_idx) pairs
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    folds = []
    for tr, va in skf.split(np.arange(len(y)), y):
        folds.append((tr, va))
    return folds

# from torch.nn.utils.rnn import pad_sequence

# gnd_list = ["raise", "furrow", "shake", "none"]
# label_dic =  {value: index for index, value in enumerate(gnd_list)}
# label_dic_reverse = {index: value for index, value in enumerate(gnd_list)}

def collate_various_size(batch):
    data_list_arr = [x[0][0] for x in batch]
    data_list_imu = [x[0][1] for x in batch]
    target = [x[1] for x in batch]
    target_org = [x[2] for x in batch]
    
    # Check if metadata is present (session/sample info)
    has_metadata = len(batch[0]) > 3
    if has_metadata:
        metadata = [x[3] for x in batch]  # (session, sample) tuples
    else:
        metadata = None

    
    data_max_size = max([x.shape[1] for x in data_list_arr])

    # check the windown size, for example, if windion size 10, the target size should be dividied by windon size. 
    #target_length = ceil(target_length / 16) * 16
    window_size = 10
    target_length = data_max_size 
    target_length = ceil(target_length / window_size) * window_size
   
    #data_list_imu = [x[0][1].reshape(1, x[0][1].shape[0], x[0][1].shape[1]) for x in batch]

    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], target_length, data_list_arr[0].shape[2]))
    data_imu = np.zeros((len(batch), data_list_imu[0].shape[0], target_length, data_list_imu[0].shape[2]))
    
    # horizontal shifting time axis. 
    for i in range(0, len(data_list_arr)):
        start_x = random.randint(0, target_length - data_list_arr[i].shape[1])
        data_arr[i, :, start_x: start_x + data_list_arr[i].shape[1], :] = data_list_arr[i]
        data_imu[i, :, start_x: start_x + data_list_imu[i].shape[1], :] = data_list_imu[i]

    # data1 = Tensor(data_arr)
    # data2 = Tensor(data_imu)
    # return (data1, data2), target
    data_arr = data_arr.swapaxes(2,3)
    data_imu = data_imu.swapaxes(2,3)
    
    if has_metadata:
        return (data_arr, data_imu), (target, target_org, metadata)
    else:
        return (data_arr, data_imu), (target, target_org)


facial_markers = {"none", "raise", "furrow", "shake", "mm", "th", "puff", "cs", "oo", "cha", "pah"}
emotional_markers = {"happy", "sad", "angry", "scared", "surprised", "disgust", 'none'}
mouth_morphemes = { "oo", "mm", "cha", "cs", "th", "puff", "pah","none"}
grammatical_markers = {"raise", "furrow", "shake","none"}

# if len(gnd_list) == len(list(grammatical_markers)):
#     gnd_list = ["raise", "furrow", "shake", "none"]
# if len(gnd_list) == len(list(mouth_morphemes)):
#     gnd_list = ["oo", "mm", "cha", "cs", "th", "puff", "pah", "none"]
# if len(gnd_list) == len(list(emotional_markers)):
#     gnd_list = ["happy", "sad", "angry", "scared", "surprised", "disgust", 'none']


def collate_various_size_test(batch):
    data_list_arr = [x[0][0] for x in batch]
    data_list_imu = [x[0][1] for x in batch]
    target = [x[1] for x in batch]
    target_org = [x[2] for x in batch]
    
    # Check if metadata is present (session/sample info)
    has_metadata = len(batch[0]) > 3
    if has_metadata:
        metadata = [x[3] for x in batch]  # (session, sample) tuples
    else:
        metadata = None
    # #######
    
    data_max_size = max([x.shape[1] for x in data_list_arr])

    # check the windown size, for example, if windion size 10, the target size should be dividied by windon size. 
    #target_length = ceil(target_length / 16) * 16
    window_size = 10
    target_length = data_max_size 
    target_length = ceil(target_length / window_size) * window_size
   
    #data_list_imu = [x[0][1].reshape(1, x[0][1].shape[0], x[0][1].shape[1]) for x in batch]

    data_arr = np.zeros((len(batch), data_list_arr[0].shape[0], target_length, data_list_arr[0].shape[2]))
    data_imu = np.zeros((len(batch), data_list_imu[0].shape[0], target_length, data_list_imu[0].shape[2]))
    
    # horizontal shifting time axis. 
    for i in range(0, len(data_list_arr)):
        start_x = 0
        data_arr[i, :, start_x: start_x + data_list_arr[i].shape[1], :] = data_list_arr[i]
        data_imu[i, :, start_x: start_x + data_list_imu[i].shape[1], :] = data_list_imu[i]

    # data1 = Tensor(data_arr)
    # data2 = Tensor(data_imu)
    # return (data1, data2), target
    data_arr = data_arr.swapaxes(2,3)
    data_imu = data_imu.swapaxes(2,3)
    
    if has_metadata:
        return (data_arr, data_imu), (target, target_org, metadata)
    else:
        return (data_arr, data_imu), (target, target_org)


class CNNDataset(torch.utils.data.Dataset):

    def __init__(self, data, target_height, is_train):
        self.data = data
        self.is_train = is_train
        self.target_height = target_height
        
    def __getitem__(self, index):
        input_arr = self.data[index][0]
        input_imu = self.data[index][1]
        #print(input_arr.shape, input_imu.shape)
        output_arr = deepcopy(self.data[index][2])
        output_arr_org = deepcopy(self.data[index][3])
        # Extract session and sample metadata if available
        session_num = self.data[index][4] if len(self.data[index]) > 4 else None
        sample_idx = self.data[index][5] if len(self.data[index]) > 5 else None

        input_arr_copy = deepcopy(input_arr)
        input_imu_copy = deepcopy(input_imu)

        aug_arr = input_arr_copy
        aug_imu = input_imu_copy

        
        if self.is_train:
            max_poi = aug_arr.shape[2]
            random_start = random.randint(0, max_poi-  self.target_height)
            aug_arr = aug_arr[:, :, random_start:random_start+ self.target_height]
        else:
            max_poi = aug_arr.shape[2]
            random_start = 0 # should start from 300. 
            aug_arr = aug_arr[:, :, random_start:random_start+self.target_height]
   
        if self.is_train:
            if random.random() > 0.2:
                mask_width = random.randint(10, 40)
                rand_start = random.randint(0, aug_arr.shape[1] - mask_width)
                aug_arr[:, rand_start: rand_start + mask_width, :] = 0.0
                aug_imu[:, rand_start: rand_start + mask_width, :] = 0.0


        padded_input = aug_arr
        padded_imu = aug_imu

        if self.is_train:
            if random.random() > 0.2:
                noise_arr = np.random.random(padded_input.shape).astype(np.float32) * 0.1 + 0.95
                noise_imu = np.random.random(padded_imu.shape).astype(np.float32) * 0.1 + 0.95
                padded_input *= noise_arr
                padded_imu *= noise_imu
                #print('noise: ', noise_arr.shape, noise_imu.shape)

        padded_input_list = []
        
        for j in range(0, padded_input.shape[0]):
            padded_input_tmp = padded_input[j]
            for c in range(padded_input_tmp.shape[0]):
                # instance-level norm
                mu, sigma = np.mean(padded_input_tmp[c]), np.std(padded_input_tmp[c])
                #print( mu, sigma)
                
                if sigma != 0.0:
                    padded_input_tmp[c] = (padded_input_tmp[c] - mu) / sigma
                else:
                    padded_input_tmp[c] = 0.0

            padded_input_tmp = np.nan_to_num(padded_input_tmp, nan=0.0, posinf=0.0, neginf=0.0)
            padded_input_list.append(padded_input_tmp)
            #print(j, padded_input_tmp.shape)

        padded_input_fn = np.array(padded_input_list)
        padded_imu = np.nan_to_num(padded_imu, nan=0.0, posinf=0.0, neginf=0.0)

        # Return session and sample info if available
        if session_num is not None and sample_idx is not None:
            return (padded_input_fn, padded_imu), output_arr, output_arr_org, (session_num, sample_idx)
        else:
            return (padded_input_fn, padded_imu), output_arr, output_arr_org

    def __len__(self):
        return len(self.data)
    
class DataBatches:

    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.all_indices = self.batches()
        random.shuffle(self.all_indices)
        
    def __len__(self):
        return ceil(self.dataset_size / self.batch_size)

    def __iter__(self):
        for x in self.all_indices:
            yield x
        random.shuffle(self.all_indices)

    def batches(self):
        all_indices = []
        for i in range(0, self.dataset_size, self.batch_size):
            all_indices += [list(range(i, min(i + self.batch_size, self.dataset_size)))]
        return all_indices


class DataSplitter:
    train_loader: DataLoader
    val_loader: DataLoader

    def __init__(self, train_data, test_data, BATCH_SIZE, WORKER_NUM, target_height):
        train_data = train_data
        val_data = test_data
        target_height = target_height

        print('train length', len(train_data))
        print('test length', len(val_data))
        
        # convert to 'Dataloader'
        if len(train_data):
            train_dataset = CNNDataset(train_data, target_height, is_train=True)
            self.train_loader = DataLoader(
                train_dataset,
                #batch_size=BATCH_SIZE,
                # shuffle=shuffle,
                # drop_last=True,
                #sampler=sampler,
                pin_memory=True,
                prefetch_factor=2,               # PyTorch>=1.7
                persistent_workers=True,    
                num_workers=WORKER_NUM,
                collate_fn=collate_various_size,
                batch_sampler=DataBatches(len(train_dataset), BATCH_SIZE)
            )
        else:
            self.train_loader = None
        test_dataset = CNNDataset(val_data, target_height, is_train=False)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            prefetch_factor=2,               # PyTorch>=1.7
            persistent_workers=True,    
            shuffle=False,
            num_workers=WORKER_NUM,
            collate_fn=collate_various_size_test,
            # batch_sampler=DataBatches(len(test_dataset), BATCH_SIZE)
        )


# def make_Y_aux(tokens, lab2id=label_dic, V=None):
#     """
#     Returns (Ti, V) multi-hot per time step (usually one-hot; 'none' -> all zeros except optional 'none' class if you include it)
#     If you want 'none' to be a real class in the aux head, keep it in LAB2ID; else remove it from CLASSES.
#     """
#     T = len(tokens)
#     Y = torch.zeros(T, V, dtype=torch.float32)
#     for t, tok in enumerate(tokens):
#         if tok in lab2id:
#             Y[t, lab2id[tok]] = 1.0
#         # else: unknown/none -> keep zeros
#     return Y

# def make_Y_aux(seq_labels, label2id, num_classes=None):
#     """
#     seq_labels: list of labels, e.g. ["none","cha","none","pah","none"]
#     label2id: dict mapping label->int
#     num_classes: total number of classes (optional, infer if None)
#     """
#     if num_classes is None:
#         num_classes = len(label2id)
    
#     T = len(seq_labels)
#     Y_aux = np.zeros((T, num_classes), dtype=np.float32)
    
#     for t, lab in enumerate(seq_labels):
#         if lab is None or lab == "blank":
#             continue
#         if lab in label2id:
#             idx = label2id[lab]
#             Y_aux[t, idx] = 1.0   # one-hot
#         else:
#             print(f"[warn] Unknown label: {lab}")
    
#     #return torch.tensor(Y_aux)  # shape (T, num_classes)
#     return Y_aux  # shape (T, num_classes)
        

def ctc_loss_weighted(logits_TBV, tgt_list, device, blank_index):
    """
    logits_TBV: (T', B, V+1) raw logits from model
    tgt_list  : list of 1D LongTensors (targets per sample, no blanks)
    blank_index: index used for blank (usually V)
    returns scalar loss (weighted mean over batch) and per-sample losses
    """
    Tprime, B, _ = logits_TBV.shape

    # CTC expects log-probs FP32
    logp = logits_TBV.log_softmax(2).float()
    input_lengths  = torch.full((B,), Tprime, dtype=torch.long, device=device)
    target_lengths = torch.tensor([t.numel() for t in tgt_list], dtype=torch.long, device=device)
    targets_concat = (torch.cat([t.to(device) for t in tgt_list])
                      if B > 0 else torch.empty(0, dtype=torch.long, device=device))

    ctc = nn.CTCLoss(blank=blank_index, zero_infinity=True, reduction='none')
    per_sample = ctc(logp, targets_concat, input_lengths, target_lengths)  # (B,)
    return per_sample

def sample_weight_from_target_ids(tgt_ids_1d: torch.Tensor,
                                  class_rarity: torch.Tensor,
                                  agg="max",
                                  empty_default=1.0,
                                  clip=(0.5, 10.0)) -> torch.Tensor:
    """
    tgt_ids_1d: 1D LongTensor (no blanks, no 'none')
    class_rarity: Tensor[V]
    returns: scalar tensor weight
    """
    if tgt_ids_1d.numel() == 0:
        w = torch.tensor(empty_default, dtype=torch.float32)
    else:
        vals = class_rarity[tgt_ids_1d]
        if   agg == "mean": w = vals.mean()
        elif agg == "sum":  w = (vals.sum() / tgt_ids_1d.numel())
        else:               w = vals.max()
    lo, hi = clip
    return torch.clamp(w, lo, hi)



def save_checkpoint(model, optimizer, epoch, filename):
    """Save model, optimizer, and epoch number."""
    checkpoint = {
        "epoch": epoch,  
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(checkpoint, filename + "best_checkpoint_ind.pth")
    print(f"✅ Checkpoint saved at epoch {epoch+1}")

def save_cm_figure(true_label, predict_label, label_dic_reverse, best_save_path, acc, lst, anotation): 
    true_labels= [label_dic_reverse[i] for i in  true_label]
    #predicted_labels = df["Predicted Label"].tolist()
    predicted_labels= [label_dic_reverse[i] for i in predict_label]
    # Get unique class names and sort them (ensures correct label order)
    unique_classes = sorted(set(true_labels) | set(predicted_labels))
    # Compute confusion matrix with string labels
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    top_n_is = False
    
    if top_n_is == True:
        label = lst
        num_classes = len(true_label)
        # 3. Visualize a subset of the Confusion Matrix (Top N classes or most confused)
        # This is crucial for visualizing large matrices.
        top_n_classes_to_show = 50

        # Identify the most frequent classes (you might have other criteria)
        top_classes_indices = [label[0] for label in Counter(predict_label).most_common()[:top_n_classes_to_show]]
        # Create labels for the selected top classes
        top_classes_labels = [label[i] for i in top_classes_indices]

        # Filter the confusion matrix to show only the selected top classes
        cm_subset = cm[np.ix_(top_classes_indices, top_classes_indices)]
        plt.figure(figsize=(16, 14)) # Adjust size for readability
        sns.heatmap(cm_subset, annot=True, fmt='g', cmap='Blues',
                    xticklabels=top_classes_labels, yticklabels=top_classes_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title("Confusion Matrix (Top %s Most Frequent Classes) - Best Accuracy : %.3f"%(str(top_n_classes_to_show), acc) + " %")
        plt.tight_layout()
        plt.savefig(best_save_path+"confusion_matrix_top_%s.png"%(str(top_n_classes_to_show)), dpi=300, bbox_inches="tight")  # Saves as a high-quality PNG


    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5)
    # Keep the label order in figure
    plt.xticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=90)
    plt.yticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=0)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Best Accuracy : %.3f"%acc + " %")
    plt.xticks(rotation=90)  # Rotate class labels for better visibility
    plt.yticks(rotation=0)
    plt.savefig(best_save_path+"confusion_matrix_%s.png"%anotation, dpi=300, bbox_inches="tight")  # Saves as a high-quality PNG

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5)
    # Keep the label order in figure
    plt.xticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=90)
    plt.yticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=0)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Best Accuracy : %.3f"%acc + " %")
    plt.xticks(rotation=90)  # Rotate class labels for better visibility
    plt.yticks(rotation=0)
    plt.savefig(best_save_path+"confusion_matrix_cm_%s.png"%anotation, dpi=300, bbox_inches="tight")  # Saves as a high-quality PNG

