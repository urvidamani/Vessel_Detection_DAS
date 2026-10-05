import h5py
import numpy as np
import pandas as pd

FAULTY_CHANNELS=[59,60,61]

def load_dataset(data_path,indices=None):
    """
    Loads the Marklinks-NS DAS dataset.

    Returns:
        X : DAS feature data
        y_distance : distance to nearest vessel
        datatime : recording timestamps
        ship_info : vessel information
    """
    with h5py.File(data_path,"r") as f:
        if indices is not None:
            X=f["X"][indices]
            y_distance=f["y"][indices]
            datetimes_raw=f["datetimes"][indices]
        else:
            X=f["X"][:]
            y_distance=f["y"][:]
            datetimes_raw=f["datetimes"][:]
        
        ship_info={}
        for field in f["ship_info"].keys():
            ship_info[field]=f["ship_info"][field][indices] if indices is not None else f["ship_info"][field][:]      
    return X,y_distance,datetimes_raw,ship_info

def decode_datetimes(datetimes_raw):
    decoded=[]
    for value in datetimes_raw:
        if isinstance(value,bytes):
            value=value.decode()
        decoded.append(value)
    return pd.to_datetime(decoded,errors="coerce")

def remove_faulty_channels(X):
    good_channels=[
        channel for channel in range(X.shape[1])
        if channel not in FAULTY_CHANNELS
    ]
    x_clean=X[:,good_channels,:]
    return x_clean,good_channels

def create_binary_label(y_distance,threshold=1000):
    """
    1 = vessel within threshold
    0 = vessel at or beyond threshold 
    """
    y_binary=(y_distance<threshold).astype(np.float32)
    return y_binary

def load_subset_by_days(data_path, days_list):
    """
    Loads only the specific calender days with faulty channels already removed.
    """
    import h5py
    with h5py.File(data_path,"r") as f:
        datetimes_raw_all=f["datetimes"][:]
    
    datetimes_all=decode_datetimes(datetimes_raw_all)
    days_all=datetimes_all.date
    
    target_days=[pd.to_datetime(d).date() for d in days_list]
    mask=np.isin(days_all,target_days)
    indices=np.where(mask)[0]
    
    X,y_distance,datetimes_raw,ship_info=load_dataset(data_path,indices=indices)
    X_clean,good_channels=remove_faulty_channels(X)
    
    return X_clean,y_distance,datetimes_raw,ship_info,good_channels