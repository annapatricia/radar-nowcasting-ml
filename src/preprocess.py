import numpy as np

DBZ_MIN = 0.0
DBZ_MAX = 70.0

def normalize_dbz(dbz):
    dbz = np.clip(dbz, DBZ_MIN, DBZ_MAX)
    return (dbz - DBZ_MIN) / (DBZ_MAX - DBZ_MIN)

def label_strong_cell(dbz, threshold_dbz=45.0):
    return int(np.nanmax(dbz) >= threshold_dbz)
