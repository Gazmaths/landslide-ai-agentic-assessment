import numpy as np

EARTH_R_M = 6_371_000.0

def haversine_m(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance (in meters) from (lat0, lon0) to arrays of (lats, lons)."""
    lat0r = np.radians(lat0)
    lon0r = np.radians(lon0)
    latsr = np.radians(lats)
    lonsr = np.radians(lons)
    dlat = latsr - lat0r
    dlon = lonsr - lon0r
    a = np.sin(dlat/2.0)**2 + np.cos(lat0r) * np.cos(latsr) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_R_M * c
