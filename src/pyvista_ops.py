# pyvista_ops.py
# Minimal backend for vertical section sampling using PyVista/NumPy.
# Focus: compute-only. No GUI. Ready for Streamlit+Plotly frontend.
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# Optional: make PyVista safe in headless servers (no GUI needed).
os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
try:
    import pyvista as pv  # reserved for future: contours/triangulation
except Exception:
    pv = None  # not strictly needed for MVP

# ---------------------------
# Helpers: geo distance <-> deg
# ---------------------------
_KM_PER_DEG_LAT = 111.0

def _km_per_deg_lon_at_lat(lat_deg: float) -> float:
    return _KM_PER_DEG_LAT * float(np.cos(np.deg2rad(lat_deg)))

def _xy_deg_to_km(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float):
    """Convert lon/lat (deg) to dx,dy in km around (lon0,lat0) using equirectangular approx."""
    kx = _km_per_deg_lon_at_lat(lat0)
    dx = (lon - lon0) * kx
    dy = (lat - lat0) * _KM_PER_DEG_LAT
    return dx, dy

def _xy_km_to_deg(dx_km: np.ndarray, dy_km: np.ndarray, lat_ref: float, lon_ref: float):
    """Inverse of _xy_deg_to_km."""
    kx = _km_per_deg_lon_at_lat(lat_ref)
    dlon = dx_km / kx
    dlat = dy_km / _KM_PER_DEG_LAT
    return lon_ref + dlon, lat_ref + dlat

def _project_point_to_segment_u_dxdy(px, py, x0, y0, x1, y1):
    """
    Project point P(px,py) onto segment AB((x0,y0)->(x1,y1)) in *km* coordinates.
    Returns: (u, dist_perp, t_clamped)
      - u: along-segment distance from A in km (0..L)
      - dist_perp: perpendicular distance to segment in km
      - t_clamped: 0..1 param after clamping to segment
    """
    vx, vy = x1 - x0, y1 - y0
    L2 = vx*vx + vy*vy
    if L2 <= 0:
        return 0.0, float(np.hypot(px - x0, py - y0)), 0.0
    t = ((px - x0)*vx + (py - y0)*vy) / L2
    t_clamped = np.clip(t, 0.0, 1.0)
    qx = x0 + t_clamped * vx
    qy = y0 + t_clamped * vy
    dist_perp = float(np.hypot(px - qx, py - qy))
    u_along = float(np.hypot(qx - x0, qy - y0))
    return u_along, dist_perp, t_clamped

# Vectorized wrapper for many points
def _project_many_to_segment(dx, dy, x0, y0, x1, y1):
    vx, vy = (x1 - x0), (y1 - y0)
    L2 = vx*vx + vy*vy
    if L2 <= 0:
        # degenerate: return distance to A
        dist = np.hypot(dx - x0, dy - y0)
        return np.zeros_like(dist), dist, np.zeros_like(dist)
    t = ((dx - x0)*vx + (dy - y0)*vy) / L2
    t_clamped = np.clip(t, 0.0, 1.0)
    qx = x0 + t_clamped * vx
    qy = y0 + t_clamped * vy
    dist_perp = np.hypot(dx - qx, dy - qy)
    u_along = np.hypot(qx - x0, qy - y0)
    return u_along, dist_perp, t_clamped

# ---------------------------
# Core: build section grid
# ---------------------------
def make_section_grid(
    points_df: pd.DataFrame,
    p1_deg: tuple[float, float],
    p2_deg: tuple[float, float],
    half_thickness_km: float = 0.25,
    res_u: int = 400,
    res_v: int = 200,
    agg: str = "mean",               # "mean"|"median"|"max"|"absmax"
    use_log_color: bool = True,
    v_range: tuple[float, float] | None = None,  # optional depth window; default from data
    min_count: int = 1,
):
    """
    Build a vertical section (2D plane) along p1->p2 with corridor half-thickness in XY (km),
    aggregating points onto a (res_u × res_v) grid.

    Parameters
    ----------
    points_df : DataFrame
        Columns required: LONGITUDE, LATITUDE (deg), DEPTH (same unit as data, e.g., m),
        VALUE (float). SOURCE is ignored here; pre-filter outside if needed.
    p1_deg, p2_deg : (lon, lat) in degrees
        Two endpoints defining section trace in XY.
    half_thickness_km : float
        Corridor half-width in km (distance in XY to the section line).
    res_u, res_v : int
        Grid resolution along-trace (u, in km) and vertical (v=DEPTH).
    agg : str
        Aggregation: "mean", "median", "max", or "absmax".
    use_log_color : bool
        If True, val_disp = log10(max(val_raw, 1e-6)) for coloring only.
    v_range : (vmin, vmax) or None
        Vertical/depth range for the section. Default spans the data's min..max DEPTH.
    min_count : int
        Minimum sample count per cell to keep in the output.

    Returns
    -------
    dict with keys:
      - "X","Y","Z"           : 1D arrays of grid cell centers (world coords)
      - "val_raw","val_disp"  : aggregated raw value and display value (maybe log10)
      - "count"               : samples per cell
      - "u_centers_km"        : 1D array (len=res_u) of along-trace centers (km)
      - "v_centers_depth"     : 1D array (len=res_v) of vertical centers (DEPTH units)
      - "grid_shape"          : (res_v, res_u)
      - "meta"                : dict with L_km, p1_deg, p2_deg, half_thickness_km
    """
    need = ["LONGITUDE", "LATITUDE", "DEPTH", "VALUE"]
    miss = [c for c in need if c not in points_df.columns]
    if miss:
        raise ValueError(f"Required columns missing: {miss}")

    # Clean + arrays
    df = points_df.dropna(subset=need).copy()
    if df.empty:
        return _empty_section()

    lon = df["LONGITUDE"].to_numpy(float)
    lat = df["LATITUDE"].to_numpy(float)
    dep = df["DEPTH"].to_numpy(float)
    val = df["VALUE"].to_numpy(float)

    # Section reference
    lon1, lat1 = float(p1_deg[0]), float(p1_deg[1])
    lon2, lat2 = float(p2_deg[0]), float(p2_deg[1])
    lat_ref = 0.5 * (lat1 + lat2)

    # Convert XY to km around p1 (local frame)
    x0, y0 = 0.0, 0.0
    x1y1 = _xy_deg_to_km(np.array([lon2]), np.array([lat2]), lon1, lat1)
    x1, y1 = float(x1y1[0]), float(x1y1[1])
    L_km = float(np.hypot(x1 - x0, y1 - y0))
    if not np.isfinite(L_km) or L_km <= 0:
        # Degenerate line; treat as vertical billboard above p1
        x1, y1, L_km = 0.0, 0.0, 1e-6

    # Project all points to u (km) + dist_perp (km)
    dx, dy = _xy_deg_to_km(lon, lat, lon1, lat1)
    u_km, dist_perp_km, _ = _project_many_to_segment(dx, dy, x0, y0, x1, y1)

    # Filter within corridor and u in [0,L]
    mask = (dist_perp_km <= float(half_thickness_km)) & (u_km >= 0.0) & (u_km <= L_km)
    if not np.any(mask):
        return _empty_section(meta=dict(L_km=L_km, p1_deg=p1_deg, p2_deg=p2_deg, half_thickness_km=half_thickness_km))

    u_km = u_km[mask]
    dep  = dep[mask]
    val  = val[mask]
    dx   = dx[mask]
    dy   = dy[mask]

    # Vertical range
    if v_range is None:
        vmin, vmax = float(np.nanmin(dep)), float(np.nanmax(dep))
    else:
        vmin, vmax = float(v_range[0]), float(v_range[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(dep)), float(np.nanmax(dep))

    # Bins
    u_edges = np.linspace(0.0, L_km, int(res_u) + 1)
    v_edges = np.linspace(vmin, vmax, int(res_v) + 1)
    # Bin indices
    iu = np.clip(np.searchsorted(u_edges, u_km, side="right") - 1, 0, int(res_u) - 1)
    iv = np.clip(np.searchsorted(v_edges, dep,  side="right") - 1, 0, int(res_v) - 1)

    # Aggregate per cell
    gdf = pd.DataFrame({"iu": iu, "iv": iv, "val": val})
    if agg == "median":
        agg_ser = gdf.groupby(["iv", "iu"])["val"].median()
    elif agg == "max":
        agg_ser = gdf.groupby(["iv", "iu"])["val"].max()
    elif agg == "absmax":
        # take the element whose |val| is maximal, preserving sign
        def _absmax(s):
            a = s.to_numpy()
            return a[int(np.argmax(np.abs(a)))] if len(a) else np.nan
        agg_ser = gdf.groupby(["iv", "iu"])["val"].agg(_absmax)
    else:
        agg_ser = gdf.groupby(["iv", "iu"])["val"].mean()

    cnt_ser = gdf.groupby(["iv", "iu"])["val"].size()
    agg_df = pd.DataFrame({"val_raw": agg_ser, "count": cnt_ser}).reset_index()
    if min_count > 1:
        agg_df = agg_df[agg_df["count"] >= int(min_count)]
    if len(agg_df) == 0:
        return _empty_section(meta=dict(L_km=L_km, p1_deg=p1_deg, p2_deg=p2_deg, half_thickness_km=half_thickness_km))

    # Centers in (u_km, v_depth)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    # Map (iu,iv) -> world XYZ centers
    iu_arr = agg_df["iu"].to_numpy(int)
    iv_arr = agg_df["iv"].to_numpy(int)
    u_sel  = u_centers[iu_arr]
    v_sel  = v_centers[iv_arr]  # depth

    # Direction unit vector along the section in km
    if L_km <= 0:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = (x1 - x0) / L_km, (y1 - y0) / L_km

    # Convert (u_km along trace) back to lon/lat (deg)
    dx_sel = u_sel * ux
    dy_sel = u_sel * uy
    lon_sel, lat_sel = _xy_km_to_deg(dx_sel, dy_sel, lat_ref=lat1, lon_ref=lon1)

    # Final arrays
    val_raw = agg_df["val_raw"].to_numpy(float)
    val_disp = np.log10(np.clip(val_raw, 1e-6, None)) if use_log_color else val_raw
    count = agg_df["count"].to_numpy(int)

    out = {
        "X": lon_sel,
        "Y": lat_sel,
        "Z": v_sel,                 # Depth as-is (frontend may set zaxis autorange='reversed')
        "val_raw": val_raw,
        "val_disp": val_disp,
        "count": count,
        "u_centers_km": u_centers,
        "v_centers_depth": v_centers,
        "grid_shape": (res_v, res_u),
        "meta": dict(L_km=L_km, p1_deg=p1_deg, p2_deg=p2_deg, half_thickness_km=half_thickness_km),
    }
    return out

def _empty_section(meta: dict | None = None):
    return {
        "X": np.array([], dtype=float),
        "Y": np.array([], dtype=float),
        "Z": np.array([], dtype=float),
        "val_raw": np.array([], dtype=float),
        "val_disp": np.array([], dtype=float),
        "count": np.array([], dtype=int),
        "u_centers_km": np.array([], dtype=float),
        "v_centers_depth": np.array([], dtype=float),
        "grid_shape": (0, 0),
        "meta": (meta or {}),
    }

# ---------------------------
# (Optional) future hooks
# ---------------------------
def build_contours_on_section(values_2d: np.ndarray, x_2d: np.ndarray, y_2d: np.ndarray, z_2d: np.ndarray, levels: list[float]):
    """
    Placeholder for future: use PyVista StructuredGrid to extract contour polylines.
    Not required for MVP and not used by the current frontend. Returns [] for now.
    """
    # Implementation can be added later once we render full surfaces instead of points.
    return []
def line_lonlat_from_u(p1_deg, p2_deg, u_km):
    """
    从 p1->p2 的剖面线上，把一维弧长 u_km（km）转换回 (lon, lat)。
    与 make_section_grid 使用同一近似：以 p1 的纬度作为换算基准。
    """
    lon1, lat1 = float(p1_deg[0]), float(p1_deg[1])
    lon2, lat2 = float(p2_deg[0]), float(p2_deg[1])
    # 向量（km）
    from math import cos, radians, hypot
    kx = 111.0 * cos(radians(lat1))
    dx = (lon2 - lon1) * kx
    dy = (lat2 - lat1) * 111.0
    L = hypot(dx, dy) if (dx or dy) else 1e-6
    ux, uy = dx / L, dy / L
    u = np.asarray(u_km, dtype=float)
    dxu, dyu = u * ux, u * uy
    dlon = dxu / kx
    dlat = dyu / 111.0
    return lon1 + dlon, lat1 + dlat
# ---------- Azimuth helpers ----------

def _dxdy_from_azimuth(lat_ref_deg: float, az_deg: float, L_km: float):
    """
    Convert azimuth (clockwise from North) and length to dx,dy in km.
    North -> +y (km), East -> +x (km).
    """
    az = np.deg2rad(float(az_deg))
    ux = np.sin(az)  # along East when az=90
    uy = np.cos(az)  # along North when az=0
    dx = float(L_km) * ux
    dy = float(L_km) * uy
    return dx, dy

def endpoints_by_azimuth(center_lonlat: tuple[float, float], az_deg: float, length_km: float):
    """
    Build two endpoints (p1, p2) centered at (lon0,lat0), oriented by az_deg, with total length=length_km.
    """
    lon0, lat0 = float(center_lonlat[0]), float(center_lonlat[1])
    dx, dy = _dxdy_from_azimuth(lat0, az_deg, length_km * 0.5)
    kx = _km_per_deg_lon_at_lat(lat0)
    dlon = dx / kx
    dlat = dy / _KM_PER_DEG_LAT
    p1 = (lon0 - dlon, lat0 - dlat)
    p2 = (lon0 + dlon, lat0 + dlat)
    return p1, p2

def make_section_by_azimuth(
    points_df: pd.DataFrame,
    center_lonlat: tuple[float, float],
    az_deg: float = 45.0,
    length_km: float = 20.0,
    half_thickness_km: float = 0.5,
    res_u: int = 400,
    res_v: int = 200,
    agg: str = "mean",
    use_log_color: bool = True,
    v_range: tuple[float, float] | None = None,
    min_count: int = 1,
):
    """
    Wrapper: build a vertical section by azimuth around a given center point.
    Internally calls make_section_grid with computed p1/p2 endpoints.
    """
    p1, p2 = endpoints_by_azimuth(center_lonlat, az_deg, length_km)
    return make_section_grid(
        points_df, p1, p2,
        half_thickness_km=half_thickness_km,
        res_u=res_u, res_v=res_v,
        agg=agg, use_log_color=use_log_color,
        v_range=v_range, min_count=min_count
    )

# ---------- Horizontal (depth) slice ----------

def make_horizontal_slice(
    points_df: pd.DataFrame,
    depth_value: float,
    half_window: float = 5.0,     # same unit as DEPTH (e.g., meters)
    grid_km: float = 1.0,         # XY grid size in km
    agg: str = "mean",
    use_log_color: bool = True,
    lonlat_bounds: tuple[float, float, float, float] | None = None,  # (lon_min, lat_min, lon_max, lat_max)
    min_count: int = 1,
):
    """
    Aggregate points around a horizontal slice at given depth (depth_value +/- half_window).
    Returns 2D grids ready for Plotly Surface.
    """
    need = ["LONGITUDE", "LATITUDE", "DEPTH", "VALUE"]
    miss = [c for c in need if c not in points_df.columns]
    if miss:
        raise ValueError(f"Required columns missing: {miss}")

    df = points_df.dropna(subset=need).copy()
    if df.empty:
        return _empty_horizontal()

    # Filter by depth window
    d0, hw = float(depth_value), float(half_window)
    mask = (df["DEPTH"] >= d0 - hw) & (df["DEPTH"] <= d0 + hw)
    df = df[mask]
    if df.empty:
        return _empty_horizontal(meta={"depth": d0, "half_window": hw})

    # Bounds in lon/lat (either provided or from data)
    if lonlat_bounds is None:
        lon_min, lon_max = float(df["LONGITUDE"].min()), float(df["LONGITUDE"].max())
        lat_min, lat_max = float(df["LATITUDE"].min()), float(df["LATITUDE"].max())
    else:
        lon_min, lat_min, lon_max, lat_max = map(float, lonlat_bounds)

    # Convert lon/lat to local km coordinates around the lower-left bound (lat_ref = mid-lat)
    lat_ref = 0.5 * (lat_min + lat_max)
    kx = _km_per_deg_lon_at_lat(lat_ref)
    x_km = (df["LONGITUDE"].to_numpy() - lon_min) * kx
    y_km = (df["LATITUDE"].to_numpy() - lat_min) * _KM_PER_DEG_LAT

    # Build grid edges
    width_km  = max(1e-6, (lon_max - lon_min) * kx)
    height_km = max(1e-6, (lat_max - lat_min) * _KM_PER_DEG_LAT)
    nx = max(1, int(np.ceil(width_km  / float(grid_km))))
    ny = max(1, int(np.ceil(height_km / float(grid_km))))
    x_edges = np.linspace(0.0, nx * grid_km, nx + 1)
    y_edges = np.linspace(0.0, ny * grid_km, ny + 1)

    ix = np.clip(np.searchsorted(x_edges, x_km, side="right") - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(y_edges, y_km, side="right") - 1, 0, ny - 1)

    gdf = pd.DataFrame({"ix": ix, "iy": iy, "val": df["VALUE"].to_numpy(float)})
    if agg == "median":
        agg_ser = gdf.groupby(["iy", "ix"])["val"].median()
    elif agg == "max":
        agg_ser = gdf.groupby(["iy", "ix"])["val"].max()
    elif agg == "absmax":
        def _absmax(s):
            a = s.to_numpy()
            return a[int(np.argmax(np.abs(a)))] if len(a) else np.nan
        agg_ser = gdf.groupby(["iy", "ix"])["val"].agg(_absmax)
    else:
        agg_ser = gdf.groupby(["iy", "ix"])["val"].mean()

    cnt_ser = gdf.groupby(["iy", "ix"])["val"].size()
    agg_df = pd.DataFrame({"val_raw": agg_ser, "count": cnt_ser}).reset_index()
    if min_count > 1:
        agg_df = agg_df[agg_df["count"] >= int(min_count)]
    if agg_df.empty:
        return _empty_horizontal(meta={"depth": d0, "half_window": hw})

    # Grid centers (km)
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Convert centers back to lon/lat and expand to 2D arrays
    lon_cent = lon_min + x_cent / kx
    lat_cent = lat_min + y_cent / _KM_PER_DEG_LAT
    X2d = np.tile(lon_cent, (len(lat_cent), 1))           # (ny, nx)
    Y2d = np.tile(lat_cent.reshape(-1, 1), (1, len(lon_cent)))
    Z2d = np.full_like(X2d, fill_value=d0, dtype=float)   # constant depth plane

    # Paint values into a 2D grid (ny, nx)
    grid = np.full((len(lat_cent), len(lon_cent)), np.nan, dtype=float)
    for _, row in agg_df.iterrows():
        grid[int(row["iy"]), int(row["ix"])] = float(row["val_raw"])

    C2d = np.log10(np.clip(grid, 1e-6, None)) if use_log_color else grid

    out = {
        "X2d": X2d, "Y2d": Y2d, "Z2d": Z2d,
        "val_raw_2d": grid, "val_disp_2d": C2d,
        "nx": X2d.shape[1], "ny": X2d.shape[0],
        "meta": {"depth": d0, "half_window": hw, "grid_km": grid_km}
    }
    return out

def _empty_horizontal(meta: dict | None = None):
    return {
        "X2d": np.zeros((0, 0), float),
        "Y2d": np.zeros((0, 0), float),
        "Z2d": np.zeros((0, 0), float),
        "val_raw_2d": np.zeros((0, 0), float),
        "val_disp_2d": np.zeros((0, 0), float),
        "nx": 0, "ny": 0, "meta": (meta or {})
    }
