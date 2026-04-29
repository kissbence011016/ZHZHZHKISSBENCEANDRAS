import unicodedata
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from matplotlib import colormaps
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely.geometry import Point

# Try fast vectorized point-in-polygon if available (Shapely 2.x)
try:
    from shapely import contains_xy  # type: ignore
except Exception:
    contains_xy = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CSV_PATH = Path("./Szeged_with_coordinates.csv")
SHP_PATH = Path("./SHP/border.shp")

PARAMETER_OPTIONS = [
    "Cu",
    "Ni",
    "Co",
    "Cr",
    "Cd",
    "Pb",
    "Zn",
    "As",
    "NH4",
    "PO4",
    "NO3",
    "NO2",
    "pH",
    "temperature",
    "conductivity",
]

# Known aliases in the CSV for required UI parameter names.
PARAMETER_ALIASES = {
    "temperature": ["temperature", "temp", "t"],
    "conductivity": ["conductivity", "vez.kep", "vez.kép", "vezkep", "ec"],
}

# Illustrative drinking-water style limits in micrograms/liter (ug/L),
# except pH (unitless), temperature (C), conductivity (uS/cm).
# These are editable in the sidebar and should be validated against your exact
# measurement units before formal health assessment.
HEALTH_LIMITS = {
    "Cu": 2000.0,
    "Ni": 70.0,
    "Co": 100.0,
    "Cr": 50.0,
    "Cd": 3.0,
    "Pb": 10.0,
    "Zn": 3000.0,
    "As": 10.0,
    "NH4": 500.0,
    "PO4": 500.0,
    "NO3": 50000.0,
    "NO2": 3000.0,
    "pH": 8.5,
    "temperature": 25.0,
    "conductivity": 2500.0,
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalize text for robust column-name matching (accents/case/punctuation)."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    return "".join(ch for ch in text if ch.isalnum())


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find best matching column from candidate names using normalized comparison."""
    normalized_cols = {normalize_text(c): c for c in columns}

    for candidate in candidates:
        key = normalize_text(candidate)
        if key in normalized_cols:
            return normalized_cols[key]

    # Soft fallback: partial match.
    for candidate in candidates:
        key = normalize_text(candidate)
        for norm_col, original_col in normalized_cols.items():
            if key and key in norm_col:
                return original_col

    return None


@st.cache_data
def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load well measurement data (cached)."""
    return pd.read_csv(csv_path, encoding="utf-8")


@st.cache_data
def load_border(shp_path: Path) -> gpd.GeoDataFrame:
    """Load boundary polygon (cached)."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        # Fallback assumption if CRS metadata is missing.
        gdf = gdf.set_crs(epsg=3857)
    return gdf


def resolve_parameter_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map UI parameter names to actual CSV columns."""
    cols = list(df.columns)
    mapping: dict[str, str | None] = {}

    for param in PARAMETER_OPTIONS:
        if param in {"temperature", "conductivity"}:
            mapping[param] = find_column(cols, PARAMETER_ALIASES[param])
        else:
            mapping[param] = param if param in cols else find_column(cols, [param])

    return mapping


def build_time_axis(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Build a slider-ready time axis.

    Preference:
    1) Year+Month columns (for true temporal grouping of wells per campaign),
    2) fallback to a single time/id column if present.
    """
    cols = list(df.columns)
    year_col = find_column(cols, ["Év", "ev", "year"])
    month_col = find_column(cols, ["Hónap", "honap", "month"])

    df = df.copy()

    if year_col and month_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
        df = df.dropna(subset=[year_col, month_col])

        # Build human-readable labels such as "Year 2 - Month 07".
        df["_time_label"] = (
            "Year "
            + df[year_col].astype(int).astype(str)
            + " - Month "
            + df[month_col].astype(int).astype(str).str.zfill(2)
        )

        time_order = (
            df[[year_col, month_col, "_time_label"]]
            .drop_duplicates()
            .sort_values([year_col, month_col])
        )
        options = time_order["_time_label"].tolist()
        return df, options, "_time_label"

    time_col = find_column(cols, ["idő", "idõ", "time", "date", "datum"])
    if not time_col:
        raise ValueError("No usable time/date column found in the CSV.")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    options_num = sorted(df[time_col].unique().tolist())
    options = [str(v) for v in options_num]
    df["_time_label"] = df[time_col].astype(str)

    return df, options, "_time_label"


def idw_interpolate(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    power: float = 2.0,
    k: int = 8,
) -> np.ndarray:
    """
    Inverse Distance Weighting (IDW):

    z(x0) = sum( z_i / d_i^p ) / sum( 1 / d_i^p )

    where d_i is distance from query location x0 to observation i,
    and p is the IDW power parameter.
    """
    points = np.column_stack([x_obs, y_obs])
    queries = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    tree = cKDTree(points)
    k_eff = min(max(1, k), len(points))
    distances, indices = tree.query(queries, k=k_eff)

    if k_eff == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    z_neighbors = z_obs[indices]

    # If a query lands exactly on a sampled point (distance=0), return exact value.
    zero_mask = distances == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / np.power(distances, power)

    weights[~np.isfinite(weights)] = 0.0

    # Exact-value override for zero-distance cases.
    exact_value = np.any(zero_mask, axis=1)
    z_pred = np.sum(weights * z_neighbors, axis=1) / np.sum(weights, axis=1)

    if np.any(exact_value):
        first_zero_idx = np.argmax(zero_mask, axis=1)
        z_pred[exact_value] = z_neighbors[np.arange(len(z_neighbors)), first_zero_idx][exact_value]

    return z_pred.reshape(x_grid.shape)


def polygon_mask(union_geom, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Return boolean mask of grid cells that are inside the boundary polygon."""
    xv = x_grid.ravel()
    yv = y_grid.ravel()

    if contains_xy is not None:
        inside = np.asarray(contains_xy(union_geom, xv, yv), dtype=bool)
    else:
        # Fallback for older Shapely versions.
        inside = np.array([union_geom.contains(Point(x, y)) for x, y in zip(xv, yv)], dtype=bool)

    return inside.reshape(x_grid.shape)


def smooth_clipped_grid(values: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply NaN-aware Gaussian smoothing to reduce interpolation speckle/noise.

    We smooth both:
    1) the filled value grid, and
    2) a binary validity mask,
    then divide them so NaN regions do not bleed into valid areas.
    """
    if sigma <= 0:
        return values

    valid = np.isfinite(values)
    if not np.any(valid):
        return values

    filled = np.where(valid, values, 0.0)
    smooth_values = gaussian_filter(filled, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = np.where(smooth_weights > 1e-9, smooth_values / smooth_weights, np.nan)

    # Keep outside-border pixels transparent/invalid.
    smoothed[~valid] = np.nan
    return smoothed


def exceedance_rgba(values: np.ndarray, threshold: float, alpha: int = 165) -> np.ndarray:
    """
    Build a transparent red overlay where interpolated values exceed threshold.
    """
    overlay = np.zeros((*values.shape, 4), dtype=np.uint8)
    mask = np.isfinite(values) & (values >= threshold)
    overlay[mask] = np.array([220, 30, 30, alpha], dtype=np.uint8)
    return overlay


def rgba_from_grid(values: np.ndarray, cmap_name: str = "turbo") -> tuple[np.ndarray, float, float]:
    """Convert interpolated grid to RGBA image array with transparent NaNs."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("No valid interpolated values to render.")

    vmin = float(np.nanpercentile(valid, 2))
    vmax = float(np.nanpercentile(valid, 98))
    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps.get_cmap(cmap_name)

    rgba = (cmap(norm(values)) * 255).astype(np.uint8)
    rgba[~np.isfinite(values), 3] = 0  # transparent outside clipped area

    return rgba, vmin, vmax


def create_interpolation_map(
    wells_wgs84: gpd.GeoDataFrame,
    border_wgs84: gpd.GeoDataFrame,
    raster_rgba: np.ndarray,
    raster_bounds_wgs84: list[list[float]],
    parameter_name: str,
    vmin: float,
    vmax: float,
    threshold_value: float | None = None,
    exceed_rgba: np.ndarray | None = None,
) -> folium.Map:
    """Build folium map with OSM basemap, markers, boundary, raster, and legend."""
    center = [
        float(wells_wgs84.geometry.y.mean()),
        float(wells_wgs84.geometry.x.mean()),
    ]

    fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)

    # Boundary overlay
    folium.GeoJson(
        border_wgs84,
        name="Border",
        style_function=lambda _: {"fillOpacity": 0.0, "color": "black", "weight": 2},
    ).add_to(fmap)

    # Interpolation raster overlay (already clipped outside polygon via alpha=0).
    folium.raster_layers.ImageOverlay(
        image=raster_rgba,
        bounds=raster_bounds_wgs84,
        name=f"IDW {parameter_name}",
        opacity=0.7,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(fmap)

    # Optional threshold exceedance overlay (red mask).
    if exceed_rgba is not None and threshold_value is not None:
        folium.raster_layers.ImageOverlay(
            image=exceed_rgba,
            bounds=raster_bounds_wgs84,
            name=f"Exceedance >= {threshold_value:g}",
            opacity=0.85,
            interactive=False,
            cross_origin=False,
            zindex=2,
        ).add_to(fmap)

    # Well markers
    for _, row in wells_wgs84.iterrows():
        val = row.get("_selected_value", np.nan)
        popup = f"Well: {row.get('_well_id', 'N/A')}<br>{parameter_name}: {val:.3f}" if pd.notna(val) else f"Well: {row.get('_well_id', 'N/A')}<br>{parameter_name}: NaN"
        over_limit = bool(pd.notna(val) and threshold_value is not None and val >= threshold_value)

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="white",
            weight=1,
            fill=True,
            fill_color="#d62728" if over_limit else "#1f77b4",
            fill_opacity=0.95,
            popup=folium.Popup(popup, max_width=280),
        ).add_to(fmap)

    # Legend using branca colormap.
    legend = LinearColormap(
        colors=["#313695", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#fdae61", "#f46d43", "#d73027"],
        vmin=vmin,
        vmax=vmax,
        caption=f"{parameter_name} (IDW)"
    )
    legend.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Geoinformatics Dashboard - IDW", layout="wide")
    st.title("Programming for Geoinformatics - Well Interpolation Dashboard")

    if not CSV_PATH.exists():
        st.error(f"CSV not found: {CSV_PATH}")
        st.stop()
    if not SHP_PATH.exists():
        st.error(f"Shapefile not found: {SHP_PATH}")
        st.stop()

    df_raw = load_csv(CSV_PATH)
    border = load_border(SHP_PATH)

    # Resolve columns robustly (accents/aliases).
    param_col_map = resolve_parameter_columns(df_raw)

    # Sidebar controls.
    st.sidebar.header("Controls")
    selected_param_ui = st.sidebar.selectbox("Select parameter", PARAMETER_OPTIONS, index=0)
    selected_param_col = param_col_map.get(selected_param_ui)

    if selected_param_col is None:
        st.error(f"Parameter column for '{selected_param_ui}' not found in CSV.")
        st.stop()

    df_time, time_options, time_col = build_time_axis(df_raw)
    selected_time = st.sidebar.select_slider("Select time/date", options=time_options, value=time_options[0])
    st.sidebar.subheader("Interpolation Settings")
    idw_power = st.sidebar.slider("IDW power (p)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    idw_neighbors = st.sidebar.slider("Neighbors (k)", min_value=3, max_value=20, value=8, step=1)
    grid_size = st.sidebar.slider("Grid resolution", min_value=120, max_value=320, value=220, step=20)
    smooth_sigma = st.sidebar.slider("Smoothing sigma", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
    st.sidebar.subheader("Health Limit / Deadline")
    default_limit = HEALTH_LIMITS.get(selected_param_ui, 0.0)
    threshold_value = st.sidebar.number_input(
        "Limit value",
        min_value=0.0,
        value=float(default_limit),
        step=max(0.1, float(default_limit) * 0.05 if default_limit > 0 else 0.1),
        help="Editable threshold used for exceedance highlighting on map and diagrams.",
    )

    # Filter for selected time.
    filtered = df_time[df_time[time_col] == selected_time].copy()

    # Detect core columns.
    lat_col = find_column(list(filtered.columns), ["latitude", "lat"])
    lon_col = find_column(list(filtered.columns), ["longitude", "lon", "lng"])
    well_col = find_column(list(filtered.columns), ["kút", "kut", "well", "wellid", "id"])

    if lat_col is None or lon_col is None:
        st.error("Latitude/longitude columns were not found in the CSV.")
        st.stop()

    # Convert numeric and drop invalid coordinates.
    filtered[lat_col] = pd.to_numeric(filtered[lat_col], errors="coerce")
    filtered[lon_col] = pd.to_numeric(filtered[lon_col], errors="coerce")
    filtered[selected_param_col] = pd.to_numeric(filtered[selected_param_col], errors="coerce")

    filtered = filtered.dropna(subset=[lat_col, lon_col])

    if filtered.empty:
        st.warning("No records available for selected timestamp after coordinate cleaning.")
        st.stop()

    # Keep all wells for markers, but drop NaN parameter values for interpolation input.
    marker_df = filtered.copy()
    interp_df = filtered.dropna(subset=[selected_param_col]).copy()

    st.sidebar.markdown(f"**Records at selected time:** {len(filtered)}")
    st.sidebar.markdown(f"**Used for interpolation:** {len(interp_df)}")

    if len(interp_df) < 3:
        st.warning("Not enough valid points for interpolation (need at least 3 with non-NaN values).")
        st.stop()

    # Build GeoDataFrames in WGS84 from CSV coordinates.
    marker_gdf = gpd.GeoDataFrame(
        marker_df,
        geometry=gpd.points_from_xy(marker_df[lon_col], marker_df[lat_col]),
        crs="EPSG:4326",
    )
    interp_gdf = gpd.GeoDataFrame(
        interp_df,
        geometry=gpd.points_from_xy(interp_df[lon_col], interp_df[lat_col]),
        crs="EPSG:4326",
    )

    # Harmonize boundary CRS to web map CRS.
    border_wgs84 = border.to_crs(epsg=4326)

    # Reproject to metric CRS for IDW distance computations.
    interp_m = interp_gdf.to_crs(epsg=3857)
    border_m = border_wgs84.to_crs(epsg=3857)
    border_union = border_m.union_all() if hasattr(border_m, "union_all") else border_m.unary_union

    x_obs = interp_m.geometry.x.to_numpy()
    y_obs = interp_m.geometry.y.to_numpy()
    z_obs = interp_m[selected_param_col].to_numpy(dtype=float)

    # Build interpolation grid over boundary bounding box (metric coordinates).
    minx, miny, maxx, maxy = border_m.total_bounds

    x_lin = np.linspace(minx, maxx, grid_size)
    # y is descending so raster row 0 corresponds to north (top of map image).
    y_lin = np.linspace(maxy, miny, grid_size)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)

    # IDW interpolation.
    z_grid = idw_interpolate(
        x_obs=x_obs,
        y_obs=y_obs,
        z_obs=z_obs,
        x_grid=x_grid,
        y_grid=y_grid,
        power=idw_power,
        k=idw_neighbors,
    )

    # Clip raster strictly to the border polygon: outside pixels become NaN/transparent.
    inside_mask = polygon_mask(border_union, x_grid, y_grid)
    z_grid_clipped = np.where(inside_mask, z_grid, np.nan)
    z_grid_clipped = smooth_clipped_grid(z_grid_clipped, sigma=smooth_sigma)

    # Convert interpolated grid to RGBA for map overlay.
    raster_rgba, vmin, vmax = rgba_from_grid(z_grid_clipped, cmap_name="turbo")
    exceed_rgba = exceedance_rgba(z_grid_clipped, threshold=threshold_value, alpha=150)

    # Convert raster extent to WGS84 bounds for folium ImageOverlay.
    # Bounds format: [[south, west], [north, east]]
    bbox_m = gpd.GeoSeries([border_union], crs="EPSG:3857").to_crs(epsg=4326).total_bounds
    west, south, east, north = bbox_m
    raster_bounds_wgs84 = [[south, west], [north, east]]

    # Prepare marker fields.
    marker_gdf = marker_gdf.copy()
    marker_gdf["_selected_value"] = pd.to_numeric(marker_gdf[selected_param_col], errors="coerce")
    marker_gdf["_well_id"] = marker_gdf[well_col] if well_col else np.arange(1, len(marker_gdf) + 1)

    fmap = create_interpolation_map(
        wells_wgs84=marker_gdf,
        border_wgs84=border_wgs84,
        raster_rgba=raster_rgba,
        raster_bounds_wgs84=raster_bounds_wgs84,
        parameter_name=selected_param_ui,
        vmin=vmin,
        vmax=vmax,
        threshold_value=threshold_value,
        exceed_rgba=exceed_rgba,
    )

    tab_map, tab_diagrams = st.tabs(["Map", "Diagrams"])

    with tab_map:
        # Render folium map in Streamlit with a large fixed viewport.
        map_html = fmap.get_root().render()
        st.components.v1.html(map_html, height=780, scrolling=False)

        # Keep the table out of the main visual focus so the map remains dominant.
        with st.expander("Filtered Well Data (click to expand)", expanded=False):
            show_cols = [c for c in [well_col, lat_col, lon_col, selected_param_col] if c]
            st.dataframe(
                filtered[show_cols].reset_index(drop=True),
                width="stretch",
                height=260,
            )

    with tab_diagrams:
        st.subheader(f"Diagrams for {selected_param_ui}")

        chart_df = df_time.copy()
        chart_df[selected_param_col] = pd.to_numeric(chart_df[selected_param_col], errors="coerce")
        chart_df = chart_df.dropna(subset=[selected_param_col])

        # 1) Time trend of average concentration/value.
        time_mean = (
            chart_df.groupby("_time_label", as_index=True)[selected_param_col]
            .mean()
            .reindex(time_options)
        )
        st.markdown("**Average value over time**")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(time_mean.index, time_mean.values, color="#1f77b4", linewidth=2, marker="o", markersize=3)
        ax1.axhline(threshold_value, color="#d62728", linestyle="--", linewidth=2, label=f"Limit ({threshold_value:g})")
        ax1.set_ylabel(selected_param_ui)
        ax1.set_xlabel("Time")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right")
        st.pyplot(fig1, width="stretch")
        plt.close(fig1)

        # 2) Distribution at the selected timestamp.
        st.markdown(f"**Distribution at selected time: {selected_time}**")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        vals = interp_df[selected_param_col].to_numpy(dtype=float)
        ax2.hist(vals, bins=10, color="#4C78A8", edgecolor="white", alpha=0.85)
        ax2.axvline(threshold_value, color="#d62728", linestyle="--", linewidth=2, label=f"Limit ({threshold_value:g})")
        ax2.set_xlabel(selected_param_ui)
        ax2.set_ylabel("Count")
        ax2.grid(alpha=0.2)
        ax2.legend(loc="upper right")
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)

        # 3) Well ranking for selected timestamp (helps inspect hotspots).
        st.markdown("**Well ranking at selected time**")
        ranking_col = "_well_id"
        ranking_df = marker_gdf[[ranking_col, "_selected_value"]].copy()
        ranking_df = ranking_df.dropna(subset=["_selected_value"]).sort_values("_selected_value", ascending=False)
        ranking_df["status"] = np.where(ranking_df["_selected_value"] >= threshold_value, "Above limit", "Below limit")
        ranking_df = ranking_df.rename(columns={ranking_col: "well_id", "_selected_value": selected_param_ui})
        st.dataframe(ranking_df.reset_index(drop=True), width="stretch", height=280)
        above_count = int((ranking_df[selected_param_ui] >= threshold_value).sum())
        total_count = int(len(ranking_df))
        st.markdown(f"**Wells above limit:** {above_count} / {total_count}")

    st.caption(
        f"IDW settings: power={idw_power:.1f}, neighbors={idw_neighbors}, grid={grid_size}x{grid_size}, "
        f"smoothing sigma={smooth_sigma:.1f}. Limit={threshold_value:g}. "
        "Interpolation uses EPSG:3857 distances and is displayed in EPSG:4326."
    )


if __name__ == "__main__":
    main()
