# ZHZHZHKISSBENCEANDRAS

Streamlit dashboard for **Programming for Geoinformatics** that visualizes well measurements and performs **IDW spatial interpolation** inside a border shapefile.

## Features
- Interactive map with OpenStreetMap basemap
- 28 well points displayed as markers
- IDW interpolation for selected parameter and timestamp
- Interpolation clipped to `SHP/border.shp` boundary
- Exceedance/limit highlighting on map
- Diagrams tab with trends, distribution, and well ranking

## Project Structure
```text
.
├── main.py
├── requirements.txt
├── Szeged_with_coordinates.csv
└── SHP/
    ├── border.shp
    ├── border.shx
    ├── border.dbf
    └── border.prj
```

## Required Libraries
From `requirements.txt`:
- streamlit
- pandas
- geopandas
- leafmap
- scipy
- numpy
- matplotlib
- folium

## Prerequisites
- Python 3.10+ (tested with Python 3.14 on Windows)
- `pip` package manager
- Internet access for first package installation

## Installation
From the project root folder:

```powershell
python -m pip install -r requirements.txt
```

If you have multiple Python versions, use the exact interpreter path (example):

```powershell
C:\Users\kissb\AppData\Local\Programs\Python\Python314\python.exe -m pip install -r requirements.txt
```

## Run the Dashboard
Start Streamlit from the project root:

```powershell
python -m streamlit run main.py
```

Then open the URL shown in terminal (usually `http://localhost:8501`).

## Important Note
Run with `streamlit run`, not `python main.py`.

Wrong:
```powershell
python main.py
```

Correct:
```powershell
python -m streamlit run main.py
```

## Data/CRS Notes
- CSV well coordinates are used as WGS84 (`EPSG:4326`)
- Border shapefile is reprojected to WGS84 for web display
- IDW distances are computed in metric CRS (`EPSG:3857`)

## Troubleshooting
- **`ModuleNotFoundError`**: reinstall dependencies with the same Python interpreter used to run Streamlit.
- **Map not loading / no points visible**: check coordinate columns and CRS conversion.
- **Shapefile errors**: make sure all four files exist: `.shp`, `.shx`, `.dbf`, `.prj`.
- **Port already in use**: run with another port:
  ```powershell
  python -m streamlit run main.py --server.port 8502
  ```
