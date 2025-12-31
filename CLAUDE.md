# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced Streamlit-based web application for visualizing time series data from meteorological/hydrological data files. The app reads whitespace-delimited text files with temporal data (year, month, day, hour columns), processes them into pandas DataFrames, and provides comprehensive interactive visualization and statistical analysis capabilities. It also supports NetCDF raster file visualization for gridded hydrological model outputs.

## Running the Application

```bash
# Windows - Activate virtual environment
.venv\Scripts\activate

# Linux/Mac - Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Architecture

### Main Application (app.py)

The application is a single-file Streamlit app (~1450 lines) with modular components:

1. **Data Reading**:
   - `read_data_file(file_path)`: Auto-detects and skips metadata rows by validating year values (1900-2100)
   - Handles cp1252/ISO-8859-1 encoding for European datasets
   - Parses YY, MM, DD, HH columns into datetime index
   - Missing values coded as -9999 are handled automatically
   - Extracts physical units from metadata (second line of file)
   - `read_netcdf_file(file_path)`: Reads NetCDF raster files using xarray

2. **Data Processing**:
   - `concatenate_dataframes(df_list)`: Merges multiple files along date index with intelligent column naming
   - Uses parameter descriptions from line 2 of files as column names (priority: parameter description > filename without extension > filename)
   - Handles duplicate column names by adding numeric suffixes only when necessary
   - Outer join preserves all data from all files
   - Helps users identify which columns came from which files when multiple files are uploaded

3. **Visualization Functions** (7 types for time series + NetCDF viewer):
   - `plot_timeseries(df, log_scale, title, unit)`: Raw time series plots
   - `plot_aggregated_data(df, freq, freq_name, log_scale, unit)`: Statistical aggregations with mean/min/max/std bands
   - `plot_rolling_statistics(df, window_days, log_scale, unit)`: Rolling mean and std deviation
   - `plot_seasonal_decomposition(df, columns, unit)`: Monthly boxplots and yearly trends
   - `plot_correlation_heatmap(df)`: Correlation matrix heatmap
   - `plot_model_comparison(df, col1, col2, log_scale, unit)`: Compare two time series with performance metrics (R², NSE, KGE)
   - `plot_multi_overview(df, selected_cols, log_scale, unit)`: 2-column subplot grid for multi-variable comparison

4. **Helper Functions**:
   - `fig_to_pdf_bytes(fig)`: Converts matplotlib figures to high-resolution PDF (300 DPI) for download

### Data Flow

**Time Series Files:**
1. User uploads one or multiple files via Streamlit file uploader
2. Files are temporarily saved to `temp_data/` directory (created if doesn't exist)
3. Each file is read using `read_data_file()` with auto-detection of metadata rows
4. Multiple dataframes are concatenated using `concatenate_dataframes()`
5. User selects columns and visualization options via interactive widgets
6. Plots are generated in tabbed interface (7 tabs for different analysis types)
7. Temporary files are cleaned up after processing

**NetCDF Files:**
1. User uploads NetCDF file (.nc) via separate uploader
2. File is read using `read_netcdf_file()` with xarray
3. WASIM grids are auto-detected and transposed (x, y, t coordinates)
4. User selects time layer and NoData masking value
5. Raster is visualized with cividis colormap and quantile statistics
6. PDF and CSV export available for selected layer

### Key Implementation Details

- **Temporary File Handling**: Files are saved to `temp_data/` during processing and cleaned up afterward. The directory is created if it doesn't exist.

- **Date Parsing**: Time series readers expect columns named YY, MM, DD, HH and parse them into a datetime index. The HH column is dropped after date creation.

- **Encoding**: Files use ISO-8859-1/cp1252 encoding to handle special characters in European datasets.

- **NetCDF Coordinate Detection**: The app detects WASIM model output grids by looking for 'x', 'y', 't' coordinates and transposes them to (t, y, x) for proper visualization.

## Dependencies

Core dependencies (from requirements.txt):
- **streamlit** (>=1.36.0): Web application framework
- **pandas** (>=2.2.2): Data manipulation and analysis
- **matplotlib** (>=3.9.0): Plotting library
- **seaborn** (>=0.13.2): Statistical visualization (for heatmaps)
- **numpy** (>=2.0.0): Numerical operations
- **xarray** (>=2024.7.0): NetCDF file handling
- **netCDF4** (>=1.7.2): NetCDF backend for xarray
- **hydroeval** (>=0.1.0): Hydrological model evaluation metrics (NSE, KGE)

The project uses a virtual environment located in `.venv/`.

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Development Notes

### Data Reader Function

- **`read_data_file(file_path)`**: Auto-detects metadata rows by validating year values (1900-2100)
  - Auto-detects separator type (tab or whitespace)
  - Scans lines after header to find first valid data row
  - Skips all metadata rows before first valid year
  - Works with files having variable numbers of metadata rows (3-5 typical)
  - Handles quoted column names and special characters
  - **Extracts parameter description from line 2**: Used for intelligent column naming when merging multiple files
  - **Extracts physical units from metadata**: Reads second line for unit information (e.g., "_in_mm", "evaporation")
  - **Stores filename information**: Both full filename and filename without extension
  - Stores all metadata in DataFrame.attrs for later use in plotting and column naming
  - Returns pandas DataFrame with datetime index
  - **Accepts any file format** (no extension restriction)

### Key Features (v3.0)

**Multiple File Support:**
- Upload and merge multiple files simultaneously
- Automatic concatenation along date index (outer join)
- **Intelligent column naming**: Uses parameter descriptions from line 2 of each file, or filename if not available
- Column names clearly indicate which file/parameter they came from (e.g., "precipitation_in_mm", "evaporation_layer1")
- Duplicate column names handled with numeric suffixes only when necessary

**Interactive Visualizations (7 types in tabs):**
1. **Time Series**: Raw data plots with log scale option
2. **Aggregated Stats**: Mean/Min/Max/StdDev bands over time (Daily/Weekly/Monthly/Yearly)
3. **Rolling Statistics**: Configurable rolling window (7-365 days) with mean and std bands
4. **Seasonal Patterns**: Monthly boxplots and yearly trend analysis
5. **Correlations**: Heatmap of column correlations
6. **Model Performance**: Compare two time series with R² (coefficient of determination), NSE (Nash-Sutcliffe Efficiency), and KGE (Kling-Gupta Efficiency) metrics using the hydroeval library
7. **Multi-Overview**: 2-column subplot grid for comparing multiple variables

**NetCDF Raster Viewer:**
- Upload and visualize NetCDF (.nc) files
- Auto-detect and transpose WASIM grids (x, y, t coordinates)
- Time layer selection with slider for multi-temporal datasets
- Interactive value masking (default: -9999 for NoData)
- Cividis colormap for better perceptual uniformity
- Quantile statistics display (0%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 100%)
- PDF and CSV export functionality
- Variable attribute display

**Professional Plotting Features:**
- **LaTeX-style fonts**: Computer Modern (serif) font family for publication-quality plots
- **Physical units on Y-axis**: Auto-extracted from file headers (e.g., "mm", "m³/s")
- **Clean design**: No redundant x-axis labels (dates/years/months are self-explanatory)
- **Mathematical notation**: Proper symbols like ±1σ for standard deviation
- **PDF export**: Each figure can be downloaded as high-resolution PDF (300 dpi)

**Analysis Tools:**
- Prominent log scale Y-axis toggle (applies to all plots)
- Column statistics table with missing value percentages
- Interactive column selection
- CSV export (complete or selected columns)

**UI Enhancements:**
- Wide layout for better visualization
- Tabbed interface for different analysis types
- Collapsible raw data preview
- Loading spinners and status messages
- Clean, modern design with seaborn styling

### File Format Requirements

Input files must have:
- First row: Column headers (YY MM DD HH [data columns])
- Rows 1-N: Metadata with non-year values in YY column (auto-detected and skipped)
- Data rows: Temporal data with numeric values
- Missing values coded as -9999 (or 0.0)
- Encoding: cp1252 or ISO-8859-1
- Separators: Tab-delimited or whitespace-delimited (auto-detected)
- **Any file extension** (no restriction on file type)

### Code Optimization Notes

**Performance:**
- Single reader function (`read_data_file()`) reduces code complexity
- Matplotlib figures explicitly closed after rendering to prevent memory leaks
- Seaborn styling applied globally for consistent appearance
- High-resolution PDF exports (300 DPI) for publication quality
- NetCDF files use xarray for efficient lazy loading of large raster datasets

**Plotting Enhancements:**
- LaTeX-style fonts configured via matplotlib rcParams:
  - Font family: serif (Times New Roman/DejaVu Serif)
  - Math fonts: Computer Modern (cm)
  - Consistent font sizes across all elements (12pt labels, 10pt ticks/legend)
- Unit extraction logic supports: mm, m, m³/s, precipitation, evaporation, discharge, flow
- All plots return figures for PDF export instead of direct rendering
- Cividis colormap used for NetCDF rasters (perceptually uniform, colorblind-friendly)

**Model Evaluation:**
- Uses hydroeval library for standard hydrological metrics
- NSE (Nash-Sutcliffe Efficiency): Ranges from -∞ to 1, where 1 is perfect fit
- KGE (Kling-Gupta Efficiency): Ranges from -∞ to 1, decomposed into correlation, bias, and variability
- R² (Coefficient of Determination): Ranges from 0 to 1, measures variance explained
