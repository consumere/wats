# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced Streamlit-based web application for visualizing time series data from meteorological/hydrological data files. The app reads whitespace-delimited text files with temporal data (year, month, day, hour columns), processes them into pandas DataFrames, and provides comprehensive interactive visualization and statistical analysis capabilities.

## Running the Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Architecture

### Main Application (app.py)

The application is a single-file Streamlit app (~440 lines) with optimized, modular components:

1. **Data Reading**:
   - `read_data_file(file_path)`: Auto-detects and skips metadata rows by validating year values (1900-2100)
   - Handles cp1252/ISO-8859-1 encoding
   - Parses YY, MM, DD columns into datetime index
   - Missing values coded as -9999 are handled automatically

2. **Data Processing**:
   - `concatenate_dataframes(df_list)`: Merges multiple files along date index
   - Handles duplicate column names by adding numeric suffixes
   - Outer join preserves all data from all files

3. **Visualization Functions** (5 types):
   - `plot_timeseries(df, log_scale, title)`: Raw time series plots
   - `plot_aggregated_data(df, freq, freq_name, log_scale)`: Statistical aggregations with mean/min/max/std bands
   - `plot_rolling_statistics(df, window_days, log_scale)`: Rolling mean and std deviation
   - `plot_seasonal_decomposition(df, column)`: Monthly boxplots and yearly trends
   - `plot_correlation_heatmap(df)`: Correlation matrix heatmap

### Data Flow

1. User uploads one or multiple files via Streamlit file uploader
2. Files are temporarily saved to `temp_uploads/` directory
3. Each file is read using `read_data_file()` with auto-detection of metadata rows
4. Multiple dataframes are concatenated using `concatenate_dataframes()`
5. User selects columns and visualization options via interactive widgets
6. Plots are generated in tabbed interface (5 tabs for different analysis types)
7. Temporary files are cleaned up after processing

### Key Implementation Details

- **Active Reader Selection**: The current implementation uses `wain()` at line 110. Previous versions used `waread3()` and `watab()` (commented out at lines 108-109). When modifying data reading logic, ensure the correct reader is uncommented.

- **Temporary File Handling**: Files are saved to `temp_data/` during processing and cleaned up afterward. The directory is created if it doesn't exist.

- **Date Parsing**: All readers expect columns named YY, MM, DD, HH and parse them into a datetime index. The HH column is dropped after date creation.

- **Encoding**: Files use ISO-8859-1/cp1252 encoding to handle special characters in European datasets.

## Dependencies

Core dependencies (install via pip):
- streamlit: Web application framework
- pandas: Data manipulation and analysis
- matplotlib: Plotting library
- seaborn: Statistical visualization (for heatmaps)
- numpy: Numerical operations

The project uses Python 3.9 with a virtual environment located in `.venv/`.

Install all dependencies:
```bash
source .venv/bin/activate
pip install streamlit pandas matplotlib seaborn numpy
```

## Development Notes

### Data Reader Function

- **`read_data_file(file_path)`**: Auto-detects metadata rows by validating year values (1900-2100)
  - Auto-detects separator type (tab or whitespace)
  - Scans lines after header to find first valid data row
  - Skips all metadata rows before first valid year
  - Works with files having variable numbers of metadata rows (3-5 typical)
  - Handles quoted column names and special characters
  - **Extracts physical units from metadata**: Reads second line for unit information (e.g., "_in_mm", "evaporation")
  - Stores metadata in DataFrame.attrs for later use in plotting
  - Returns pandas DataFrame with datetime index
  - **Accepts any file format** (no extension restriction)

### Key Features (v3.0)

**Multiple File Support:**
- Upload and merge multiple files simultaneously
- Automatic concatenation along date index (outer join)
- Duplicate column names handled with numeric suffixes

**Interactive Visualizations (7 types in tabs):**
1. **Time Series**: Raw data plots with log scale option
2. **Aggregated Stats**: Mean/Min/Max/StdDev bands over time (Daily/Weekly/Monthly/Yearly)
3. **Rolling Statistics**: Configurable rolling window (7-365 days) with mean and std bands
4. **Seasonal Patterns**: Monthly boxplots and yearly trend analysis
5. **Correlations**: Heatmap of column correlations
6. **Model Performance**: Compare two time series with R², NSE, KGE metrics
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

**Removed Functions:**
- Removed `waread3()` and `watab()` - replaced by single optimized `read_data_file()`
- Removed `app_bak.py` backup file

**Performance:**
- Single reader function reduces code complexity
- Matplotlib figures explicitly closed after rendering to prevent memory leaks
- Seaborn styling applied globally for consistent appearance
- High-resolution PDF exports (300 dpi) for publication

**Plotting Enhancements:**
- LaTeX-style fonts configured via matplotlib rcParams:
  - Font family: serif (Times New Roman/DejaVu Serif)
  - Math fonts: Computer Modern (cm)
  - Consistent font sizes across all elements
- Unit extraction logic supports: mm, m, m³/s, precipitation, evaporation, discharge
- All plots return figures for PDF export instead of direct rendering
