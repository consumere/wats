# WATS - Water & Atmospheric Time Series Viewer

An advanced Streamlit-based web application for visualizing and analyzing time series data from meteorological and hydrological model outputs.

## Features

### 📁 Data Import
- **Multiple file upload**: Load and merge multiple files simultaneously
- **Auto-detection**: Automatically detects metadata rows, separators (tab/whitespace), and file formats
- **Flexible format**: No file extension required - handles any delimited text file
- **Unit extraction**: Automatically extracts physical units from file headers (mm, m³/s, etc.)

### 📊 Visualization Types
1. **Time Series**: Raw data plots with interactive column selection
2. **Aggregated Statistics**: Mean/Min/Max/StdDev bands (Daily/Weekly/Monthly/Yearly)
3. **Rolling Statistics**: Configurable moving averages with uncertainty bands (7-365 days)
4. **Seasonal Patterns**: Monthly boxplots and yearly trend analysis
5. **Correlation Analysis**: Interactive heatmap of variable correlations

### 🎨 Professional Output
- **LaTeX-style fonts**: Publication-ready plots with Computer Modern fonts
- **Physical units**: Automatic Y-axis labeling with extracted units
- **Clean design**: No redundant labels, optimized for clarity
- **Mathematical notation**: Proper symbols (±1σ)
- **PDF export**: High-resolution (300 DPI) PDF download for each plot

### 📈 Analysis Features
- Logarithmic scale toggle (applies to all plots)
- Interactive column selection
- Comprehensive statistics table with missing value tracking
- CSV export (complete dataset or selected columns)

## Running Locally

```bash
# Clone the repository
git clone https://github.com/consumere/wats.git
cd wats

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## File Format

Input files should have:
- **First row**: Column headers (YY MM DD HH [data columns...])
- **Metadata rows**: Auto-detected and skipped
- **Data rows**: Temporal data with numeric values
- **Missing values**: Coded as -9999 or 0.0
- **Encoding**: cp1252 or ISO-8859-1
- **Separators**: Tab or whitespace (auto-detected)

Example:
```
YY	MM	DD	HH	Temperature	Precipitation
interception_evaporation_in_mm_layer_1
2000	1	1	24	15.2	2.3
2000	1	2	24	14.8	0.0
```

## Technologies

- **Streamlit**: Interactive web framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: High-quality plotting
- **Seaborn**: Statistical visualizations
- **NumPy**: Numerical operations

## License

MIT License

## Author

Created for hydrological and meteorological data analysis.
