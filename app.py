import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import xarray as xr
from hydroeval import evaluator, nse, kge

# Configure Streamlit page
st.set_page_config(
    page_title="Time Series Data Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set plotting style with LaTeX fonts
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern (LaTeX default)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def read_data_file(file_path):
    """Read data file with automatic metadata detection.

    Auto-detects and skips metadata rows (rows containing non-numeric year values).
    Handles both whitespace and tab-separated files.
    Returns DataFrame and metadata dictionary with units.
    """
    with open(file_path, 'r', encoding='cp1252') as f:
        lines = f.readlines()

    # Detect separator (tab or whitespace)
    first_line = lines[0]
    if '\t' in first_line:
        sep = '\t'
    else:
        sep = r'\s+'

    # Extract metadata for units/descriptions
    metadata = {}
    if len(lines) > 1:
        # Check second line for unit information
        second_line = lines[1].strip()
        if second_line and not second_line[0].isdigit():
            metadata['description'] = second_line
            # Try to extract units from description
            if '_in_mm' in second_line.lower():
                metadata['unit'] = 'mm'
            elif '_in_m' in second_line.lower():
                metadata['unit'] = 'm'
            elif 'evaporation' in second_line.lower():
                metadata['unit'] = 'mm'
            elif 'discharge' in second_line.lower() or 'flow' in second_line.lower():
                metadata['unit'] = 'm³/s'
            elif 'precipitation' in second_line.lower() or 'rain' in second_line.lower():
                metadata['unit'] = 'mm'

    # Find rows to skip (metadata rows where YY column is not a 4-digit year)
    skip_rows = []
    for i, line in enumerate(lines[1:], start=1):  # Skip first line (header)
        if sep == '\t':
            parts = line.split('\t')
        else:
            parts = line.split()

        if len(parts) >= 4:
            try:
                # Clean the first part (remove quotes, dashes, etc.)
                year_str = parts[0].strip().strip('"').strip("'").replace('--', '')
                if not year_str or year_str == '--':
                    skip_rows.append(i)
                    continue

                year = int(year_str)
                if year < 1900 or year > 2100:
                    skip_rows.append(i)
                else:
                    break  # Found first data row, stop checking
            except (ValueError, AttributeError):
                skip_rows.append(i)
        else:
            skip_rows.append(i)

    # Read the file with detected skip rows
    df = pd.read_table(file_path,
                       sep=sep,
                       engine='c',
                       skiprows=skip_rows if skip_rows else None,
                       na_values=[-9999, 0.0],
                       skip_blank_lines=True,
                       encoding='cp1252',
                       low_memory=False,
                       parse_dates={'date': ['YY', 'MM', 'DD']})

    # Drop HH column if it exists
    if 'HH' in df.columns:
        df.drop(labels='HH', axis=1, inplace=True)

    df.set_index('date', inplace=True)

    # Store metadata as attribute
    df.attrs['metadata'] = metadata

    return df


def concatenate_dataframes(df_list):
    """Concatenate multiple dataframes and handle overlapping columns."""
    if len(df_list) == 1:
        return df_list[0]

    # Concatenate along columns (join on date index)
    combined_df = pd.concat(df_list, axis=1, join='outer')

    # Handle duplicate column names by adding numeric suffix
    cols = pd.Series(combined_df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup
                                                           for i in range(sum(cols == dup))]
    combined_df.columns = cols

    return combined_df.sort_index()


def plot_timeseries(df, log_scale=False, title="Time Series Data", unit=None):
    """Plot time series data with improved styling."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for column in df.columns:
        df[column].plot(ax=ax, label=column, linewidth=1.5, alpha=0.8)

    ax.set_yscale('log' if log_scale else 'linear')

    # No x-label (dates are self-explanatory from ticks)
    ax.set_xlabel('')

    # Y-label with unit if available
    if unit:
        ylabel = f'Value [{unit}]' + (' (log)' if log_scale else '')
    else:
        ylabel = 'Value' + (' (log)' if log_scale else '')
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='best', frameon=True, shadow=False, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    plt.tight_layout()

    return fig


def plot_aggregated_data(df, freq, freq_name, log_scale=False, unit=None):
    """Compute and plot aggregated data with statistics."""
    aggregated_df = df.resample(freq).agg(['mean', 'min', 'max', 'std'])

    num_columns = len(df.columns)
    fig, axes = plt.subplots(num_columns, 1, figsize=(14, 4 * num_columns), squeeze=False)

    for idx, column in enumerate(df.columns):
        ax = axes[idx, 0]

        # Plot mean with error band (std)
        mean_data = aggregated_df[column]['mean']
        std_data = aggregated_df[column]['std']
        min_data = aggregated_df[column]['min']
        max_data = aggregated_df[column]['max']

        ax.plot(mean_data.index, mean_data.values, label='Mean', linewidth=2, marker='o', markersize=4)
        ax.fill_between(mean_data.index,
                        mean_data - std_data,
                        mean_data + std_data,
                        alpha=0.3, label=r'$\pm 1\sigma$')
        ax.plot(min_data.index, min_data.values, '--', alpha=0.5, label='Min', linewidth=1)
        ax.plot(max_data.index, max_data.values, '--', alpha=0.5, label='Max', linewidth=1)

        ax.set_yscale('log' if log_scale else 'linear')
        ax.set_xlabel('')  # No x-label

        # Y-label with unit
        if unit:
            ylabel = f'{column} [{unit}]' + (' (log)' if log_scale else '')
        else:
            ylabel = f'{column}' + (' (log)' if log_scale else '')
        ax.set_ylabel(ylabel, fontsize=11)

        ax.set_title(f'{freq_name} Aggregation', fontsize=12)
        ax.legend(loc='best', frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_rolling_statistics(df, window_days, log_scale=False, unit=None):
    """Plot rolling mean and standard deviation."""
    num_columns = len(df.columns)
    fig, axes = plt.subplots(num_columns, 1, figsize=(14, 4 * num_columns), squeeze=False)

    for idx, column in enumerate(df.columns):
        ax = axes[idx, 0]

        rolling_mean = df[column].rolling(window=window_days, center=True).mean()
        rolling_std = df[column].rolling(window=window_days, center=True).std()

        # Plot original data
        ax.plot(df.index, df[column], alpha=0.3, linewidth=0.8, label='Original')

        # Plot rolling mean
        ax.plot(rolling_mean.index, rolling_mean.values, linewidth=2.5,
                label=f'{window_days}-day Moving Avg.', color='red')

        # Plot rolling std as shaded area
        ax.fill_between(rolling_mean.index,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2, color='red', label=r'$\pm 1\sigma$')

        ax.set_yscale('log' if log_scale else 'linear')
        ax.set_xlabel('')  # No x-label

        # Y-label with unit
        if unit:
            ylabel = f'{column} [{unit}]' + (' (log)' if log_scale else '')
        else:
            ylabel = f'{column}' + (' (log)' if log_scale else '')
        ax.set_ylabel(ylabel, fontsize=11)

        ax.set_title(f'Rolling Statistics', fontsize=12)
        ax.legend(loc='best', frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_seasonal_decomposition(df, columns, unit=None):
    """Plot yearly patterns using box plots for multiple columns."""
    # Limit to 10 columns max
    if len(columns) > 10:
        st.warning(f"⚠️ Too many columns ({len(columns)}). Plotting first 10 columns only.")
        columns = columns[:10]

    n_cols = len(columns)
    if n_cols == 0:
        return None

    # Create grid layout: each column gets 1 row with 2 subplots
    fig, axes = plt.subplots(n_cols, 2, figsize=(14, 5 * n_cols), squeeze=False)
    fig.suptitle('Seasonal Patterns Analysis', fontsize=16, y=0.995)

    for idx, column in enumerate(columns):
        data = df[column].dropna()

        if len(data) == 0:
            axes[idx, 0].text(0.5, 0.5, f'No data for {column}',
                            ha='center', va='center', fontsize=12)
            axes[idx, 1].text(0.5, 0.5, f'No data for {column}',
                            ha='center', va='center', fontsize=12)
            continue

        # Create year and month columns
        plot_df = pd.DataFrame({
            'value': data.values,
            'month': data.index.month,
            'year': data.index.year
        })

        # Monthly boxplot
        monthly_data = [plot_df[plot_df['month'] == m]['value'].values for m in range(1, 13)]
        bp1 = axes[idx, 0].boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                               patch_artist=True, showmeans=True)

        for patch in bp1['boxes']:
            patch.set_facecolor(f'C{idx}')
            patch.set_alpha(0.7)

        axes[idx, 0].set_xlabel('')  # No x-label (months are clear)
        if unit:
            axes[idx, 0].set_ylabel(f'{column} [{unit}]', fontsize=10)
        else:
            axes[idx, 0].set_ylabel(f'{column}', fontsize=10)
        axes[idx, 0].set_title(f'Monthly Distribution - {column}', fontsize=11)
        axes[idx, 0].grid(True, alpha=0.3, axis='y')

        # Yearly trend
        yearly_mean = plot_df.groupby('year')['value'].mean()
        axes[idx, 1].plot(yearly_mean.index, yearly_mean.values, marker='o',
                         linewidth=2, markersize=6, color=f'C{idx}')
        axes[idx, 1].fill_between(yearly_mean.index, yearly_mean.values, alpha=0.3, color=f'C{idx}')
        axes[idx, 1].set_xlabel('')  # No x-label (years are clear)
        if unit:
            axes[idx, 1].set_ylabel(f'Mean [{unit}]', fontsize=10)
        else:
            axes[idx, 1].set_ylabel(f'Mean Value', fontsize=10)
        axes[idx, 1].set_title(f'Yearly Trend - {column}', fontsize=11)
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Plot correlation heatmap of all columns."""
    if len(df.columns) < 2:
        st.info("Correlation heatmap requires at least 2 columns.")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate correlation matrix
    corr = df.corr()

    # Create heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title('Correlation Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    return fig


def fig_to_pdf_bytes(fig):
    """Convert matplotlib figure to PDF bytes for download."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf.getvalue()


def read_netcdf_file(file_path):
    """Read NetCDF file using xarray."""
    try:
        ds = xr.open_dataset(file_path)
        return ds
    except Exception as e:
        st.error(f"Error reading NetCDF file: {e}")
        return None


def plot_model_comparison(df, col1, col2, log_scale=False, unit=None):
    """Plot model vs observation comparison with statistics (like fplot4.py)."""
    # Calculate statistics
    valid_data = df[[col1, col2]].dropna()

    if len(valid_data) < 2:
        st.warning("Not enough valid data points for comparison")
        return None

    # Pearson correlation
    corr_matrix = valid_data.corr()
    pearson_r = corr_matrix.iloc[0, 1]
    r2 = pearson_r ** 2

    # NSE and KGE
    try:
        nse_score = evaluator(nse, valid_data[col1].values, valid_data[col2].values)
        nse_val = nse_score[0].item() if hasattr(nse_score[0], 'item') else nse_score[0]
    except:
        nse_val = np.nan

    try:
        kge_score = evaluator(kge, valid_data[col1].values, valid_data[col2].values)
        kge_val = kge_score[0].item() if hasattr(kge_score[0], 'item') else kge_score[0]
    except:
        kge_val = np.nan

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot data
    df[col1].plot(ax=ax, style='b--', linewidth=1.5, label=col1)
    df[col2].plot(ax=ax, style='r-', linewidth=1.5, label=col2)

    ax.set_yscale('log' if log_scale else 'linear')
    ax.set_xlabel('')

    if unit:
        ax.set_ylabel(f'Value [{unit}]' + (' (log)' if log_scale else ''), fontsize=12)
    else:
        ax.set_ylabel('Value' + (' (log)' if log_scale else ''), fontsize=12)

    # Title with statistics
    title = f'Model Performance Comparison'
    subtitle = f'Pearson R²: {r2:.3f} | NSE: {nse_val:.3f} | Kling-Gupta Efficiency: {kge_val:.3f}'

    ax.set_title(subtitle, fontsize=14, pad=10)
    fig.suptitle(title, fontsize=16, y=0.995)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

    plt.tight_layout()
    return fig


def plot_multi_overview(df, selected_cols, log_scale=False, unit=None):
    """Create overview plot with multiple subplots for selected columns."""
    n_cols = len(selected_cols)
    if n_cols == 0:
        return None

    # Calculate grid layout
    n_rows = int(np.ceil(n_cols / 2))
    n_plot_cols = min(2, n_cols)

    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(14, 4 * n_rows), squeeze=False)
    fig.suptitle('Multi-Variable Overview', fontsize=16, y=0.995)

    for idx, col in enumerate(selected_cols):
        row = idx // 2
        col_idx = idx % 2
        ax = axes[row, col_idx]

        df[col].plot(ax=ax, linewidth=1.5, color=f'C{idx}')
        ax.set_yscale('log' if log_scale else 'linear')
        ax.set_xlabel('')

        if unit:
            ax.set_ylabel(f'{col} [{unit}]', fontsize=10)
        else:
            ax.set_ylabel(col, fontsize=10)

        # Add basic statistics
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.set_title(f'{col} | Mean: {mean_val:.2f} | Std: {std_val:.2f}', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

    # Hide empty subplots
    if n_cols % 2 != 0 and n_cols > 1:
        axes[n_rows-1, 1].axis('off')

    plt.tight_layout()
    return fig


# Streamlit app
st.title("📊 WATS - Water & Atmospheric Time Series Viewer")
st.markdown("""
Advanced visualization tool for hydrological and meteorological data.
Upload time series files (text/CSV) or raster data (NetCDF).
""")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    **File Format:**
    - Whitespace-delimited text file
    - Header row with column names
    - Columns: YY MM DD HH [data columns...]
    - Missing values coded as -9999
    - Metadata rows automatically detected

    **Features:**
    - ✅ Multiple file upload & merge
    - 📈 Interactive visualizations
    - 📊 Statistical aggregations
    - 🔄 Rolling statistics
    - 📅 Seasonal analysis
    - 🔗 Correlation analysis
    - 📈 Model performance (R², NSE, KGE)
    - 🗺️ Multi-variable overview
    - 🌐 NetCDF raster viewer
    - 📥 PDF/CSV export
    """)

    st.header("ℹ️ About")
    st.info("WATS v3.0\n\nOptimized for hydrological and meteorological data analysis")

# Main content with tabs for different data types
main_tab1, main_tab2 = st.tabs(["📈 Time Series Data", "🌐 NetCDF Raster Data"])

# ====================== TIME SERIES TAB ======================
with main_tab1:
    # File upload - MULTIPLE FILES (NO TYPE RESTRICTION)
    uploaded_files = st.file_uploader(
        "Upload your time series data file(s)",
        accept_multiple_files=True,
        help="Upload one or more delimited files with temporal data (any file type)",
        key="ts_upload"
    )

if uploaded_files:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Read all uploaded files
        df_list = []
        file_names = []

        with st.spinner('Loading and processing files...'):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                df_temp = read_data_file(file_path)
                df_list.append(df_temp)
                file_names.append(uploaded_file.name)

                # Clean up individual file
                os.remove(file_path)

        # Concatenate all dataframes
        df = concatenate_dataframes(df_list)

        # Extract unit from first dataframe with metadata
        unit = None
        for temp_df in df_list:
            if hasattr(temp_df, 'attrs') and 'metadata' in temp_df.attrs:
                if 'unit' in temp_df.attrs['metadata']:
                    unit = temp_df.attrs['metadata']['unit']
                    break

        st.success(f"✅ Successfully loaded {len(uploaded_files)} file(s): {', '.join(file_names)}")

        if df is not None and not df.empty:
            # Data statistics
            st.subheader("📈 Data Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Files Loaded", len(uploaded_files))
            with col2:
                st.metric("Records", f"{len(df):,}")
            with col3:
                st.metric("Start Date", f"{df.index.min().date()}")
            with col4:
                st.metric("End Date", f"{df.index.max().date()}")
            with col5:
                years = (df.index.max() - df.index.min()).days / 365.25
                st.metric("Duration", f"{years:.1f} years")

            # Column statistics
            st.subheader("📊 Column Statistics")
            stats_df = df.describe().T
            stats_df['missing_pct'] = (df.isna().sum() / len(df) * 100).values
            stats_df = stats_df[['count', 'mean', 'std', 'min', 'max', 'missing_pct']]
            stats_df.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Missing %']
            st.dataframe(stats_df.style.format({
                'Mean': '{:.3f}',
                'Std Dev': '{:.3f}',
                'Min': '{:.3f}',
                'Max': '{:.3f}',
                'Missing %': '{:.1f}%'
            }), use_container_width=True)

            # Display raw data
            with st.expander("🔍 View Raw Data Preview (first 100 rows)", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)

            st.markdown("---")

            # Column selector
            st.subheader("📈 Visualization Options")
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to plot",
                all_columns,
                default=all_columns[:3] if len(all_columns) > 3 else all_columns
            )

            # Log scale toggle - PROMINENT
            col_a, col_b = st.columns([1, 4])
            with col_a:
                log_scale = st.checkbox("📊 Log Scale Y-Axis", value=False)
            with col_b:
                if log_scale:
                    st.info("🔍 Logarithmic scale is active for all plots")

            if selected_columns:
                df_selected = df[selected_columns]

                # Tab layout for different visualizations
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "📉 Time Series",
                    "📊 Aggregated Stats",
                    "🔄 Rolling Statistics",
                    "📅 Seasonal Patterns",
                    "🔗 Correlations",
                    "📈 Model Performance",
                    "🗺️ Multi-Overview"
                ])

                with tab1:
                    st.markdown("### Raw Time Series Data")
                    fig1 = plot_timeseries(df_selected, log_scale, "Complete Time Series", unit)
                    st.pyplot(fig1)
                    plt.close(fig1)

                    # PDF export button
                    pdf_bytes1 = fig_to_pdf_bytes(fig1)
                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_bytes1,
                        file_name="timeseries.pdf",
                        mime="application/pdf",
                        key="pdf_tab1"
                    )

                with tab2:
                    st.markdown("### Aggregated Statistics Over Time")
                    aggregation_type = st.selectbox(
                        "Select aggregation period",
                        ["Daily", "Weekly", "Monthly", "Yearly"],
                        index=2
                    )
                    freq_map = {
                        "Daily": ('D', 'Daily'),
                        "Weekly": ('W', 'Weekly'),
                        "Monthly": ('ME', 'Monthly'),
                        "Yearly": ('YE', 'Yearly')
                    }
                    freq, freq_name = freq_map[aggregation_type]
                    fig2 = plot_aggregated_data(df_selected, freq, freq_name, log_scale, unit)
                    st.pyplot(fig2)
                    plt.close(fig2)

                    # PDF export button
                    pdf_bytes2 = fig_to_pdf_bytes(fig2)
                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_bytes2,
                        file_name=f"aggregated_{freq_name.lower()}.pdf",
                        mime="application/pdf",
                        key="pdf_tab2"
                    )

                with tab3:
                    st.markdown("### Rolling Window Statistics")
                    window_days = st.slider(
                        "Rolling window size (days)",
                        min_value=7,
                        max_value=365,
                        value=30,
                        step=7
                    )
                    fig3 = plot_rolling_statistics(df_selected, window_days, log_scale, unit)
                    st.pyplot(fig3)
                    plt.close(fig3)

                    # PDF export button
                    pdf_bytes3 = fig_to_pdf_bytes(fig3)
                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_bytes3,
                        file_name=f"rolling_{window_days}days.pdf",
                        mime="application/pdf",
                        key="pdf_tab3"
                    )

                with tab4:
                    st.markdown("### Seasonal Patterns Analysis")
                    st.markdown("Monthly distribution and yearly trends for all selected columns")
                    if len(selected_columns) > 0:
                        fig4 = plot_seasonal_decomposition(df, selected_columns, unit)
                        if fig4:
                            st.pyplot(fig4)
                            plt.close(fig4)

                            # PDF export button
                            pdf_bytes4 = fig_to_pdf_bytes(fig4)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes4,
                                file_name=f"seasonal_patterns_all.pdf",
                                mime="application/pdf",
                                key="pdf_tab4"
                            )
                    else:
                        st.warning("Please select at least one column.")

                with tab5:
                    st.markdown("### Correlation Analysis")
                    if len(selected_columns) >= 2:
                        fig5 = plot_correlation_heatmap(df_selected)
                        if fig5:
                            st.pyplot(fig5)
                            plt.close(fig5)

                            # PDF export button
                            pdf_bytes5 = fig_to_pdf_bytes(fig5)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes5,
                                file_name="correlation_matrix.pdf",
                                mime="application/pdf",
                                key="pdf_tab5"
                            )
                    else:
                        st.info("Please select at least 2 columns to view correlations.")

                with tab6:
                    st.markdown("### Model Performance Comparison")
                    st.markdown("Compare two time series (simulated vs observed) with statistical metrics")

                    if len(selected_columns) >= 2:
                        # Auto-detect Sim and Obs columns
                        # Sim: starts with 'C' followed by integer (e.g., C4, C10)
                        # Obs: anything else (usually labeled as OBS or similar)
                        import re

                        sim_cols = [c for c in selected_columns if re.match(r'^C\d+', c)]
                        obs_cols = [c for c in selected_columns if c not in sim_cols]

                        # Set defaults
                        default_sim = sim_cols[0] if sim_cols else selected_columns[0]
                        default_obs = obs_cols[0] if obs_cols else (selected_columns[1] if len(selected_columns) > 1 else selected_columns[0])

                        # Get index for defaults
                        try:
                            sim_idx = selected_columns.index(default_sim)
                        except ValueError:
                            sim_idx = 0

                        col_compare1, col_compare2 = st.columns(2)
                        with col_compare1:
                            compare_col1 = st.selectbox("First variable (Simulated)",
                                                       selected_columns,
                                                       index=sim_idx,
                                                       key="comp1",
                                                       help="Should be simulation (e.g., C4, C10)")
                        with col_compare2:
                            # Filter out selected sim column
                            obs_options = [c for c in selected_columns if c != compare_col1]
                            # Try to default to obs column
                            try:
                                obs_idx = obs_options.index(default_obs) if default_obs in obs_options else 0
                            except (ValueError, IndexError):
                                obs_idx = 0

                            compare_col2 = st.selectbox("Second variable (Observed)",
                                                        obs_options,
                                                        index=obs_idx,
                                                        key="comp2",
                                                        help="Should be observation (e.g., OBS)")

                        fig6 = plot_model_comparison(df, compare_col1, compare_col2, log_scale, unit)
                        if fig6:
                            st.pyplot(fig6)
                            plt.close(fig6)

                            # PDF export
                            pdf_bytes6 = fig_to_pdf_bytes(fig6)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes6,
                                file_name=f"model_comparison_{compare_col1}_vs_{compare_col2}.pdf",
                                mime="application/pdf",
                                key="pdf_tab6"
                            )
                    else:
                        st.info("Please select at least 2 columns for comparison")

                with tab7:
                    st.markdown("### Multi-Variable Overview")
                    st.markdown("Compare multiple variables in subplot grid")

                    if len(selected_columns) > 0:
                        fig7 = plot_multi_overview(df, selected_columns, log_scale, unit)
                        if fig7:
                            st.pyplot(fig7)
                            plt.close(fig7)

                            # PDF export
                            pdf_bytes7 = fig_to_pdf_bytes(fig7)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes7,
                                file_name="multi_overview.pdf",
                                mime="application/pdf",
                                key="pdf_tab7"
                            )

                # Download section
                st.markdown("---")
                st.subheader("💾 Download Processed Data")
                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    csv = df.to_csv()
                    st.download_button(
                        label="📥 Download Complete Dataset (CSV)",
                        data=csv,
                        file_name="merged_timeseries_data.csv",
                        mime="text/csv"
                    )

                with col_dl2:
                    csv_selected = df_selected.to_csv()
                    st.download_button(
                        label="📥 Download Selected Columns (CSV)",
                        data=csv_selected,
                        file_name="selected_columns_data.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ Please select at least one column to visualize.")
        else:
            st.error("❌ The uploaded file(s) could not be processed. Please check the file format.")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass  # Directory not empty, that's ok

# ====================== NETCDF TAB ======================
with main_tab2:
    st.markdown("### NetCDF Raster Data Viewer & Comparison")
    st.markdown("Upload multiple NetCDF (.nc) files to compare model outputs side-by-side")

    nc_uploaded_files = st.file_uploader(
        "Upload NetCDF file(s)",
        type=['nc', 'nc4', 'netcdf'],
        accept_multiple_files=True,
        help="Upload one or more NetCDF files for visualization and comparison",
        key="nc_upload"
    )

    if nc_uploaded_files:
        temp_dir_nc = "temp_nc"
        os.makedirs(temp_dir_nc, exist_ok=True)

        try:
            # Save all uploaded files
            nc_paths = []
            datasets = []

            for nc_file in nc_uploaded_files:
                nc_path = os.path.join(temp_dir_nc, nc_file.name)
                with open(nc_path, "wb") as f:
                    f.write(nc_file.getbuffer())
                nc_paths.append(nc_path)

                # Read NetCDF file
                ds = read_netcdf_file(nc_path)
                if ds is not None:
                    datasets.append({'name': nc_file.name, 'path': nc_path, 'ds': ds})

            if len(datasets) > 0:
                st.success(f"✅ Successfully loaded {len(datasets)} NetCDF file(s)")

                # Check if dimensions are compatible for stacking
                if len(datasets) > 1:
                    st.markdown("---")
                    st.markdown("#### 📊 Multi-File Comparison")

                    # Check dimension compatibility
                    first_dims = set(datasets[0]['ds'].dims.keys())
                    all_compatible = all(set(ds['ds'].dims.keys()) == first_dims for ds in datasets)

                    if all_compatible:
                        st.success(f"✅ All {len(datasets)} files have compatible dimensions: {list(first_dims)}")

                        # Try to stack datasets
                        try:
                            stacked_ds = xr.concat([ds['ds'] for ds in datasets], dim='file')
                            stacked_ds.coords['file'] = [ds['name'] for ds in datasets]
                            st.info(f"📚 Files stacked along 'file' dimension")
                        except Exception as e:
                            st.warning(f"⚠️ Could not stack datasets: {e}. Will display separately.")
                            stacked_ds = None
                    else:
                        st.warning(f"⚠️ Files have incompatible dimensions. Will display separately.")
                        dim_info = "\n".join([f"- **{ds['name']}**: {list(ds['ds'].dims.keys())}" for ds in datasets])
                        st.markdown(dim_info)
                        stacked_ds = None

                # For simplicity, work with first dataset for metadata display
                ds = datasets[0]['ds']

                # Display dataset information
                st.markdown("---")
                st.subheader("📋 Dataset Information (First File)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dimensions", len(ds.dims))
                with col2:
                    st.metric("Variables", len(ds.data_vars))
                with col3:
                    st.metric("Attributes", len(ds.attrs))

                # Show dimensions
                with st.expander("📐 Dimensions", expanded=True):
                    dim_df = pd.DataFrame({
                        'Dimension': list(ds.dims.keys()),
                        'Size': list(ds.dims.values())
                    })
                    st.dataframe(dim_df, use_container_width=True)

                # Show variables
                with st.expander("📊 Variables", expanded=True):
                    var_list = []
                    for var_name in ds.data_vars:
                        var = ds[var_name]
                        var_list.append({
                            'Variable': var_name,
                            'Dimensions': str(var.dims),
                            'Shape': str(var.shape),
                            'Type': str(var.dtype)
                        })
                    var_df = pd.DataFrame(var_list)
                    st.dataframe(var_df, use_container_width=True)

                # Show global attributes
                with st.expander("ℹ️ Global Attributes"):
                    if ds.attrs:
                        attr_df = pd.DataFrame(list(ds.attrs.items()), columns=['Attribute', 'Value'])
                        st.dataframe(attr_df, use_container_width=True)
                    else:
                        st.info("No global attributes found")

                # Variable visualization
                st.markdown("---")
                st.subheader("🗺️ Variable Visualization")

                # Select variable to plot
                var_names = list(ds.data_vars.keys())
                if var_names:
                    selected_var = st.selectbox("Select variable to visualize", var_names,
                                               help="Select which variable to display from the NetCDF files")

                    # Check if all datasets have this variable
                    datasets_with_var = [d for d in datasets if selected_var in d['ds'].data_vars]

                    if len(datasets_with_var) < len(datasets):
                        st.warning(f"⚠️ Variable '{selected_var}' found in {len(datasets_with_var)}/{len(datasets)} files")

                    var_data = ds[selected_var]

                    # Show variable metadata
                    st.markdown(f"**Variable:** `{selected_var}`")
                    st.markdown(f"**Dimensions:** {var_data.dims}")
                    st.markdown(f"**Shape:** {var_data.shape}")

                    # Display variable attributes if available
                    if var_data.attrs:
                        with st.expander("📝 Variable Attributes"):
                            attr_text = "\n".join([f"**{k}**: {v}" for k, v in var_data.attrs.items()])
                            st.markdown(attr_text)

                    # Create visualization based on dimensions
                    if len(var_data.dims) >= 2:
                        # === COORDINATE DETECTION (like fnc.py) ===
                        # Detect coordinate names
                        coord_names = list(ds.indexes._coord_name_id) if hasattr(ds.indexes, '_coord_name_id') else list(ds.dims.keys())

                        # Find longitude, latitude, time coordinates
                        lng = next((i for i in coord_names if i.startswith("lon") or i.endswith("x") or i == "x"), None)
                        lat = next((i for i in coord_names if i.startswith("lat") or i.endswith("y") or i == "y"), None)
                        tim = next((i for i in coord_names if i.startswith("t") or i == "time" or i == "t"), None)

                        st.markdown(f"**Detected coordinates:** x={lng}, y={lat}, time={tim}")

                        # === TRANSPOSE FOR WASIM GRIDS (like fnc.py line 34-43) ===
                        needs_transpose = False
                        if coord_names == ["x", "y", "t"] or (lng == "x" and lat == "y" and tim == "t"):
                            st.info("🔄 Detected WASIM grid format - will transpose for correct orientation")
                            needs_transpose = True

                        # === MASKING CONTROLS ===
                        st.markdown("---")
                        st.markdown("#### 🎛️ Data Filtering Controls")

                        col_mask1, col_mask2 = st.columns(2)
                        with col_mask1:
                            # Mask value slider (default -9999 like in time series)
                            enable_mask = st.checkbox("Enable value masking", value=True,
                                                     help="Filter out NoData values (e.g., -9999)")
                        with col_mask2:
                            if enable_mask:
                                # Get data range for slider
                                data_min = float(var_data.min().values)
                                data_max = float(var_data.max().values)

                                # Default mask value
                                default_mask = -9999.0 if data_min <= -9999.0 else data_min

                                mask_value = st.number_input(
                                    "Mask threshold (show values > threshold)",
                                    value=default_mask,
                                    format="%.2f",
                                    help="Values less than or equal to this will be masked (e.g., -9999 for NoData)"
                                )
                            else:
                                mask_value = None

                        # === TIME LAYER SELECTION ===
                        time_layer_idx = 0
                        show_multi_layers = False
                        layer_indices = [0]

                        if tim and tim in var_data.dims:
                            time_size = var_data.sizes[tim]
                            if time_size > 1:
                                st.markdown("---")
                                st.markdown("#### ⏱️ Time Layer Selection")

                                # Option to show single or multiple layers
                                display_mode = st.radio(
                                    "Display mode",
                                    ["Single layer", "Multiple layers (subplots)"],
                                    horizontal=True,
                                    help="Choose to display one layer or multiple layers side-by-side"
                                )

                                if display_mode == "Single layer":
                                    time_layer_idx = st.slider(
                                        f"Select time layer (dimension: {tim})",
                                        0, time_size - 1, 0,
                                        help=f"Select which time step to visualize (0-{time_size-1})"
                                    )
                                    st.markdown(f"**Selected layer:** {time_layer_idx} of {time_size}")
                                    layer_indices = [time_layer_idx]
                                else:
                                    # Multi-layer subplot mode
                                    show_multi_layers = True

                                    # Limit to 8 layers
                                    max_layers = min(8, time_size)
                                    if time_size > 8:
                                        st.warning(f"⚠️ File has {time_size} time layers. Limiting to first 8 for visualization.")

                                    # Let user select how many layers to show
                                    n_layers = st.slider(
                                        "Number of layers to display",
                                        1, max_layers, min(4, max_layers),
                                        help=f"Select how many time layers to show (max 8)"
                                    )

                                    # Let user select which layers
                                    st.markdown(f"**Select {n_layers} layers to display:**")
                                    col_select1, col_select2 = st.columns(2)

                                    with col_select1:
                                        start_layer = st.number_input(
                                            "Start layer",
                                            min_value=0,
                                            max_value=max(0, time_size - n_layers),
                                            value=0,
                                            help="Starting layer index"
                                        )

                                    with col_select2:
                                        layer_step = st.number_input(
                                            "Layer step",
                                            min_value=1,
                                            max_value=max(1, time_size // n_layers),
                                            value=1,
                                            help="Step between layers (1=consecutive, 2=every other, etc.)"
                                        )

                                    # Calculate layer indices
                                    layer_indices = [start_layer + i * layer_step for i in range(n_layers)]
                                    layer_indices = [idx for idx in layer_indices if idx < time_size]

                                    st.info(f"📊 Will display layers: {layer_indices}")

                        # === PLOT ===
                        st.markdown("---")

                        # Determine what to plot: multi-layer subplots OR multi-file subplots OR single plot
                        n_files = len(datasets_with_var)

                        if n_files == 0:
                            st.error(f"No files contain variable '{selected_var}'")
                        elif show_multi_layers:
                            # === MULTI-LAYER SUBPLOT MODE ===
                            n_layers = len(layer_indices)
                            n_cols_grid = min(2, n_layers)
                            n_rows_grid = int(np.ceil(n_layers / n_cols_grid))

                            fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                                                    figsize=(12 * n_cols_grid, 8 * n_rows_grid),
                                                    squeeze=False)
                            fig.suptitle(f'{selected_var} - Multi-Layer Comparison (File: {datasets_with_var[0]["name"]})',
                                       fontsize=18, y=0.995)

                            # Use first file for multi-layer display
                            file_ds = datasets_with_var[0]['ds']
                            file_var_data = file_ds[selected_var]

                            # Detect coordinates
                            file_coord_names = list(file_ds.indexes._coord_name_id) if hasattr(file_ds.indexes, '_coord_name_id') else list(file_ds.dims.keys())
                            file_lng = next((i for i in file_coord_names if i.startswith("lon") or i.endswith("x") or i == "x"), None)
                            file_lat = next((i for i in file_coord_names if i.startswith("lat") or i.endswith("y") or i == "y"), None)
                            file_tim = next((i for i in file_coord_names if i.startswith("t") or i == "time" or i == "t"), None)
                            file_needs_transpose = file_coord_names == ["x", "y", "t"] or (file_lng == "x" and file_lat == "y" and file_tim == "t")

                            for idx, layer_idx in enumerate(layer_indices):
                                row = idx // n_cols_grid
                                col = idx % n_cols_grid
                                ax = axes[row, col]

                                # Process data for this layer
                                layer_plot_data = file_var_data.copy()

                                # Transpose if WASIM grid
                                if file_needs_transpose:
                                    if file_tim and file_tim in layer_plot_data.dims:
                                        layer_plot_data = layer_plot_data.isel({file_tim: layer_idx})
                                        layer_plot_data = layer_plot_data.transpose(file_lat, file_lng) if file_lat and file_lng else layer_plot_data.transpose()
                                    else:
                                        layer_plot_data = layer_plot_data.transpose()
                                else:
                                    if file_tim and file_tim in layer_plot_data.dims:
                                        layer_plot_data = layer_plot_data.isel({file_tim: layer_idx})

                                # Apply masking
                                if enable_mask and mask_value is not None:
                                    layer_plot_data = layer_plot_data.where(layer_plot_data.values > mask_value)

                                # Plot
                                im = layer_plot_data.plot(ax=ax, cmap='cividis', add_colorbar=True)

                                # Title with layer info
                                ax.set_title(f'Layer {layer_idx}', fontsize=12)
                                ax.set_xlabel('')
                                ax.set_ylabel('')
                                ax.grid(True, alpha=0.3)

                            # Hide empty subplots
                            for idx in range(n_layers, n_rows_grid * n_cols_grid):
                                row = idx // n_cols_grid
                                col = idx % n_cols_grid
                                axes[row, col].axis('off')

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                            # PDF export
                            pdf_bytes_nc = fig_to_pdf_bytes(fig)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes_nc,
                                file_name=f"netcdf_{selected_var}_multilayer.pdf",
                                mime="application/pdf",
                                key="pdf_nc"
                            )

                        elif n_files > 1:
                            # === MULTI-FILE SUBPLOT MODE ===
                            n_cols_grid = min(2, n_files)
                            n_rows_grid = int(np.ceil(n_files / n_cols_grid))

                            fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                                                    figsize=(12 * n_cols_grid, 8 * n_rows_grid),
                                                    squeeze=False)
                            fig.suptitle(f'{selected_var} - Multi-File Comparison', fontsize=18, y=0.995)

                            for idx, ds_info in enumerate(datasets_with_var):
                                row = idx // n_cols_grid
                                col = idx % n_cols_grid
                                ax = axes[row, col]

                                # Process data for this file
                                file_ds = ds_info['ds']
                                file_var_data = file_ds[selected_var]

                                # Detect coordinates for this file
                                file_coord_names = list(file_ds.indexes._coord_name_id) if hasattr(file_ds.indexes, '_coord_name_id') else list(file_ds.dims.keys())
                                file_lng = next((i for i in file_coord_names if i.startswith("lon") or i.endswith("x") or i == "x"), None)
                                file_lat = next((i for i in file_coord_names if i.startswith("lat") or i.endswith("y") or i == "y"), None)
                                file_tim = next((i for i in file_coord_names if i.startswith("t") or i == "time" or i == "t"), None)
                                file_needs_transpose = file_coord_names == ["x", "y", "t"] or (file_lng == "x" and file_lat == "y" and file_tim == "t")

                                # Process data
                                file_plot_data = file_var_data.copy()

                                # Transpose if WASIM grid
                                if file_needs_transpose:
                                    if file_tim and file_tim in file_plot_data.dims:
                                        file_plot_data = file_plot_data.isel({file_tim: layer_indices[0]})
                                        file_plot_data = file_plot_data.transpose(file_lat, file_lng) if file_lat and file_lng else file_plot_data.transpose()
                                    else:
                                        file_plot_data = file_plot_data.transpose()
                                else:
                                    if file_tim and file_tim in file_plot_data.dims and file_plot_data.sizes[file_tim] > 1:
                                        file_plot_data = file_plot_data.isel({file_tim: layer_indices[0]})

                                # Apply masking
                                if enable_mask and mask_value is not None:
                                    file_plot_data = file_plot_data.where(file_plot_data.values > mask_value)

                                # Plot
                                im = file_plot_data.plot(ax=ax, cmap='cividis', add_colorbar=True)

                                # Title with file name and layer info
                                title = f'{ds_info["name"]}'
                                if file_tim and file_tim in file_var_data.dims and file_var_data.sizes[file_tim] > 1:
                                    title += f'\nLayer: {layer_indices[0]}'

                                ax.set_title(title, fontsize=12)
                                ax.set_xlabel('')
                                ax.set_ylabel('')
                                ax.grid(True, alpha=0.3)

                            # Hide empty subplots
                            for idx in range(n_files, n_rows_grid * n_cols_grid):
                                row = idx // n_cols_grid
                                col = idx % n_cols_grid
                                axes[row, col].axis('off')

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                            # PDF export
                            pdf_bytes_nc = fig_to_pdf_bytes(fig)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes_nc,
                                file_name=f"netcdf_{selected_var}_comparison_layer{layer_indices[0]}.pdf",
                                mime="application/pdf",
                                key="pdf_nc"
                            )

                        else:
                            # === SINGLE PLOT MODE ===
                            file_ds = datasets_with_var[0]['ds']
                            file_var_data = file_ds[selected_var]

                            # Detect coordinates
                            file_coord_names = list(file_ds.indexes._coord_name_id) if hasattr(file_ds.indexes, '_coord_name_id') else list(file_ds.dims.keys())
                            file_lng = next((i for i in file_coord_names if i.startswith("lon") or i.endswith("x") or i == "x"), None)
                            file_lat = next((i for i in file_coord_names if i.startswith("lat") or i.endswith("y") or i == "y"), None)
                            file_tim = next((i for i in file_coord_names if i.startswith("t") or i == "time" or i == "t"), None)
                            file_needs_transpose = file_coord_names == ["x", "y", "t"] or (file_lng == "x" and file_lat == "y" and file_tim == "t")

                            # Process data
                            plot_data = file_var_data.copy()

                            # Transpose if WASIM grid
                            if file_needs_transpose:
                                if file_tim and file_tim in plot_data.dims:
                                    plot_data = plot_data.isel({file_tim: layer_indices[0]})
                                    plot_data = plot_data.transpose(file_lat, file_lng) if file_lat and file_lng else plot_data.transpose()
                                else:
                                    plot_data = plot_data.transpose()
                            else:
                                if file_tim and file_tim in plot_data.dims and plot_data.sizes[file_tim] > 1:
                                    plot_data = plot_data.isel({file_tim: layer_indices[0]})

                            # Apply masking
                            if enable_mask and mask_value is not None:
                                plot_data = plot_data.where(plot_data.values > mask_value)

                                # Check if all data is masked
                                if np.all(np.isnan(plot_data.values)):
                                    st.error(f"⚠️ All data is masked with threshold {mask_value}. Try adjusting the mask value.")
                                else:
                                    valid_pct = 100 * (1 - np.isnan(plot_data.values).sum() / plot_data.values.size)
                                    st.success(f"✅ {valid_pct:.1f}% of data remains after masking")

                            # Plot
                            fig, ax = plt.subplots(figsize=(12, 8))
                            im = plot_data.plot(ax=ax, cmap='cividis', add_colorbar=True)

                            # Title with layer info
                            title = f'{selected_var}'
                            if file_tim and file_tim in file_var_data.dims and file_var_data.sizes[file_tim] > 1:
                                title += f' | Layer: {layer_indices[0]}'
                            if enable_mask and mask_value is not None:
                                title += f' | Masked > {mask_value}'

                            ax.set_title(title, fontsize=14)
                            ax.set_xlabel('')
                            ax.set_ylabel('')
                            ax.grid(True, alpha=0.3)

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                            # PDF export
                            pdf_bytes_nc = fig_to_pdf_bytes(fig)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes_nc,
                                file_name=f"netcdf_{selected_var}_layer{layer_indices[0]}.pdf",
                                mime="application/pdf",
                                key="pdf_nc"
                            )

                        # === STATISTICS ===
                        st.markdown("---")
                        st.subheader("📊 Statistics (First File)")

                        # Use first file for statistics display
                        if len(datasets_with_var) > 0:
                            first_file_ds = datasets_with_var[0]['ds']
                            first_file_var = first_file_ds[selected_var]

                            # Process data like we did for plotting
                            first_coord_names = list(first_file_ds.indexes._coord_name_id) if hasattr(first_file_ds.indexes, '_coord_name_id') else list(first_file_ds.dims.keys())
                            first_lng = next((i for i in first_coord_names if i.startswith("lon") or i.endswith("x") or i == "x"), None)
                            first_lat = next((i for i in first_coord_names if i.startswith("lat") or i.endswith("y") or i == "y"), None)
                            first_tim = next((i for i in first_coord_names if i.startswith("t") or i == "time" or i == "t"), None)
                            first_needs_transpose = first_coord_names == ["x", "y", "t"] or (first_lng == "x" and first_lat == "y" and first_tim == "t")

                            stats_plot_data = first_file_var.copy()
                            if first_needs_transpose:
                                if first_tim and first_tim in stats_plot_data.dims:
                                    stats_plot_data = stats_plot_data.isel({first_tim: time_layer_idx})
                                    stats_plot_data = stats_plot_data.transpose(first_lat, first_lng) if first_lat and first_lng else stats_plot_data.transpose()
                                else:
                                    stats_plot_data = stats_plot_data.transpose()
                            else:
                                if first_tim and first_tim in stats_plot_data.dims and stats_plot_data.sizes[first_tim] > 1:
                                    stats_plot_data = stats_plot_data.isel({first_tim: time_layer_idx})

                            if enable_mask and mask_value is not None:
                                stats_plot_data = stats_plot_data.where(stats_plot_data.values > mask_value)

                            # Calculate quantiles like fnc.py
                            quantiles = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
                            quant_values = np.nanquantile(stats_plot_data.values, quantiles)
                            quant_df = pd.DataFrame({
                                'Quantile': [f'{q:.0%}' for q in quantiles],
                                'Value': [f'{v:.4f}' for v in quant_values]
                            })

                            col_stats1, col_stats2 = st.columns([1, 1])
                            with col_stats1:
                                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                                with stats_col1:
                                    st.metric("Mean", f"{float(np.nanmean(stats_plot_data.values)):.4f}")
                                with stats_col2:
                                    st.metric("Std Dev", f"{float(np.nanstd(stats_plot_data.values)):.4f}")
                                with stats_col3:
                                    st.metric("Min", f"{float(np.nanmin(stats_plot_data.values)):.4f}")
                                with stats_col4:
                                    st.metric("Max", f"{float(np.nanmax(stats_plot_data.values)):.4f}")

                            with col_stats2:
                                st.markdown("**Quantiles:**")
                                st.dataframe(quant_df, hide_index=True, use_container_width=True)

                    else:
                        st.warning("Variable needs at least 2 dimensions for visualization")

                    # Export to CSV
                    st.subheader("💾 Export Data")
                    if st.button("Convert to CSV"):
                        # Convert to pandas dataframe
                        df_nc = var_data.to_dataframe().reset_index()
                        csv_data = df_nc.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"{selected_var}.csv",
                            mime="text/csv"
                        )

                else:
                    st.warning("No variables found in NetCDF file")

        except Exception as e:
            st.error(f"❌ Error processing NetCDF file(s): {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        finally:
            # Clean up all temporary files
            for nc_path in nc_paths:
                if os.path.exists(nc_path):
                    os.remove(nc_path)
            if os.path.exists(temp_dir_nc):
                try:
                    os.rmdir(temp_dir_nc)
                except OSError:
                    pass  # Directory not empty, that's ok
