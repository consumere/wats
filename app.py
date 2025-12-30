import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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


def plot_seasonal_decomposition(df, column, unit=None):
    """Plot yearly patterns using box plots."""
    data = df[column].dropna()

    # Create year and month columns
    plot_df = pd.DataFrame({
        'value': data.values,
        'month': data.index.month,
        'year': data.index.year
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Monthly boxplot
    monthly_data = [plot_df[plot_df['month'] == m]['value'].values for m in range(1, 13)]
    bp1 = axes[0].boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                           patch_artist=True, showmeans=True)

    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    axes[0].set_xlabel('')  # No x-label (months are clear)
    if unit:
        axes[0].set_ylabel(f'{column} [{unit}]', fontsize=11)
    else:
        axes[0].set_ylabel(f'{column}', fontsize=11)
    axes[0].set_title(f'Monthly Distribution', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Yearly trend
    yearly_mean = plot_df.groupby('year')['value'].mean()
    axes[1].plot(yearly_mean.index, yearly_mean.values, marker='o', linewidth=2, markersize=6)
    axes[1].fill_between(yearly_mean.index, yearly_mean.values, alpha=0.3)
    axes[1].set_xlabel('')  # No x-label (years are clear)
    if unit:
        axes[1].set_ylabel(f'Mean [{unit}]', fontsize=11)
    else:
        axes[1].set_ylabel(f'Mean Value', fontsize=11)
    axes[1].set_title(f'Yearly Trend', fontsize=12)
    axes[1].grid(True, alpha=0.3)

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


# Streamlit app
st.title("📊 Time Series Data Viewer")
st.markdown("""
This application reads and visualizes whitespace-delimited time series data files.
Upload one or multiple files with columns: YY, MM, DD, HH followed by data columns.
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
    - 📥 CSV export
    """)

    st.header("ℹ️ About")
    st.info("Time Series Viewer v2.0\n\nOptimized for hydrological and meteorological data")

# File upload - MULTIPLE FILES (NO TYPE RESTRICTION)
uploaded_files = st.file_uploader(
    "Upload your data file(s)",
    accept_multiple_files=True,
    help="Upload one or more delimited files with temporal data (any file type)"
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
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📉 Time Series",
                    "📊 Aggregated Stats",
                    "🔄 Rolling Statistics",
                    "📅 Seasonal Patterns",
                    "🔗 Correlations"
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
                    if len(selected_columns) > 0:
                        season_column = st.selectbox("Select column for seasonal analysis", selected_columns)
                        fig4 = plot_seasonal_decomposition(df, season_column, unit)
                        st.pyplot(fig4)
                        plt.close(fig4)

                        # PDF export button
                        pdf_bytes4 = fig_to_pdf_bytes(fig4)
                        st.download_button(
                            label="📥 Download as PDF",
                            data=pdf_bytes4,
                            file_name=f"seasonal_{season_column}.pdf",
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
