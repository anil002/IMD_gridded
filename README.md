# Advanced Climate Rainfall Dashboard

A powerful, interactive Streamlit dashboard for advanced rainfall and climate data analysis.  
Upload NetCDF rainfall datasets and perform international-standard statistical, geospatial, and parametric analyses with professional visualizations and export options.

## Features

- **Flexible Data Upload:** NetCDF file upload or path input
- **Geospatial Mapping:** Interactive maps with Folium, AOI selection (GeoJSON/Shapefile), and point selection
- **Time Series Analysis:** Daily, monthly, annual, and seasonal rainfall trends
- **Extreme Events & Drought:** SPI, threshold exceedance, and anomaly detection
- **Parametric Analysis:** Histogram, PDF fits (Normal, Gamma), Q-Q plots, return period, IDF curves, and more
- **Professional Exports:** Download results as CSV, Excel, or PDF reports (with plots and tables)
- **International Standards:** Methods and references from WMO, Wilks, Chow, and more

## Requirements

- Python 3.8+
- See [`requirements.txt`](requirements.txt) for all dependencies

## Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    > For PDF export, you may also need to install [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html) on your system.

3. **Run the app:**
    ```bash
    streamlit run app5.py
    ```

4. **Open in your browser:**  
   The app will open automatically, or visit [http://localhost:8501](http://localhost:8501)

## Usage

- Upload your NetCDF rainfall data or enter a file path.
- Select your region, point(s), or area of interest.
- Choose the analysis mode (mapping, time series, parametric, etc.).
- Download results as CSV, Excel, or PDF.

## Citation & References

- Wilks, D.S. (2011). *Statistical Methods in the Atmospheric Sciences*
- Chow, V.T. (1964). *Handbook of Applied Hydrology*
- WMO-No. 168, *Guide to Hydrological Practices*
- WMO-No. 1173, *Manual on Flood Forecasting and Warning*

## License

MIT License

---

**Developed for research, education, and operational hydrology/climate analysis.**
