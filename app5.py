import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, gamma
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import folium
from folium import plugins
from streamlit_folium import st_folium
import json
import tempfile
import os
from geopy.geocoders import Nominatim
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced Climate Rainfall Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #17becf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .upload-zone {
        border: 2px dashed #1f77b4;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        background-color: #f0f2f6;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_netcdf_data(uploaded_file):
    """Load and preprocess the uploaded NetCDF rainfall data"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        ds = xr.open_dataset(tmp_file_path)
        os.unlink(tmp_file_path)
        
        coord_mapping = {}
        coords = list(ds.coords.keys())
        dims = list(ds.dims.keys())
        all_vars = coords + dims
        
        time_vars = ['time', 'TIME', 'Time', 't', 'T']
        for var in time_vars:
            if var in all_vars:
                coord_mapping[var] = 'TIME'
                break
        
        lat_vars = ['lat', 'latitude', 'LATITUDE', 'Latitude', 'y', 'Y']
        for var in lat_vars:
            if var in all_vars:
                coord_mapping[var] = 'LATITUDE'
                break
        
        lon_vars = ['lon', 'longitude', 'LONGITUDE', 'Longitude', 'x', 'X']
        for var in lon_vars:
            if var in all_vars:
                coord_mapping[var] = 'LONGITUDE'
                break
        
        if coord_mapping:
            ds = ds.rename(coord_mapping)
        
        data_vars = list(ds.data_vars.keys())
        rainfall_var = None
        rainfall_names = ['rainfall', 'RAINFALL', 'Rainfall', 'precip', 'precipitation', 
                         'PRECIP', 'PRECIPITATION', 'rain', 'RAIN', 'pr', 'PR']
        
        for var in rainfall_names:
            if var in data_vars:
                rainfall_var = var
                break
        
        if rainfall_var is None and data_vars:
            rainfall_var = data_vars[0]
            st.warning(f"Using '{rainfall_var}' as rainfall variable. Please verify this is correct.")
        
        if rainfall_var and rainfall_var != 'RAINFALL':
            ds = ds.rename({rainfall_var: 'RAINFALL'})
        
        if 'TIME' in ds.coords:
            ds['TIME'] = pd.to_datetime(ds['TIME'])
        
        if 'TIME' in ds.dims:
            ds = ds.sortby('TIME')
        
        return ds
    except Exception as e:
        st.error(f"Error loading NetCDF file: {str(e)}")
        st.error("Please ensure your file contains rainfall/precipitation data with time, latitude, and longitude coordinates.")
        return None

@st.cache_data
def load_geojson_file(uploaded_file):
    """Load and parse GeoJSON file"""
    try:
        if uploaded_file is not None:
            geojson_data = json.load(uploaded_file)
            gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
            return gdf, geojson_data
        return None, None
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return None, None

@st.cache_data
def load_shapefile(uploaded_files):
    """Load and parse Shapefile"""
    try:
        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = {}
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths[uploaded_file.name] = file_path
                shp_file = None
                for name, path in file_paths.items():
                    if name.endswith('.shp'):
                        shp_file = path
                        break
                if shp_file:
                    gdf = gpd.read_file(shp_file)
                    return gdf
                else:
                    st.error("No .shp file found in uploaded files")
                    return None
    except Exception as e:
        st.error(f"Error loading Shapefile: {e}")
        return None

def geocode_city(city_name):
    """Geocode city name to coordinates"""
    try:
        geolocator = Nominatim(user_agent="rainfall_dashboard")
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        st.error(f"Error geocoding city: {e}")
        return None, None

def create_point_selection_map(ds):
    """Create a Folium map for selecting single or multiple points"""
    center_lat = float(ds.LATITUDE.mean())
    center_lon = float(ds.LONGITUDE.mean())
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    lats = ds.LATITUDE.values
    lons = ds.LONGITUDE.values
    step = max(1, len(lats) // 100)
    for i in range(0, len(lats), step):
        for j in range(0, len(lons), step):
            folium.CircleMarker(
                location=[float(lats[i]), float(lons[j])],
                radius=5,
                color='blue',
                fill=True,
                fill_opacity=0.6,
                popup=f"Lat: {lats[i]:.2f}, Lon: {lons[j]:.2f}",
                tooltip=f"Lat: {lats[i]:.2f}, Lon: {lons[j]:.2f}"
            ).add_to(m)
    return m

def create_folium_map(ds, analysis_type="mean_rainfall", aoi_geometry=None, 
                     threshold=None, time_period=None, spi_data=None, 
                     selected_month=None, selected_week=None, selected_points=None):
    """Create advanced Folium map with various analysis options"""
    
    if selected_points:
        lats = [point[0] for point in selected_points]
        lons = [point[1] for point in selected_points]
        ds_subset = ds.sel(LATITUDE=lats, LONGITUDE=lons, method='nearest')
        bounds = [[min(lats) - 0.5, min(lons) - 0.5], [max(lats) + 0.5, max(lons) + 0.5]]
    else:
        ds_subset = ds
        bounds = [[float(ds.LATITUDE.min()) - 0.5, float(ds.LONGITUDE.min()) - 0.5],
                  [float(ds.LATITUDE.max()) + 0.5, float(ds.LONGITUDE.max()) + 0.5]]

    center_lat = float(ds_subset.LATITUDE.mean())
    center_lon = float(ds_subset.LONGITUDE.mean())
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8 if selected_points else 6,
        tiles=None
    )
    
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
        name='Terrain',
        attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors'
    ).add_to(m)
    
    time_subset = ds_subset.sel(TIME=slice(time_period[0], time_period[1])) if time_period else ds_subset
    
    if not time_subset['RAINFALL'].size:
        st.error("No valid rainfall data available for the selected time period and location.")
        return m
    
    if analysis_type == "mean_rainfall":
        data_array = time_subset['RAINFALL'].mean(dim='TIME')
        title = "Mean Daily Rainfall (mm/day)"
    elif analysis_type == "rainfall_variability":
        data_array = time_subset['RAINFALL'].std(dim='TIME')
        title = "Rainfall Variability (Standard Deviation)"
    elif analysis_type == "extreme_events" and threshold:
        extreme_events = (time_subset['RAINFALL'] > threshold).sum(dim='TIME')
        data_array = extreme_events
        title = f"Extreme Events Count (>{threshold} mm/day)"
    elif analysis_type == "drought_spi" and spi_data is not None:
        data_array = xr.DataArray(
            spi_data,
            coords={'LATITUDE': ds_subset.LATITUDE, 'LONGITUDE': ds_subset.LONGITUDE},
            dims=['LATITUDE', 'LONGITUDE']
        )
        title = "Standardized Precipitation Index (SPI)"
    elif analysis_type == "weekly_cumulative_average":
        if selected_points and len(selected_points) == 1:
            # Handle single point case
            rainfall = time_subset['RAINFALL']
            if 'TIME' not in rainfall.dims or rainfall.sizes.get('TIME', 0) == 0:
                st.error("No valid time data available for the selected point.")
                return m
            time_df = pd.DataFrame({'TIME': rainfall['TIME'].values})
            time_df['Week'] = pd.to_datetime(time_df['TIME']).isocalendar().week
            time_df['Year'] = pd.to_datetime(time_df['TIME']).dt.year
            rainfall_df = rainfall.to_dataframe().reset_index()
            rainfall_df['Week'] = time_df['Week']
            rainfall_df['Year'] = time_df['Year']
            week_mask = (rainfall_df['Week'] >= 1) & (rainfall_df['Week'] <= 52)
            if not week_mask.any():
                st.error("No data available for weeks 1-52 in the selected date range.")
                return m
            rainfall_df = rainfall_df[week_mask]
            week_sum = rainfall_df.groupby(['Year', 'Week'])['RAINFALL'].sum().reset_index()
            all_weeks = pd.DataFrame({'Week': range(1, 53)})
            week_sum = week_sum.groupby('Week')['RAINFALL'].mean().reindex(all_weeks['Week']).reset_index()
            if selected_week is not None:
                if selected_week < 1 or selected_week > 52 or week_sum[week_sum['Week'] == selected_week].empty:
                    st.error(f"No data available for ISO week {selected_week}.")
                    return m
                data_array = xr.DataArray(
                    week_sum[week_sum['Week'] == selected_week]['RAINFALL'].values[0],
                    coords={'LATITUDE': ds_subset.LATITUDE, 'LONGITUDE': ds_subset.LONGITUDE},
                    dims=['LATITUDE', 'LONGITUDE']
                )
                title = f"Avg Rainfall for Week {selected_week} (mm/week)"
            else:
                data_array = xr.DataArray(
                    week_sum['RAINFALL'].mean(),
                    coords={'LATITUDE': ds_subset.LATITUDE, 'LONGITUDE': ds_subset.LONGITUDE},
                    dims=['LATITUDE', 'LONGITUDE']
                )
                title = "Weekly Cumulative Average Rainfall (mm/week)"
        else:
            # Handle grid or multiple points
            week_numbers = pd.to_datetime(time_subset['TIME'].values).isocalendar().week
            years = pd.to_datetime(time_subset['TIME'].values).year
            time_subset = time_subset.assign_coords(WEEK=("TIME", week_numbers), YEAR=("TIME", years))
            
            if 'RAINFALL' not in time_subset:
                st.error("Rainfall variable not found in dataset.")
                return m
            
            rainfall = time_subset['RAINFALL']
            if 'TIME' not in rainfall.dims or rainfall.sizes.get('TIME', 0) == 0:
                st.error("No valid time data available for the selected location.")
                return m
            
            week_mask = (time_subset['WEEK'] >= 1) & (time_subset['WEEK'] <= 52)
            if not week_mask.any().item():
                st.error("No data available for weeks 1-52 in the selected date range.")
                return m
            rainfall = rainfall.where(week_mask, drop=True)
            
            if not rainfall.size or 'TIME' not in rainfall.dims:
                st.error("No valid data after filtering weeks 1-52.")
                return m
            
            week_sum = rainfall.groupby(["YEAR", "WEEK"]).sum(dim="TIME")
            all_weeks = np.arange(1, 53)
            week_sum = week_sum.reindex(WEEK=all_weeks)
            week_cum_avg = week_sum.mean(dim="YEAR")  # Average of weekly sums over years
            title = "Weekly Cumulative Rainfall (mm/week)"
            
            if selected_week is not None:
                if selected_week < 1 or selected_week > 52 or np.all(np.isnan(week_cum_avg.sel(WEEK=selected_week))):
                    st.error(f"No data available for ISO week {selected_week} in the selected date range.")
                    return m
                data_array = week_cum_avg.sel(WEEK=selected_week)
                title = f"Cumulative Rainfall for Week {selected_week} (mm/week)"
            else:
                data_array = week_cum_avg
    elif analysis_type == "monthly_cumulative_average":
        months = pd.to_datetime(time_subset['TIME'].values).month
        years = pd.to_datetime(time_subset['TIME'].values).year
        time_subset = time_subset.assign_coords(MONTH=("TIME", months), YEAR=("TIME", years))
        
        if 'RAINFALL' not in time_subset:
            st.error("Rainfall variable not found in dataset.")
            return m
        
        rainfall = time_subset['RAINFALL']
        if 'TIME' not in rainfall.dims or rainfall.sizes.get('TIME', 0) == 0:
            st.error("No valid time data available for the selected location.")
            return m
        
        rainfall = rainfall.where((time_subset['MONTH'] >= 1) & (time_subset['MONTH'] <= 12), drop=True)
        
        if not rainfall.size or 'TIME' not in rainfall.dims:
            st.error("No valid data after filtering months 1-12.")
            return m
        
        month_sum = rainfall.groupby(["YEAR", "MONTH"]).sum(dim="TIME")
        all_months = np.arange(1, 13)
        month_sum = month_sum.reindex(MONTH=all_months)
        month_cum_avg = month_sum.mean(dim="YEAR")  # Average of monthly sums over years
        title = "Monthly Cumulative Rainfall (mm/month)"
        
        if selected_month is not None:
            if selected_month < 1 or selected_month > 12 or np.all(np.isnan(month_cum_avg.sel(MONTH=selected_month))):
                st.error(f"No data available for month {selected_month} in the selected date range.")
                return m
            data_array = month_cum_avg.sel(MONTH=selected_month)
            month_name = pd.Timestamp(month=selected_month, day=1, year=2000).strftime('%B')
            title = f"Cumulative Rainfall for {month_name} (mm/month)"
        else:
            data_array = month_cum_avg
    elif analysis_type == "custom_range_cumulative":
        data_array = time_subset['RAINFALL'].sum(dim='TIME')
        title = f"Total Rainfall (mm) ({time_period[0]} to {time_period[1]})"
    else:
        data_array = time_subset['RAINFALL'].mean(dim='TIME')
        title = "Mean Daily Rainfall (mm/day)"
    
    lats = ds_subset.LATITUDE.values
    lons = ds_subset.LONGITUDE.values
    values = data_array.values

    if np.isnan(values).all():
        min_val, max_val = 0, 0
    else:
        min_val = float(np.nanmin(values))
        max_val = float(np.nanmax(values))
    q20 = min_val + 0.2 * (max_val - min_val)
    q40 = min_val + 0.4 * (max_val - min_val)
    q60 = min_val + 0.6 * (max_val - min_val)
    q80 = min_val + 0.8 * (max_val - min_val)

    heat_data = []
    if values.ndim == 0:
        if not np.isnan(values):
            for lat, lon in zip(lats, lons):
                heat_data.append([float(lat), float(lon), float(values)])
    else:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                val = values[i, j] if values.ndim == 2 else values[0] if values.ndim == 1 else values
                if not np.isnan(val):
                    heat_data.append([float(lat), float(lon), float(val)])

    if heat_data:
        HeatMap = plugins.HeatMap(
            heat_data,
            name=title,
            min_opacity=0.2,
            max_zoom=18,
            radius=15 if not selected_points else 10,
            blur=10,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'} if analysis_type != "drought_spi" 
                     else {0: 'red', 0.5: 'yellow', 1: 'blue'}
        )
        HeatMap.add_to(m)

        max_markers = 2000 if not selected_points else len(selected_points) * 2
        step_lat = max(1, len(lats) // int(np.sqrt(max_markers)))
        step_lon = max(1, len(lons) // int(np.sqrt(max_markers)))
        for i in range(0, len(lats), step_lat):
            for j in range(0, len(lons), step_lon):
                val = values[i, j] if values.ndim == 2 else values[0] if values.ndim == 1 else values
                if not np.isnan(val):
                    folium.CircleMarker(
                        location=[float(lats[i]), float(lons[j])],
                        radius=5,
                        color='black',
                        fill=True,
                        fill_opacity=0.01,
                        opacity=0.01,
                        tooltip=f"{title}: {val:.2f}"
                    ).add_to(m)

    if aoi_geometry is not None and not getattr(aoi_geometry, "empty", False):
        if isinstance(aoi_geometry, dict):
            folium.GeoJson(
                aoi_geometry,
                name="Area of Interest",
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': '#000000',
                    'weight': 4,
                    'fillOpacity': 0.1
                }
            ).add_to(m)
        elif hasattr(aoi_geometry, 'geometry'):
            for idx, row in aoi_geometry.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Polygon':
                    coords = [[point[1], point[0]] for point in geom.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        popup=f"AOI Feature {idx}",
                        color='red',
                        weight=4,
                        fillOpacity=0.1
                    ).add_to(m)

    if selected_points:
        for lat, lon in selected_points:
            folium.Marker(
                location=[float(lat), float(lon)],
                popup=f"Selected Point: {lat:.2f}, {lon:.2f}",
                icon=folium.Icon(color='red', icon='map-marker')
            ).add_to(m)

    legend_html = ""
    if analysis_type in ["mean_rainfall", "cumulative_average", "custom_range_cumulative"]:
        legend_html = f"""
        <div style="background: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; position: absolute; top: 10px; left: 10px;">
        <b>Legend: {title}</b><br>
        <span style='color:blue;'>■</span> Low ({min_val:.1f} - {q20:.1f})<br>
        <span style='color:lime;'>■</span> Medium ({q20:.1f} - {q40:.1f})<br>
        <span style='color:yellow;'>■</span> Moderate ({q40:.1f} - {q60:.1f})<br>
        <span style='color:orange;'>■</span> High ({q60:.1f} - {q80:.1f})<br>
        <span style='color:red;'>■</span> Very High ({q80:.1f} - {max_val:.1f})
        </div>
        """
    elif analysis_type == "trend_slope":
        legend_html = f"""
        <div style="background: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; position: absolute; top: 10px; left: 10px;">
        <b>Legend: Trend (Slope) Map</b><br>
        <span style='color:blue;'>■</span> Strong Decrease<br>
        <span style='color:lime;'>■</span> Moderate Decrease<br>
        <span style='color:yellow;'>■</span> Stable<br>
        <span style='color:orange;'>■</span> Moderate Increase<br>
        <span style='color:red;'>■</span> Strong Increase
        </div>
        """
    elif analysis_type == "forecast_mean":
        legend_html = f"""
        <div style="background: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; position: absolute; top: 10px; left: 10px;">
        <b>Legend: Forecast Map</b><br>
        <span style='color:blue;'>■</span> Low Forecast<br>
        <span style='color:lime;'>■</span> Medium-Low<br>
        <span style='color:yellow;'>■</span> Medium<br>
        <span style='color:orange;'>■</span> Medium-High<br>
        <span style='color:red;'>■</span> High Forecast
        </div>
        """

    if legend_html:
        m.get_root().html.add_child(folium.Element(legend_html))

    m.fit_bounds(bounds)
    folium.LayerControl().add_to(m)
    return m

@st.cache_data
def calculate_spi(_precipitation, timescale=3):
    """Calculate Standardized Precipitation Index (SPI)"""
    try:
        if isinstance(_precipitation, xr.DataArray):
            precip_series = _precipitation.to_pandas()
        else:
            precip_series = pd.Series(_precipitation)
        rolling_sum = precip_series.rolling(window=timescale*30, min_periods=1).sum()
        positive_data = rolling_sum[rolling_sum > 0].dropna()
        if len(positive_data) < 50:
            return np.full_like(rolling_sum, np.nan)
        alpha, loc, beta = gamma.fit(positive_data, floc=0)
        cdf_values = gamma.cdf(rolling_sum, alpha, loc=loc, scale=beta)
        cdf_values = np.clip(cdf_values, 0.001, 0.999)
        spi = norm.ppf(cdf_values)
        return spi
    except Exception as e:
        st.error(f"Error calculating SPI: {e}")
        return np.full_like(_precipitation, np.nan)

def create_time_series_plots(ds, location_name="Selected Location", selected_points=None, extreme_threshold=50,
                            daily_start_year=None, daily_end_year=None):
    """Create comprehensive time series analysis plots"""
    
    if selected_points:
        lats = [point[0] for point in selected_points]
        lons = [point[1] for point in selected_points]
        ts_data = ds.sel(LATITUDE=lats, LONGITUDE=lons, method='nearest')['RAINFALL'].mean(dim=['LATITUDE', 'LONGITUDE'])
    else:
        ts_data = ds['RAINFALL'].mean(dim=['LATITUDE', 'LONGITUDE'])

    if not ts_data.size:
        st.error("No valid rainfall data available for time series analysis.")
        return go.Figure()

    ts_df = ts_data.to_dataframe().reset_index()
    ts_df['Year'] = ts_df['TIME'].dt.year
    ts_df['Month'] = ts_df['TIME'].dt.month
    ts_df['Week'] = ts_df['TIME'].dt.isocalendar().week
    ts_df['Day'] = ts_df['TIME'].dt.day

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Daily Rainfall Time Series',
            'Monthly Average Rainfall',
            'Annual Rainfall Totals',
            'Seasonal Pattern',
            'Rainfall Distribution',
            'Extreme Events',
            'Weekly Cumulative Rainfall',
            'Monthly Cumulative Average Rainfall'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    if daily_start_year is not None and daily_end_year is not None:
        mask = (ts_df['Year'] >= daily_start_year) & (ts_df['Year'] <= daily_end_year)
        recent_data = ts_df[mask]
    else:
        recent_data = ts_df[ts_df['Year'] >= ts_df['Year'].max() - 5]
    fig.add_trace(
        go.Scatter(
            x=recent_data['TIME'],
            y=recent_data['RAINFALL'],
            mode='lines',
            name='Daily Rainfall',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )

    monthly_avg = ts_df.groupby('Month')['RAINFALL'].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=monthly_avg['Month'],
            y=monthly_avg['RAINFALL'],
            name='Monthly Average',
            marker_color='lightblue'
        ),
        row=1, col=2
    )

    annual_totals = ts_df.groupby('Year')['RAINFALL'].sum().reset_index()
    fig.add_trace(
        go.Scatter(
            x=annual_totals['Year'],
            y=annual_totals['RAINFALL'],
            mode='lines+markers',
            name='Annual Total',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )

    z = np.polyfit(annual_totals['Year'], annual_totals['RAINFALL'], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=annual_totals['Year'],
            y=p(annual_totals['Year']),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash', width=2)
        ),
        row=2, col=1
    )

    seasonal_pattern = ts_df.groupby([ts_df['TIME'].dt.dayofyear])['RAINFALL'].mean().reset_index()
    seasonal_pattern['Day_of_Year'] = seasonal_pattern['TIME']
    fig.add_trace(
        go.Scatter(
            x=seasonal_pattern['Day_of_Year'],
            y=seasonal_pattern['RAINFALL'],
            mode='lines',
            name='Seasonal Pattern',
            line=dict(color='purple', width=2)
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Histogram(
            x=ts_df['RAINFALL'],
            nbinsx=50,
            name='Distribution',
            marker_color='orange'
        ),
        row=3, col=1
    )

    extreme_events = ts_df[ts_df['RAINFALL'] > extreme_threshold]
    extreme_by_year = extreme_events.groupby('Year').size().reset_index(name='Extreme_Count')
    fig.add_trace(
        go.Bar(
            x=extreme_by_year['Year'],
            y=extreme_by_year['Extreme_Count'],
            name=f'Extreme Events/Year > {extreme_threshold} mm',
            marker_color='red'
        ),
        row=3, col=2
    )

    weekly_cum_by_week = ts_df.groupby('Week')['RAINFALL'].sum().reset_index()
    weekly_mean_by_week = ts_df.groupby('Week')['RAINFALL'].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=weekly_cum_by_week['Week'],
            y=weekly_cum_by_week['RAINFALL'],
            name='Weekly Cumulative (All Years)',
            marker_color='teal'
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=weekly_mean_by_week['Week'],
            y=weekly_mean_by_week['RAINFALL'],
            mode='lines+markers',
            name='Weekly Mean (All Years)',
            line=dict(color='navy', dash='dot')
        ),
        row=4, col=1
    )

    monthly_cum = ts_df.groupby(['Year', 'Month'])['RAINFALL'].sum().reset_index()
    monthly_cum_avg = monthly_cum.groupby('Month')['RAINFALL'].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=monthly_cum_avg['Month'],
            y=monthly_cum_avg['RAINFALL'],
            name='Monthly Cumulative Avg',
            marker_color='darkgreen'
        ),
        row=4, col=2
    )

    fig.update_layout(
        height=1200,
        title_text=f"Comprehensive Rainfall Analysis - {location_name}",
        showlegend=False
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Day of Year", row=2, col=2)
    fig.update_xaxes(title_text="Rainfall (mm)", row=3, col=1)
    fig.update_xaxes(title_text="Year", row=3, col=2)
    fig.update_xaxes(title_text="Week", row=4, col=1)
    fig.update_xaxes(title_text="Month", row=4, col=2)

    fig.update_yaxes(title_text="Rainfall (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Rainfall (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Annual Total (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Average Rainfall (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    fig.update_yaxes(title_text="Extreme Events Count", row=3, col=2)
    fig.update_yaxes(title_text="Rainfall (mm)", row=4, col=1)
    fig.update_yaxes(title_text="Rainfall (mm)", row=4, col=2)

    return fig, ts_df

def main():
    st.markdown('<h1 class="main-header">🌧️ Advanced Climate Rainfall Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.header("📁 Data Upload")
    
    st.markdown("""
    <div class="upload-zone">
        <h3>Upload Your NetCDF Rainfall Data or Enter File Path</h3>
        <p>Supported formats: .nc, .netcdf</p>
        <p>Expected variables: rainfall/precipitation with time, latitude, and longitude coordinates</p>
    </div>
    """, unsafe_allow_html=True)
    
    file_path = st.text_input(
        "Or enter NetCDF file path (e.g., /content/drive/MyDrive/yourfile.nc):",
        value=""
    )
    
    uploaded_nc_file = st.file_uploader(
        "Choose a NetCDF file",
        type=['nc', 'netcdf']
    )
    
    ds = None
    if file_path:
        try:
            with st.spinner("Loading NetCDF data from path..."):
                ds = xr.open_dataset(file_path)
                coord_mapping = {}
                coords = list(ds.coords.keys())
                dims = list(ds.dims.keys())
                all_vars = coords + dims
                time_vars = ['time', 'TIME', 'Time', 't', 'T']
                for var in time_vars:
                    if var in all_vars:
                        coord_mapping[var] = 'TIME'
                        break
                lat_vars = ['lat', 'latitude', 'LATITUDE', 'Latitude', 'y', 'Y']
                for var in lat_vars:
                    if var in all_vars:
                        coord_mapping[var] = 'LATITUDE'
                        break
                lon_vars = ['lon', 'longitude', 'LONGITUDE', 'Longitude', 'x', 'X']
                for var in lon_vars:
                    if var in all_vars:
                        coord_mapping[var] = 'LONGITUDE'
                        break
                if coord_mapping:
                    ds = ds.rename(coord_mapping)
                data_vars = list(ds.data_vars.keys())
                rainfall_var = None
                rainfall_names = ['rainfall', 'RAINFALL', 'Rainfall', 'precip', 'precipitation', 
                                 'PRECIP', 'PRECIPITATION', 'rain', 'RAIN', 'pr', 'PR']
                for var in rainfall_names:
                    if var in data_vars:
                        rainfall_var = var
                        break
                if rainfall_var is None and data_vars:
                    rainfall_var = data_vars[0]
                    st.warning(f"Using '{rainfall_var}' as rainfall variable. Please verify this is correct.")
                if rainfall_var and rainfall_var != 'RAINFALL':
                    ds = ds.rename({rainfall_var: 'RAINFALL'})
                if 'TIME' in ds.coords:
                    ds['TIME'] = pd.to_datetime(ds['TIME'])
                if 'TIME' in ds.dims:
                    ds = ds.sortby('TIME')
            st.success("✅ NetCDF file loaded from path successfully!")
        except Exception as e:
            st.error(f"Error loading NetCDF file from path: {e}")
            ds = None
    elif uploaded_nc_file is not None:
        with st.spinner("Loading NetCDF data..."):
            ds = load_netcdf_data(uploaded_nc_file)
        if ds is not None:
            st.success("✅ NetCDF file loaded successfully!")
            st.subheader("📊 Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🕒 Time Range", 
                         f"{ds.TIME.min().dt.strftime('%Y-%m-%d').values} to {ds.TIME.max().dt.strftime('%Y-%m-%d').values}")
            with col2:
                st.metric("🌍 Spatial Coverage", 
                         f"{len(ds.LATITUDE)} × {len(ds.LONGITUDE)} grid")
            with col3:
                st.metric("📈 Total Records", 
                         f"{len(ds.TIME):,} time steps")
            st.subheader("🗺️ Coordinate Information")
            coord_col1, coord_col2 = st.columns(2)
            with coord_col1:
                st.write("**Latitude Range:**")
                st.write(f"Min: {float(ds.LATITUDE.min()):.3f}°")
                st.write(f"Max: {float(ds.LATITUDE.max()):.3f}°")
            with coord_col2:
                st.write("**Longitude Range:**")
                st.write(f"Min: {float(ds.LONGITUDE.min()):.3f}°")
                st.write(f"Max: {float(ds.LONGITUDE.max()):.3f}°")
            with st.expander("🔍 Dataset Preview", expanded=False):
                st.write("**Dataset Structure:**")
                st.text(str(ds))
                st.write("**Sample Rainfall Values:**")
                sample_data = ds.RAINFALL.isel(TIME=slice(0, 5)).values
                st.write(f"Shape: {sample_data.shape}")
                st.write(f"Sample values: {sample_data.flatten()[:10]}")
    
    if ds is not None:
        st.sidebar.header("🎛️ Dashboard Controls")
        
        st.sidebar.subheader("📍 Location Selection")
        location_type = st.sidebar.radio(
            "Select Location Type",
            ["Entire Region", "Specific Point(s)", "Area of Interest (GeoJSON or Shapefile)"]
        )

        selected_points, aoi_geometry, location_name = None, None, "Entire Region"
        
        if location_type == "Specific Point(s)":
            st.sidebar.subheader("Select Point(s) on Map")
            point_map = create_point_selection_map(ds)
            map_output = st_folium(point_map, width=700, height=400, key="point_selection")
            
            selected_points = []
            if map_output.get('last_clicked'):
                selected_points.append((map_output['last_clicked']['lat'], map_output['last_clicked']['lng']))
            elif map_output.get('all_drawings'):
                for feature in map_output['all_drawings']['features']:
                    if feature['geometry']['type'] == 'Point':
                        coords = feature['geometry']['coordinates']
                        selected_points.append((coords[1], coords[0]))
            
            if selected_points:
                location_name = f"{len(selected_points)} Point(s) Selected"
                st.sidebar.success(f"Selected {len(selected_points)} point(s)")
            else:
                st.sidebar.info("Click on the map to select point(s)")
        
        elif location_type == "Area of Interest (GeoJSON or Shapefile)":
            st.sidebar.subheader("Upload AOI")
            st.sidebar.markdown(
                '<span style="font-size:1.2em;">🗺️ <b>Prepared GeoJSON file:</b> '
                '<a href="https://geojson.io/#map=9.03/25.6376/91.7331" target="_blank">'
                'https://geojson.io/#map=9.03/25.6376/91.7331</a></span>',
                unsafe_allow_html=True
            )
            aoi_type = st.sidebar.radio("AOI Format", ["GeoJSON", "Shapefile"])
            if aoi_type == "GeoJSON":
                uploaded_geojson = st.sidebar.file_uploader("Upload GeoJSON file", type=['geojson', 'json'])
                if uploaded_geojson:
                    aoi_geometry, _ = load_geojson_file(uploaded_geojson)
                    location_name = "Custom AOI (GeoJSON)"
            elif aoi_type == "Shapefile":
                uploaded_shp_files = st.sidebar.file_uploader(
                    "Upload Shapefile components (.shp, .shx, .dbf, .prj)",
                    type=['shp', 'shx', 'dbf', 'prj'],
                    accept_multiple_files=True
                )
                if uploaded_shp_files:
                    aoi_geometry = load_shapefile(uploaded_shp_files)
                    location_name = "Custom AOI (Shapefile)"
        
        if location_type == "Specific Point(s)" and selected_points:
            lats = [point[0] for point in selected_points]
            lons = [point[1] for point in selected_points]
            ds_aoi = ds.sel(LATITUDE=lats, LONGITUDE=lons, method='nearest')
        elif location_type == "Area of Interest (GeoJSON or Shapefile)" and aoi_geometry is not None:
            # If aoi_geometry is a GeoDataFrame, use it directly; else, convert
            if isinstance(aoi_geometry, gpd.GeoDataFrame):
                mask = aoi_geometry
            else:
                mask = gpd.GeoDataFrame(geometry=[aoi_geometry.unary_union], crs="EPSG:4326")
            ds_aoi = ds
            if not hasattr(ds_aoi, "rio") or not ds_aoi.rio.crs:
                ds_aoi = ds_aoi.rio.write_crs("EPSG:4326")
            ds_aoi = ds_aoi.rio.clip(mask.geometry, mask.crs, drop=True)
        else:
            ds_aoi = ds

        if not ds_aoi['RAINFALL'].size:
            st.error("No valid rainfall data available for the selected location.")
            return

        st.sidebar.subheader("🌍 Temporal Settings")
        min_year = int(ds_aoi.TIME.min().dt.year)
        max_year = int(ds_aoi.TIME.max().dt.year)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_year = st.selectbox("Start Year", range(min_year, max_year + 1), index=0)
        with col2:
            end_year = st.selectbox("End Year", range(min_year, max_year + 1), index=len(range(min_year, max_year + 1)) - 1)
        
        if start_year > end_year:
            st.sidebar.error("Start year must be less than or equal to end year")
            return
        
        time_period = (f"{start_year}-01-01", f"{end_year}-12-31")
        
        st.sidebar.subheader("🔬 Analysis Mode")
        analysis_mode = st.sidebar.selectbox(
            "Select Analysis Mode",
            ["Basic Analysis", "Geospatial Mapping", "Drought Analysis", 
             "Extreme Events", "Time Series", "Rainfall Parametric Solution"]
        )

        if analysis_mode in ["Geospatial Mapping", "Extreme Events", "Drought Analysis"]:
            st.sidebar.subheader("🛠️ Analysis Parameters")
            if analysis_mode == "Extreme Events":
                threshold = st.sidebar.slider(
                    "Rainfall Threshold (mm/day)",
                    min_value=10.0,
                    max_value=100.0,
                    value=50.0,
                    step=5.0
                )
            elif analysis_mode == "Drought Analysis":
                timescale = st.sidebar.slider(
                    "SPI Timescale (months)",
                    min_value=1,
                    max_value=24,
                    value=3,
                    step=1
                )
        
        st.header(f"📈 {analysis_mode}")
        
        if analysis_mode == "Basic Analysis":
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            time_subset = ds_aoi.sel(TIME=slice(time_period[0], time_period[1]))
            if not time_subset['RAINFALL'].size:
                st.error("No valid rainfall data available for the selected time period.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            data = time_subset['RAINFALL'].mean(dim=['LATITUDE', 'LONGITUDE'])
            data_df = data.to_dataframe().reset_index()
            st.download_button(
                label="Download Mean Rainfall Data (CSV)",
                data=data_df.to_csv(index=False),
                file_name="mean_rainfall.csv",
                mime="text/csv"
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Mean Rainfall", f"{float(data.mean()):.2f} mm/day")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Max Rainfall", f"{float(data.max()):.2f} mm/day")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Min Rainfall", f"{float(data.min()):.2f} mm/day")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Std Dev", f"{float(data.std()):.2f} mm/day")
                st.markdown('</div>', unsafe_allow_html=True)
            
            fig, ts_df = create_time_series_plots(ds_aoi, location_name, selected_points=selected_points)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_mode == "Geospatial Mapping":
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            map_type = st.selectbox(
                "Map Type",
                [
                    "Mean Rainfall",
                    "Rainfall Variability",
                    "Monthly Cumulative Average Rainfall",
                    "Custom Date Range Cumulative Rainfall"
                ]
            )
            
            selected_month, selected_week, custom_start, custom_end = None, None, None, None
            if map_type == "Monthly Cumulative Average Rainfall":
                months = list(range(1, 13))
                selected_month = st.selectbox("Select Month", months, format_func=lambda x: pd.Timestamp(month=x, day=1, year=2000).strftime('%B'))
            if map_type == "Weekly Cumulative Average Rainfall":
                weeks = list(range(1, 53))
                selected_week = st.selectbox("Select Week (ISO)", weeks)
            if map_type == "Custom Date Range Cumulative Rainfall":
                custom_col1, custom_col2 = st.columns(2)
                with custom_col1:
                    custom_start = st.date_input("Start Date", pd.to_datetime(time_period[0]))
                with custom_col2:
                    custom_end = st.date_input("End Date", pd.to_datetime(time_period[1]))
                if custom_start > custom_end:
                    st.error("Start date must be before end date.")
                    st.stop()
            
            if map_type == "Mean Rainfall":
                m = create_folium_map(ds_aoi, analysis_type="mean_rainfall", aoi_geometry=aoi_geometry, 
                                    time_period=time_period, selected_points=selected_points)
                # Add download button
                df = pd.DataFrame(data_array.values, columns=["Mean Rainfall (mm/day)"])
                st.download_button(
                    label="Download Map Data (CSV)",
                    data=df.to_csv(index=False),
                    file_name="map_data.csv",
                    mime="text/csv"
                )
            elif map_type == "Rainfall Variability":
                m = create_folium_map(ds_aoi, analysis_type="rainfall_variability", aoi_geometry=aoi_geometry, 
                                    time_period=time_period, selected_points=selected_points)
            elif map_type == "Cumulative Average Rainfall":
                m = create_folium_map(ds_aoi, analysis_type="cumulative_average", aoi_geometry=aoi_geometry, 
                                    time_period=time_period, selected_points=selected_points)
            elif map_type == "Monthly Cumulative Average Rainfall":
                m = create_folium_map(ds_aoi, analysis_type="monthly_cumulative_average", aoi_geometry=aoi_geometry, 
                                    time_period=time_period, selected_month=selected_month, selected_points=selected_points)
            elif map_type == "Weekly Cumulative Average Rainfall":
                m = create_folium_map(ds_aoi, analysis_type="weekly_cumulative_average", aoi_geometry=aoi_geometry, 
                                    time_period=time_period, selected_week=selected_week, selected_points=selected_points)
            elif map_type == "Custom Date Range Cumulative Rainfall":
                m = create_folium_map(ds_aoi, analysis_type="custom_range_cumulative", aoi_geometry=aoi_geometry, 
                                    time_period=(str(custom_start), str(custom_end)), selected_points=selected_points)
            else:
                m = None
            
            if m is not None:
                st_folium(m, width=1200, height=600)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_mode == "Drought Analysis":
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            with st.spinner("Calculating SPI..."):
                time_subset = ds_aoi.sel(TIME=slice(time_period[0], time_period[1]))
                if not time_subset['RAINFALL'].size:
                    st.error("No valid rainfall data available for the selected time period.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return
                if location_type == "Specific Point(s)" and selected_points:
                    lats = [point[0] for point in selected_points]
                    lons = [point[1] for point in selected_points]
                    precip_data = time_subset.sel(LATITUDE=lats, LONGITUDE=lons, method='nearest')['RAINFALL'].mean(dim=['LATITUDE', 'LONGITUDE'])
                    spi_data = calculate_spi(precip_data, timescale)
                    m = None
                else:
                    def spi_func(x):
                        return calculate_spi(x, timescale)
                    spi_grid = xr.apply_ufunc(
                        spi_func,
                        time_subset['RAINFALL'],
                        input_core_dims=[['TIME']],
                        output_core_dims=[['TIME']],
                        vectorize=True
                    )
                    spi_map = spi_grid.isel(TIME=-1)
                    m = create_folium_map(ds_aoi, analysis_type="drought_spi", aoi_geometry=aoi_geometry,
                                        time_period=time_period, spi_data=spi_map.values, selected_points=selected_points)
                if m is not None:
                    st_folium(m, width=1200, height=600)
                
                fig = go.Figure()
                if location_type == "Specific Point(s)" and selected_points:
                    fig.add_trace(
                        go.Scatter(
                            x=precip_data['TIME'],
                            y=spi_data,
                            mode='lines',
                            name='SPI',
                            line=dict(color='purple')
                        )
                    )
                else:
                    mean_spi = spi_grid.mean(dim=['LATITUDE', 'LONGITUDE'])
                    fig.add_trace(
                        go.Scatter(
                            x=ds_aoi['TIME'].values,
                            y=mean_spi,
                            mode='lines',
                            name='Mean SPI',
                            line=dict(color='purple')
                        )
                    )
                fig.update_layout(
                    title=f"SPI Timescale {timescale} Months - {location_name}",
                    xaxis_title="Date",
                    yaxis_title="SPI Value",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_mode == "Extreme Events":
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            m = create_folium_map(ds_aoi, analysis_type="extreme_events", aoi_geometry=aoi_geometry, 
                                threshold=threshold, time_period=time_period, selected_points=selected_points)
            st_folium(m, width=1200, height=600)
            time_subset = ds_aoi.sel(TIME=slice(time_period[0], time_period[1]))
            if time_subset['RAINFALL'].size:
                extreme_count = (time_subset['RAINFALL'] > threshold).sum(dim='TIME')
                extreme_df = extreme_count.to_dataframe().reset_index()
                st.download_button(
                    label="Download Extreme Events Data (CSV)",
                    data=extreme_df.to_csv(index=False),
                    file_name="extreme_events.csv",
                    mime="text/csv"
                )
                st.write(f"Total Extreme Events (>{threshold} mm/day): {int(extreme_count.mean())}")
            else:
                st.error("No valid rainfall data available for extreme events analysis.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_mode == "Time Series":
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            extreme_threshold = st.slider(
                "Extreme Event Threshold (mm/day) for Time Series",
                min_value=10.0,
                max_value=100.0,
                value=50.0,
                step=5.0
            )
            ts_min_year = int(ds_aoi.TIME.min().dt.year)
            ts_max_year = int(ds_aoi.TIME.max().dt.year)
            ts_col1, ts_col2 = st.columns(2)
            with ts_col1:
                ts_start_year = st.selectbox("Start Year for Daily Time Series", range(ts_min_year, ts_max_year + 1), index=0)
            with ts_col2:
                ts_end_year = st.selectbox("End Year for Daily Time Series", range(ts_min_year, ts_max_year + 1), index=(ts_max_year - ts_min_year))
            if ts_start_year > ts_end_year:
                st.error("Start year must be less than or equal to end year for daily time series.")
                st.stop()
            fig, ts_df = create_time_series_plots(
                ds_aoi, location_name, selected_points=selected_points,
                extreme_threshold=extreme_threshold,
                daily_start_year=ts_start_year,
                daily_end_year=ts_end_year
            )
            st.plotly_chart(fig, use_container_width=True)

            # Prepare individual DataFrames
            daily_df = ts_df[['TIME', 'RAINFALL']].copy()
            daily_df = daily_df[(ts_df['Year'] >= ts_start_year) & (ts_df['Year'] <= ts_end_year)]
            monthly_avg_df = ts_df.groupby('Month')['RAINFALL'].mean().reset_index()
            annual_totals_df = ts_df.groupby('Year')['RAINFALL'].sum().reset_index()
            extreme_events_df = ts_df[ts_df['RAINFALL'] > extreme_threshold].copy()
            extreme_by_year_df = extreme_events_df.groupby('Year').size().reset_index(name='Extreme_Count')
            weekly_cum_df = ts_df.groupby('Week')['RAINFALL'].sum().reset_index()
            monthly_cum = ts_df.groupby(['Year', 'Month'])['RAINFALL'].sum().reset_index()
            monthly_cum_avg_df = monthly_cum.groupby('Month')['RAINFALL'].mean().reset_index()

            st.markdown("#### Download Individual Analysis Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="Download Daily Rainfall (CSV)",
                    data=daily_df.to_csv(index=False),
                    file_name="daily_rainfall.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download Monthly Avg (CSV)",
                    data=monthly_avg_df.to_csv(index=False),
                    file_name="monthly_avg_rainfall.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="Download Annual Totals (CSV)",
                    data=annual_totals_df.to_csv(index=False),
                    file_name="annual_totals.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download Extreme Events (CSV)",
                    data=extreme_by_year_df.to_csv(index=False),
                    file_name="extreme_events_by_year.csv",
                    mime="text/csv"
                )
            with col3:
                st.download_button(
                    label="Download Weekly Cumulative (CSV)",
                    data=weekly_cum_df.to_csv(index=False),
                    file_name="weekly_cumulative.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download Monthly Cumulative Avg (CSV)",
                    data=monthly_cum_avg_df.to_csv(index=False),
                    file_name="monthly_cumulative_avg.csv",
                    mime="text/csv"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_mode == "Rainfall Parametric Solution":
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.header("Rainfall Parametric Solution")

            time_subset = ds_aoi.sel(TIME=slice(time_period[0], time_period[1]))
            if not time_subset['RAINFALL'].size:
                st.error("No valid rainfall data available for the selected time period.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            # 1. Descriptive statistics
            with st.expander("ℹ️ Descriptive Statistics - What is this?"):
                st.markdown("""
                **What is this?**  
                Descriptive statistics summarize the main features of rainfall data, including mean, standard deviation, min, max, skewness, and kurtosis.

                **Why we use?**  
                To quickly understand the distribution, variability, and shape of rainfall data.

                **Interpretation/Reference:**  
                - Mean: Average rainfall  
                - Std Dev: Variability  
                - Skewness: Asymmetry (positive = right tail)  
                - Kurtosis: "Peakedness" (high = more outliers)  
                [Reference: Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences]
                """)

            st.subheader("Descriptive Statistics")
            rain = time_subset['RAINFALL'].mean(dim=['LATITUDE', 'LONGITUDE']).values
            stats_df = pd.DataFrame({
                "Mean": [np.nanmean(rain)],
                "Std Dev": [np.nanstd(rain)],
                "Min": [np.nanmin(rain)],
                "Max": [np.nanmax(rain)],
                "Skewness": [pd.Series(rain).skew()],
                "Kurtosis": [pd.Series(rain).kurt()]
            })
            st.dataframe(stats_df)

            # 2. Histogram and PDF fit
            with st.expander("ℹ️ Histogram & Parametric Fit - What is this?"):
                st.markdown("""
                **What is this?**  
                A histogram shows the frequency of rainfall values. Parametric fits (Normal, Gamma) overlay theoretical distributions.

                **Why we use?**  
                To visually assess how well rainfall data matches common probability distributions.

                **Interpretation/Reference:**  
                - Good fit: Model can be used for risk/return period analysis  
                - Gamma is often used for rainfall  
                [Reference: Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences]
                """)

            st.subheader("Histogram & Parametric Fit")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=rain, nbinsx=50, name="Rainfall Histogram", marker_color='skyblue'))
            # Fit normal and gamma
            mu, std = norm.fit(rain)
            xmin, xmax = np.nanmin(rain), np.nanmax(rain)
            x = np.linspace(xmin, xmax, 100)
            p_norm = norm.pdf(x, mu, std)
            fig.add_trace(go.Scatter(x=x, y=p_norm * len(rain) * (xmax-xmin)/50, name="Normal Fit", line=dict(color='red')))
            ag, locg, scaleg = gamma.fit(rain[rain > 0])
            p_gamma = gamma.pdf(x, ag, locg, scaleg)
            fig.add_trace(go.Scatter(x=x, y=p_gamma * len(rain) * (xmax-xmin)/50, name="Gamma Fit", line=dict(color='green')))
            fig.update_layout(title="Rainfall Histogram with Parametric Fits", xaxis_title="Rainfall (mm)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            # 3. Return period analysis (Gumbel/Extreme Value)
            with st.expander("ℹ️ Return Period Analysis - What is this?"):
                st.markdown("""
                **What is this?**  
                Return period (or recurrence interval) estimates how often a certain rainfall magnitude is expected.

                **Why we use?**  
                For infrastructure design and flood risk assessment.

                **Interpretation/Reference:**  
                - A 100-year event has a 1% chance of occurring in any year  
                [Reference: Chow, V.T. (1964). Handbook of Applied Hydrology]
                """)

            st.subheader("Return Period Analysis (Gumbel/EV)")
            annual_max = time_subset['RAINFALL'].max(dim='TIME').values.flatten()
            annual_max = annual_max[~np.isnan(annual_max)]
            if len(annual_max) > 0:
                sorted_max = np.sort(annual_max)[::-1]
                n = len(sorted_max)
                ranks = np.arange(1, n+1)
                return_period = (n+1) / ranks
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=return_period, y=sorted_max, mode='markers+lines', name="Annual Max"))
                fig2.update_layout(xaxis_type="log", xaxis_title="Return Period (years)", yaxis_title="Rainfall (mm)",
                                   title="Return Period Plot (Annual Maxima)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Not enough data for return period analysis.")

            # 4. Probability of exceedance
            with st.expander("ℹ️ Probability of Exceedance - What is this?"):
                st.markdown("""
                **What is this?**  
                The probability that rainfall exceeds a user-defined threshold.

                **Why we use?**  
                To estimate risk for agriculture, flood management, or insurance.

                **Interpretation/Reference:**  
                - High probability: Exceedance is common  
                [Reference: WMO-No. 168, Guide to Hydrological Practices]
                """)

            st.subheader("Probability of Exceedance")
            threshold = st.number_input("Rainfall threshold (mm) for exceedance probability", min_value=0.0, value=50.0)
            prob_exceed = np.mean(rain > threshold)
            st.write(f"Probability that rainfall exceeds {threshold} mm: **{prob_exceed*100:.2f}%**")

            # 5. Download options
            with st.expander("ℹ️ Download Data - What is this?"):
                st.markdown("""
                **What is this?**  
                Download rainfall and analysis results for further use.

                **Why we use?**  
                For reporting, sharing, or advanced analysis.

                **Interpretation/Reference:**  
                - Data is in CSV/Excel format for compatibility  
                """)

            st.markdown("#### Download Parametric Analysis Data")
            st.download_button(
                label="Download Rainfall Data (CSV)",
                data=pd.DataFrame(rain, columns=["Rainfall"]).to_csv(index=False),
                file_name="rainfall_parametric_data.csv",
                mime="text/csv"
            )
            
            # 6. Probability Plots (Q-Q plots for Normal, Gamma, Gumbel)
            with st.expander("ℹ️ Probability Plots (Q-Q) - What is this?"):
                st.markdown("""
                **What is this?**  
                Q-Q plots compare observed rainfall quantiles to theoretical distributions.

                **Why we use?**  
                To visually check if rainfall follows a specific distribution.

                **Interpretation/Reference:**  
                - Points on the line: Good fit  
                [Reference: Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences]
                """)

            st.subheader("Probability Plots (Q-Q)")
            import scipy.stats as stats
            from scipy.stats import gumbel_r
            qq_fig = make_subplots(rows=1, cols=3, subplot_titles=["Normal Q-Q", "Gamma Q-Q", "Gumbel Q-Q"])
            norm_qq = stats.probplot(rain, dist="norm")
            qq_fig.add_trace(go.Scatter(x=norm_qq[0][0], y=norm_qq[0][1], mode='markers', name="Normal Q-Q"), row=1, col=1)
            qq_fig.add_trace(go.Line(x=norm_qq[0][0], y=norm_qq[1][1]*norm_qq[0][0]+norm_qq[1][0], line=dict(color='red')), row=1, col=1)
            gamma_qq = stats.probplot(rain[rain > 0], dist="gamma", sparams=(ag, locg, scaleg))
            qq_fig.add_trace(go.Scatter(x=gamma_qq[0][0], y=gamma_qq[0][1], mode='markers', name="Gamma Q-Q"), row=1, col=2)
            qq_fig.add_trace(go.Line(x=gamma_qq[0][0], y=gamma_qq[1][1]*gamma_qq[0][0]+gamma_qq[1][0], line=dict(color='green')), row=1, col=2)
            gumbel_qq = stats.probplot(rain, dist="gumbel_r", sparams=(locg, scaleg))
            qq_fig.add_trace(go.Scatter(x=gumbel_qq[0][0], y=gumbel_qq[0][1], mode='markers', name="Gumbel Q-Q"), row=1, col=3)
            qq_fig.add_trace(go.Line(x=gumbel_qq[0][0], y=gumbel_qq[1][1]*gumbel_qq[0][0]+gumbel_qq[1][0], line=dict(color='orange')), row=1, col=3)
            qq_fig.update_layout(height=400, width=1200, showlegend=False)
            st.plotly_chart(qq_fig, use_container_width=True)

            # 7. Rainfall Anomaly Detection (Z-score)
            with st.expander("ℹ️ Rainfall Anomaly Detection - What is this?"):
                st.markdown("""
                **What is this?**  
                Detects rainfall values that are unusually high or low (outliers) using Z-score.

                **Why we use?**  
                To identify extreme events or data quality issues.

                **Interpretation/Reference:**  
                - Z-score > 3: Unusual/extreme event  
                [Reference: WMO-No. 8, Guide to Meteorological Instruments and Methods of Observation]
                """)

            st.subheader("Rainfall Anomaly Detection")
            z_scores = (rain - np.nanmean(rain)) / np.nanstd(rain)
            anomaly_threshold = st.slider("Z-score threshold for anomaly", 2.0, 5.0, 3.0, step=0.1)
            anomalies = np.where(np.abs(z_scores) > anomaly_threshold)[0]
            st.write(f"Number of anomalies detected: {len(anomalies)}")
            if len(anomalies) > 0:
                st.dataframe(pd.DataFrame({
                    "Index": anomalies,
                    "Rainfall": rain[anomalies],
                    "Z-score": z_scores[anomalies]
                }))

            # 8. Seasonal Decomposition (Trend/Seasonality/Residual)
            with st.expander("ℹ️ Seasonal Decomposition (STL) - What is this?"):
                st.markdown("""
                **What is this?**  
                Decomposes rainfall into trend, seasonal, and residual components.

                **Why we use?**  
                To understand long-term changes and seasonal cycles.

                **Interpretation/Reference:**  
                - Trend: Long-term direction  
                - Seasonal: Repeating pattern  
                [Reference: Cleveland et al. (1990). STL: A Seasonal-Trend Decomposition Procedure]
                """)

            st.subheader("Seasonal Decomposition (STL)")
            try:
                from statsmodels.tsa.seasonal import STL
                rain_series = pd.Series(rain).interpolate()
                stl = STL(rain_series, period=12)
                res = stl.fit()
                st.line_chart(pd.DataFrame({
                    "Trend": res.trend,
                    "Seasonal": res.seasonal,
                    "Residual": res.resid
                }))
            except Exception as e:
                st.info("STL decomposition not available (requires statsmodels and sufficient data).")

            # 9. Rainfall Intensity-Duration-Frequency (IDF) Curve (Simple)
            with st.expander("ℹ️ Rainfall Intensity-Duration-Frequency (IDF) Curve - What is this?"):
                st.markdown("""
                **What is this?**  
                Shows the relationship between rainfall intensity, duration, and frequency.

                **Why we use?**  
                For engineering design (e.g., drainage, flood control).

                **Interpretation/Reference:**  
                - Higher intensity for shorter durations  
                [Reference: WMO-No. 1173, Manual on Flood Forecasting and Warning]
                """)

            st.subheader("Rainfall Intensity-Duration-Frequency (IDF) Curve")
            durations = [1, 3, 6, 12, 24]  # hours
            idf_data = []
            for d in durations:
                rolling_max = pd.Series(rain).rolling(window=d, min_periods=1).max()
                idf_data.append(np.nanpercentile(rolling_max, 99))  # 1% exceedance
            st.line_chart(pd.DataFrame({"Duration (h)": durations, "Intensity (mm/h)": idf_data}).set_index("Duration (h)"))

            # 10. Export all analysis as Excel
            with st.expander("ℹ️ Download All Analysis (Excel) - What is this?"):
                st.markdown("""
                **What is this?**  
                Download all rainfall analysis results in a single Excel file.

                **Why we use?**  
                For comprehensive reporting and sharing.

                **Interpretation/Reference:**  
                - Each sheet contains a different analysis result  
                """)

            st.markdown("#### Download All Analysis (Excel)")
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                pd.DataFrame(rain, columns=["Rainfall"]).to_excel(writer, sheet_name="Rainfall")
                stats_df.to_excel(writer, sheet_name="Statistics")
                if len(anomalies) > 0:
                    pd.DataFrame({
                        "Index": anomalies,
                        "Rainfall": rain[anomalies],
                        "Z-score": z_scores[anomalies]
                    }).to_excel(writer, sheet_name="Anomalies")
            st.download_button(
                label="Download All Analysis (Excel)",
                data=excel_buffer.getvalue(),
                file_name="rainfall_parametric_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
if __name__ == "__main__":
    main()