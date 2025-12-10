"""
Extract ERA5 environmental variables for TC tracks:
- 975-hPa convergence (-1 * divergence)
- 925-hPa vorticity
- 700-hPa specific humidity

For both developers and non-developers
Processes and saves data by year
"""
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import os
from glob import glob

# Settings
ERA5_BASE_DIR = '/gdex/data/d633000/e5.oper.an.pl'
DEVELOPER_FILE = './processed_tracks/developer_refined.xlsx'
NON_DEVELOPER_FILE = './processed_tracks/non_developers_12points.xlsx'
OUTPUT_DIR = './ERA5_extracted_years'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pressure levels (hPa)
PRESSURE_LEVELS = {
    'q': 700,      # specific humidity at 700 hPa
    'vo': 925,     # vorticity at 925 hPa
    'd': 975       # divergence at 975 hPa
}

# Variable codes
VAR_CODES = {
    'q': '128_133_q',
    'vo': '128_138_vo',
    'd': '128_155_d'
}


def find_era5_file(date_str, var_code):
    """
    Find ERA5 file for given date and variable
    date_str: format YYYYMMDDHH
    var_code: 'q', 'vo', or 'd'
    """
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    
    # Construct file pattern
    year_month_dir = f"{year}{month}"
    var_full = VAR_CODES[var_code]
    
    # Files are named like: e5.oper.an.pl.128_133_q.ll025sc.1980010100_1980010123.nc
    pattern = f"{ERA5_BASE_DIR}/{year_month_dir}/e5.oper.an.pl.{var_full}.ll025sc.{year}{month}{day}00_{year}{month}{day}23.nc"
    
    if os.path.exists(pattern):
        return pattern
    else:
        return None


def extract_variable_at_point(ds, var_name, lat, lon, pressure_level, time_idx):
    """
    Extract variable value at specific location and pressure level
    
    Parameters:
    - ds: xarray Dataset
    - var_name: variable name in dataset
    - lat: latitude
    - lon: longitude (0-360)
    - pressure_level: pressure level in hPa
    - time_idx: time index (0-23 for hourly data)
    
    Returns:
    - value at the point
    """
    # Convert lon to 0-360 if needed
    if lon < 0:
        lon = lon + 360
    
    # Get pressure level dimension name
    lev_dim = None
    for dim in ds.dims:
        if 'lev' in dim.lower() or 'level' in dim.lower() or 'pressure' in dim.lower():
            lev_dim = dim
            break
    
    if lev_dim is None:
        print(f"[WARNING] Could not find pressure level dimension in dataset")
        return np.nan
    
    try:
        # Select pressure level
        ds_lev = ds.sel({lev_dim: pressure_level}, method='nearest')
        
        # Get variable data
        var_data = ds_lev[var_name]
        
        # Select time
        time_dim = [d for d in var_data.dims if 'time' in d.lower()][0]
        var_data = var_data.isel({time_dim: time_idx})
        
        # Select lat/lon (nearest neighbor)
        lat_dim = [d for d in var_data.dims if 'lat' in d.lower()][0]
        lon_dim = [d for d in var_data.dims if 'lon' in d.lower()][0]
        
        value = var_data.sel({lat_dim: lat, lon_dim: lon}, method='nearest').values
        
        return float(value)
    
    except Exception as e:
        print(f"[ERROR] Error extracting {var_name} at lat={lat}, lon={lon}: {e}")
        return np.nan


def process_year(year_df, year, track_type):
    """
    Process tracks for a single year
    
    Parameters:
    - year_df: DataFrame containing tracks for this year
    - year: year to process
    - track_type: 'developer' or 'non-developer'
    
    Returns:
    - DataFrame with ERA5 variables added
    """
    print(f"\n[INFO] Processing {track_type} year {year}: {len(year_df)} track points, {year_df['id'].nunique()} unique tracks")
    
    # Add columns for ERA5 variables
    year_df['q_700'] = np.nan
    year_df['vo_925'] = np.nan
    year_df['conv_975'] = np.nan  # convergence = -1 * divergence
    
    # Group by date to minimize file I/O
    year_df['date_str'] = year_df['time'].astype(str).str[:8]  # YYYYMMDD
    year_df['hour'] = year_df['time'].astype(str).str[8:10].astype(int)
    
    unique_dates = year_df['date_str'].unique()
    print(f"[INFO] Processing {len(unique_dates)} unique dates for year {year}")
    
    for date_idx, date in enumerate(unique_dates, 1):
        date_mask = year_df['date_str'] == date
        date_df = year_df[date_mask]
        
        if date_idx % 30 == 0 or date_idx == len(unique_dates):
            print(f"[INFO]   Date {date_idx}/{len(unique_dates)}: {date}")
        
        # Load ERA5 files for this date
        era5_files = {}
        datasets = {}
        
        for var in ['q', 'vo', 'd']:
            file_path = find_era5_file(date + '00', var)  # Add hour 00
            if file_path:
                try:
                    datasets[var] = xr.open_dataset(file_path)
                    era5_files[var] = file_path
                except Exception as e:
                    print(f"[WARNING] Could not open {file_path}: {e}")
            else:
                print(f"[WARNING] File not found for date={date}, var={var}")
        
        # Process each track point for this date
        for idx in date_df.index:
            row = year_df.loc[idx]
            lat = row['lat']
            lon = row['lon']
            hour = row['hour']
            
            # Extract q at 700 hPa
            if 'q' in datasets:
                try:
                    q_val = extract_variable_at_point(datasets['q'], 'Q', lat, lon, 700, hour)
                    year_df.at[idx, 'q_700'] = q_val
                except Exception as e:
                    print(f"[WARNING] Error extracting q for idx {idx}: {e}")
            
            # Extract vorticity at 925 hPa
            if 'vo' in datasets:
                try:
                    vo_val = extract_variable_at_point(datasets['vo'], 'VO', lat, lon, 925, hour)
                    year_df.at[idx, 'vo_925'] = vo_val
                except Exception as e:
                    print(f"[WARNING] Error extracting vo for idx {idx}: {e}")
            
            # Extract divergence at 975 hPa and convert to convergence
            if 'd' in datasets:
                try:
                    d_val = extract_variable_at_point(datasets['d'], 'D', lat, lon, 975, hour)
                    year_df.at[idx, 'conv_975'] = -1.0 * d_val  # convergence = -divergence
                except Exception as e:
                    print(f"[WARNING] Error extracting d for idx {idx}: {e}")
        
        # Close datasets to free memory
        for ds in datasets.values():
            ds.close()
    
    # Remove temporary columns
    year_df = year_df.drop(columns=['date_str', 'hour'])
    
    # Print statistics for this year
    print(f"[INFO] Year {year} Statistics:")
    print(f"  - Total track points: {len(year_df)}")
    print(f"  - Unique tracks: {year_df['id'].nunique()}")
    print(f"  - q_700: {year_df['q_700'].notna().sum()}/{len(year_df)} values")
    print(f"  - vo_925: {year_df['vo_925'].notna().sum()}/{len(year_df)} values")
    print(f"  - conv_975: {year_df['conv_975'].notna().sum()}/{len(year_df)} values")
    
    return year_df


def save_year_to_netcdf(year_df, year, track_type, output_dir):
    """
    Save year data to NetCDF file
    """
    # Convert time to datetime
    year_df['time_dt'] = pd.to_datetime(year_df['time'].astype(str), format='%Y%m%d%H')
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'id': (['track_point'], year_df['id'].values),
            'time': (['track_point'], year_df['time'].values),
            'lat': (['track_point'], year_df['lat'].values),
            'lon': (['track_point'], year_df['lon'].values),
            'year': (['track_point'], year_df['year'].values),
            'month': (['track_point'], year_df['month'].values),
            'relative_vorticity': (['track_point'], year_df['relative_vorticity'].values),
            'q_700': (['track_point'], year_df['q_700'].values),
            'vo_925': (['track_point'], year_df['vo_925'].values),
            'conv_975': (['track_point'], year_df['conv_975'].values),
        },
        coords={
            'track_point': np.arange(len(year_df)),
            'time_dt': (['track_point'], year_df['time_dt'].values),
        },
        attrs={
            'description': f'{track_type.capitalize()} tracks with ERA5 environmental variables',
            'year': year,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'q_700_units': 'kg/kg (specific humidity at 700 hPa)',
            'vo_925_units': 's-1 (vorticity at 925 hPa)',
            'conv_975_units': 's-1 (convergence at 975 hPa, computed as -1*divergence)',
            'relative_vorticity_units': 'CVU (1.0 CVU = 1.0e-5 s-1)',
        }
    )
    
    # Save as NetCDF
    output_file = os.path.join(output_dir, f'{track_type}s_{year}_with_ERA5.nc')
    ds.to_netcdf(output_file)
    print(f"[INFO] Saved to {output_file}")
    
    ds.close()


def process_tracks_by_year(track_file, track_type, year_to_process=None):
    """
    Process tracks year by year and save individual files
    
    Parameters:
    - track_file: path to developer or non-developer Excel file
    - track_type: 'developer' or 'non-developer'
    - year_to_process: specific year to process (int), or None to process all years
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Processing {track_type} tracks from {track_file}")
    print(f"{'='*60}")
    
    # Load track data
    df = pd.read_excel(track_file)
    print(f"[INFO] Loaded {len(df)} total track points, {df['id'].nunique()} unique tracks")
    
    # Get unique years
    all_years = sorted(df['year'].unique())
    
    # Determine which years to process
    if year_to_process is not None:
        if year_to_process not in all_years:
            print(f"[ERROR] Year {year_to_process} not found in data. Available years: {all_years[0]}-{all_years[-1]}")
            return
        years = [year_to_process]
        print(f"[INFO] Processing single year: {year_to_process}")
    else:
        years = all_years
        print(f"[INFO] Years to process: {years[0]}-{years[-1]} ({len(years)} years)")
    
    # Process each year
    for year_idx, year in enumerate(years, 1):
        print(f"\n[INFO] ===== Processing Year {year} ({year_idx}/{len(years)}) =====")
        
        # Filter data for this year
        year_df = df[df['year'] == year].copy().reset_index(drop=True)
        
        if len(year_df) == 0:
            print(f"[WARNING] No data for year {year}, skipping...")
            continue
        
        # Process this year
        year_df_processed = process_year(year_df, year, track_type)
        
        # Save to NetCDF
        save_year_to_netcdf(year_df_processed, year, track_type, OUTPUT_DIR)
    
    print(f"\n[INFO] Completed processing all {len(years)} years for {track_type}s")


# # Process developers by year
# print("\n" + "="*60)
# print("[INFO] PROCESSING DEVELOPER TRACKS")
# print("="*60)
# process_tracks_by_year(DEVELOPER_FILE, 'developer')

# Process non-developers by year
print("\n" + "="*60)
print("[INFO] PROCESSING NON-DEVELOPER TRACKS")
print("="*60)
process_tracks_by_year(NON_DEVELOPER_FILE, 'non-developer_12points', year_to_process=2021)

# print("\n" + "="*60)
# print("[INFO] ALL PROCESSING COMPLETE")
# print("="*60)
# print(f"\nOutput directory: {OUTPUT_DIR}")
# print(f"Files saved: developers_YYYY_with_ERA5.nc and non_developers_YYYY_with_ERA5.nc")
# print(f"            for years 1980-2022"8