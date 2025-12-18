"""
Extract ERA5 environmental variables for TC tracks:
- 975-hPa convergence (-1 * divergence)
- 925-hPa vorticity
- 700-hPa specific humidity
For:
- inner region (1.5 deg box)
- outer region (5 deg box)

For both developers and non-developers
Processes and saves data by year
"""
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import os
from glob import glob

ERA5_dir = "/gdex/data/d633000/e5.oper.an.pl"
developer_file = "./processed_tracks/developer_refined.xlsx"
nondeveloper_file = "./processed_tracks/non_developers_16points.xlsx"
output_dir = "./ERA5_extracted_6hourly"

os.makedirs(output_dir, exist_ok=True)

var_code_dict = {
    "q": "128_133_q",
    "vo": "128_138_vo",
    "d": "128_155_d"
}


def find_era5_file(date_str, var_code):
    """
    Find ERA5 file for given date and variable
    date_str: format YYYYMMDDHH
    var_code: "q", "vo", or "d"
    """
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    
    year_month_dir = f"{year}{month}"
    var_full = var_code_dict[var_code]
    
    # Files are named like: e5.oper.an.pl.128_133_q.ll025sc.1980010100_1980010123.nc
    pattern = f"{ERA5_dir}/{year_month_dir}/e5.oper.an.pl.{var_full}.ll025sc.{year}{month}{day}00_{year}{month}{day}23.nc"
    
    if os.path.exists(pattern):
        return pattern
    else:
        return None


def extract_variable(ds, var_name, lat, lon, pressure_level, time_idx, d_deg=0):
    """
    Extract mean variable value at a specific pressure level for defined box
    
    Parameters:
    - ds: xarray Dataset
    - var_name: variable name in dataset
    - lat: latitude
    - lon: longitude (0-360)
    - pressure_level: pressure level in hPa
    - time_idx: time index (0-23 for hourly data)
    - d_deg: degree length of box to average over, default is zero
    
    Returns:
    - value at the point
    """
    if lon < 0:
        lon = lon + 360
    
    lev_dim = None
    for dim in ds.dims:
        if "lev" in dim.lower() or "level" in dim.lower() or "pressure" in dim.lower():
            lev_dim = dim
            break
    
    if lev_dim is None:
        print("[WARNING] Could not find pressure level dimension in dataset")
        return np.nan
    
    try:
        ds_lev = ds.sel({lev_dim: pressure_level}, method="nearest")
        var_data = ds_lev[var_name]
        time_dim = [d for d in var_data.dims if "time" in d.lower()][0]
        var_data = var_data.isel({time_dim: time_idx})
        
        # value = var_data.sel({lat_dim: lat, lon_dim: lon}, method="nearest").values
        value = var_data.sel(
                             latitude=slice(lat+d_deg/2,lat-d_deg/2),
                             longitude=slice(lon-d_deg/2,lon+d_deg/2)
                            ).mean(dim=["latitude","longitude"]).values
        
        return float(value)
    
    except Exception as e:
        print(f"[ERROR] Error extracting {var_name} at lat={lat}, lon={lon}: {e}")
        return np.nan


def extract_tcwv(ds, lat, lon, time_idx):
    """
    Extract total column water vapor (TCWV) at a single grid point.

    Parameters:
    - ds: xarray Dataset containing Q on pressure levels
    - lat, lon: location (lon in 0–360)
    - time_idx: hour index (0–23)

    Returns:
    - TCWV in mm
    """
    if lon < 0:
        lon = lon + 360

    g = 9.80665  # m s-2

    # Find pressure level dimension
    lev_dim = None
    for dim in ds.dims:
        if "lev" in dim.lower() or "level" in dim.lower():
            lev_dim = dim
            break

    if lev_dim is None:
        print("[WARNING] No pressure level dimension found for TCWV")
        return np.nan

    try:
        q = ds["Q"]

        # Time selection
        time_dim = [d for d in q.dims if "time" in d.lower()][0]
        q = q.isel({time_dim: time_idx})

        # Point selection
        q = q.sel(
            latitude=lat,
            longitude=lon,
            method="nearest"
        )

        # Pressure levels (hPa → Pa)
        p = q[lev_dim].values * 100.0

        # Sort top → bottom if needed
        sort_idx = np.argsort(p)
        p = p[sort_idx]
        qv = q.values[sort_idx]

        # Vertical integral
        tcwv = np.trapezoid(qv, p) / g  # kg m-2 = mm

        return float(tcwv)

    except Exception as e:
        print(f"[ERROR] TCWV extraction failed: {e}")
        return np.nan


def process_year(year_df, year, track_type):
    """
    Process tracks for a single year
    
    Parameters:
    - year_df: DataFrame containing tracks for this year
    - year: year to process
    - track_type: "developer" or "non-developer"
    
    Returns:
    - DataFrame with ERA5 variables added
    """
    print(f"\nProcessing {track_type} year {year}: {len(year_df)} track points, {year_df['id'].nunique()} unique tracks")
    
    year_df["q_700_inner"] = np.nan
    year_df["vo_925_inner"] = np.nan
    year_df["conv_975_inner"] = np.nan  # convergence = -1 * divergence
    year_df["q_700_outer"] = np.nan
    year_df["vo_925_outer"] = np.nan
    year_df["conv_975_outer"] = np.nan  # convergence = -1 * divergence
    year_df["tcwv"] = np.nan
    
    year_df["date_str"] = year_df["time"].astype(str).str[:8]  # YYYYMMDD
    year_df["hour"] = year_df["time"].astype(str).str[8:10].astype(int)
    
    unique_dates = year_df["date_str"].unique()
    print(f"Processing {len(unique_dates)} unique dates for year {year}")
    
    for date_idx, date in enumerate(unique_dates, 1):
        date_mask = year_df["date_str"] == date
        date_df = year_df[date_mask]
        
        if date_idx % 30 == 0 or date_idx == len(unique_dates):
            print(f"  Date {date_idx}/{len(unique_dates)}: {date}")
        
        # Load ERA5 files for this date
        era5_files = {}
        datasets = {}
        
        for var in ["q", "vo", "d"]:
            file_path = find_era5_file(date + "00", var)  # Add hour 00
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
            lat = row["lat"]
            lon = row["lon"]
            hour = row["hour"]
            
            if "q" in datasets:
                try:
                    q_val_inner = extract_variable(datasets["q"], "Q", lat, lon, 700, hour, d_deg=1.5)
                    q_val_outer = extract_variable(datasets["q"], "Q", lat, lon, 700, hour, d_deg=5)
                    year_df.at[idx, "q_700_inner"] = q_val_inner
                    year_df.at[idx, "q_700_outer"] = q_val_outer
                except Exception as e:
                    print(f"[WARNING] Error extracting q for idx {idx}: {e}")

            if "q" in datasets:
                try:
                    tcwv_val = extract_tcwv(datasets["q"], lat, lon, hour)
                    year_df.at[idx, "tcwv"] = tcwv_val
                except Exception as e:
                    print(f"[WARNING] Error extracting TCWV for idx {idx}: {e}")

            
            if "vo" in datasets:
                try:
                    vo_val_inner = extract_variable(datasets["vo"], "VO", lat, lon, 925, hour, d_deg=1.5)
                    vo_val_outer = extract_variable(datasets["vo"], "VO", lat, lon, 925, hour, d_deg=5)
                    year_df.at[idx, "vo_925_inner"] = vo_val_inner
                    year_df.at[idx, "vo_925_outer"] = vo_val_outer
                except Exception as e:
                    print(f"[WARNING] Error extracting vo for idx {idx}: {e}")
            
            if "d" in datasets:
                try:
                    d_val_inner = extract_variable(datasets["d"], "D", lat, lon, 975, hour, d_deg=1.5)
                    d_val_outer = extract_variable(datasets["d"], "D", lat, lon, 975, hour, d_deg=5)
                    year_df.at[idx, "conv_975_inner"] = -1.0 * d_val_inner  # convergence = -divergence
                    year_df.at[idx, "conv_975_outer"] = -1.0 * d_val_outer  # convergence = -divergence
                except Exception as e:
                    print(f"[WARNING] Error extracting d for idx {idx}: {e}")
        
        for ds in datasets.values():
            ds.close()
    
    year_df = year_df.drop(columns=["date_str", "hour"])
    
    return year_df


def save_year_to_netcdf(year_df, year, track_type, output_dir):
    """
    Save year data to NetCDF file
    """
    # Convert time to datetime
    year_df["time_dt"] = pd.to_datetime(year_df["time"].astype(str), format="%Y%m%d%H")
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "id": (["track_point"], year_df["id"].values),
            "time": (["track_point"], year_df["time"].values),
            "lat": (["track_point"], year_df["lat"].values),
            "lon": (["track_point"], year_df["lon"].values),
            "year": (["track_point"], year_df["year"].values),
            "month": (["track_point"], year_df["month"].values),
            "relative_vorticity": (["track_point"], year_df["relative_vorticity"].values),
            "q_700_inner": (["track_point"], year_df["q_700_inner"].values),
            "vo_925_inner": (["track_point"], year_df["vo_925_inner"].values),
            "conv_975_inner": (["track_point"], year_df["conv_975_inner"].values),
            "q_700_outer": (["track_point"], year_df["q_700_outer"].values),
            "vo_925_outer": (["track_point"], year_df["vo_925_outer"].values),
            "conv_975_outer": (["track_point"], year_df["conv_975_outer"].values),
            "tcwv": (["track_point"], year_df["tcwv"].values),
        },
        coords={
            "track_point": np.arange(len(year_df)),
            "time_dt": (["track_point"], year_df["time_dt"].values),
        },
        attrs={
            "description": f"{track_type.capitalize()} tracks with ERA5 environmental variables",
            "year": year,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "q_700_units": "kg/kg (specific humidity at 700 hPa)",
            "vo_925_units": "s-1 (vorticity at 925 hPa)",
            "conv_975_units": "s-1 (convergence at 975 hPa, computed as -1*divergence)",
            "relative_vorticity_units": "CVU (1.0 CVU = 1.0e-5 s-1)",
            "inner": "average over 1.5 degree box",
            "outer": "average over 5 degree box"
        }
    )
    
    # Save as NetCDF
    output_file = os.path.join(output_dir, f"{track_type}_{year}_ERA5.nc")
    ds.to_netcdf(output_file)
    print(f"Saved to {output_file}")
    
    ds.close()


def process_tracks_by_year(track_file, track_type, year_to_process=None):
    """
    Process tracks year by year and save individual files
    
    Parameters:
    - track_file: path to developer or non-developer Excel file
    - track_type: "developer" or "non-developer"
    - year_to_process: specific year to process (int), or None to process all years
    """
    print(f"Processing {track_type} tracks from {track_file}")
    
    # Load track data
    df = pd.read_excel(track_file)
    print(f"Loaded {len(df)} total track points, {df['id'].nunique()} unique tracks")
    
    # Get unique years
    all_years = sorted(df["year"].unique())
    
    # Determine which years to process
    if year_to_process is not None:
        if year_to_process not in all_years:
            print(f"[ERROR] Year {year_to_process} not found in data. Available years: {all_years[0]}-{all_years[-1]}")
            return
        years = [year_to_process]
        print(f"Processing single year: {year_to_process}")
    else:
        years = all_years
        print(f"Years to process: {years[0]}-{years[-1]} ({len(years)} years)")
    
    # Process each year
    for year_idx, year in enumerate(years, 1):
        print(f"\n===== Processing Year {year} ({year_idx}/{len(years)}) =====")
        
        # Filter data for this year
        year_df = df[df["year"] == year].copy().reset_index(drop=True)
        
        if len(year_df) == 0:
            print(f"[WARNING] No data for year {year}, skipping...")
            continue
        
        # Process this year
        year_df_processed = process_year(year_df, year, track_type)
        
        # Save to NetCDF
        save_year_to_netcdf(year_df_processed, year, track_type, output_dir)
    
    print(f"\nCompleted processing all {len(years)} years for {track_type}s")


# print("PROCESSING DEVELOPER TRACKS")
# process_tracks_by_year(developer_file, "developers_inner_outer_hourly", year_to_process=1980)

# print("PROCESSING NON-DEVELOPER TRACKS")
# process_tracks_by_year(nondeveloper_file, "non-developers_inner_outer_hourly", year_to_process=1980)


import sys

year = int(sys.argv[1])

# print(f"PROCESSING DEVELOPER TRACKS FOR {year}")
# process_tracks_by_year(
#     developer_file,
#     "developers_inner_outer_hourly",
#     year_to_process=year
# )

print(f"PROCESSING NON-DEVELOPER TRACKS FOR {year}")
process_tracks_by_year(
    nondeveloper_file,
    "non-developers_inner_outer_6hourly_16points_tcwv",
    year_to_process=year
)
