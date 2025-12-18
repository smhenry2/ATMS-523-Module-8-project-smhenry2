"""
Extract ERA5 environmental variables for TC tracks:
- 975-hPa convergence (-1 * divergence)
- 925-hPa vorticity
- 700-hPa specific humidity
For:
- inner region (1.5 deg box)
- outer region (5 deg box)

Processes 4-month periods: Jan-Apr, May-Aug, Sep-Dec
"""
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import os
import sys

ERA5_dir = "/gdex/data/d633000/e5.oper.an.pl"
developer_file = "./processed_tracks/developer_refined_hourly.csv"
nondeveloper_file = "./processed_tracks/non_developers_hourly_16points.csv"
output_dir = "./ERA5_extracted_hourly"

os.makedirs(output_dir, exist_ok=True)

var_code_dict = {
    "q": "128_133_q",
    "vo": "128_138_vo",
    "d": "128_155_d"
}

# Define 4-month periods
PERIODS = {
    1: ([1, 2, 3, 4], "Jan-Apr"),
    2: ([5, 6, 7, 8], "May-Aug"),
    3: ([9, 10, 11, 12], "Sep-Dec")
}


def find_era5_file(date_str, var_code):
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    year_month_dir = f"{year}{month}"
    var_full = var_code_dict[var_code]
    pattern = f"{ERA5_dir}/{year_month_dir}/e5.oper.an.pl.{var_full}.ll025sc.{year}{month}{day}00_{year}{month}{day}23.nc"
    return pattern if os.path.exists(pattern) else None


def extract_variable(ds, var_name, lat, lon, pressure_level, time_idx, d_deg=0):
    if lon < 0:
        lon = lon + 360
    
    lev_dim = None
    for dim in ds.dims:
        if "lev" in dim.lower() or "level" in dim.lower() or "pressure" in dim.lower():
            lev_dim = dim
            break
    
    if lev_dim is None:
        return np.nan
    
    try:
        ds_lev = ds.sel({lev_dim: pressure_level}, method="nearest")
        var_data = ds_lev[var_name]
        time_dim = [d for d in var_data.dims if "time" in d.lower()][0]
        var_data = var_data.isel({time_dim: time_idx})
        
        value = var_data.sel(
            latitude=slice(lat+d_deg/2, lat-d_deg/2),
            longitude=slice(lon-d_deg/2, lon+d_deg/2)
        ).mean(dim=["latitude","longitude"]).values
        
        return float(value)
    except:
        return np.nan


def extract_tcwv(ds, lat, lon, time_idx):
    if lon < 0:
        lon = lon + 360
    
    g = 9.80665
    lev_dim = None
    for dim in ds.dims:
        if "lev" in dim.lower() or "level" in dim.lower():
            lev_dim = dim
            break
    
    if lev_dim is None:
        return np.nan
    
    try:
        q = ds["Q"]
        time_dim = [d for d in q.dims if "time" in d.lower()][0]
        q = q.isel({time_dim: time_idx})
        q = q.sel(latitude=lat, longitude=lon, method="nearest")
        
        p = q[lev_dim].values * 100.0
        sort_idx = np.argsort(p)
        p = p[sort_idx]
        qv = q.values[sort_idx]
        
        tcwv = np.trapezoid(qv, p) / g
        return float(tcwv)
    except:
        return np.nan


def process_period(df, year, period_num):
    months, period_label = PERIODS[period_num]
    
    period_df = df[(df["year"] == year) & (df["month"].isin(months))].copy().reset_index(drop=True)
    
    if len(period_df) == 0:
        print(f"No data for {year}-{period_label}")
        return
    
    print(f"\nProcessing {year}-{period_label}: {len(period_df)} points, {period_df['id'].nunique()} tracks")
    
    period_df["q_700_inner"] = np.nan
    period_df["vo_925_inner"] = np.nan
    period_df["conv_975_inner"] = np.nan
    period_df["q_700_outer"] = np.nan
    period_df["vo_925_outer"] = np.nan
    period_df["conv_975_outer"] = np.nan
    period_df["tcwv"] = np.nan
    
    period_df["date_str"] = period_df["time"].astype(str).str[:8]
    period_df["hour"] = period_df["time"].astype(str).str[8:10].astype(int)
    
    unique_dates = period_df["date_str"].unique()
    
    for date_idx, date in enumerate(unique_dates, 1):
        if date_idx % 10 == 0:
            print(f"  Date {date_idx}/{len(unique_dates)}")
        
        date_mask = period_df["date_str"] == date
        
        datasets = {}
        for var in ["q", "vo", "d"]:
            file_path = find_era5_file(date + "00", var)
            if file_path:
                try:
                    datasets[var] = xr.open_dataset(file_path)
                except:
                    pass
        
        for idx in period_df[date_mask].index:
            lat = period_df.at[idx, "lat"]
            lon = period_df.at[idx, "lon"]
            hour = period_df.at[idx, "hour"]
            
            if "q" in datasets:
                period_df.at[idx, "q_700_inner"] = extract_variable(datasets["q"], "Q", lat, lon, 700, hour, d_deg=1.5)
                period_df.at[idx, "q_700_outer"] = extract_variable(datasets["q"], "Q", lat, lon, 700, hour, d_deg=5)
                period_df.at[idx, "tcwv"] = extract_tcwv(datasets["q"], lat, lon, hour)
            
            if "vo" in datasets:
                period_df.at[idx, "vo_925_inner"] = extract_variable(datasets["vo"], "VO", lat, lon, 925, hour, d_deg=1.5)
                period_df.at[idx, "vo_925_outer"] = extract_variable(datasets["vo"], "VO", lat, lon, 925, hour, d_deg=5)
            
            if "d" in datasets:
                d_inner = extract_variable(datasets["d"], "D", lat, lon, 975, hour, d_deg=1.5)
                d_outer = extract_variable(datasets["d"], "D", lat, lon, 975, hour, d_deg=5)
                period_df.at[idx, "conv_975_inner"] = -1.0 * d_inner
                period_df.at[idx, "conv_975_outer"] = -1.0 * d_outer
        
        for ds in datasets.values():
            ds.close()
    
    period_df = period_df.drop(columns=["date_str", "hour"])
    period_df["time_dt"] = pd.to_datetime(period_df["time"].astype(str), format="%Y%m%d%H")
    
    ds = xr.Dataset(
        {
            "id": (["track_point"], period_df["id"].values),
            "time": (["track_point"], period_df["time"].values),
            "lat": (["track_point"], period_df["lat"].values),
            "lon": (["track_point"], period_df["lon"].values),
            "year": (["track_point"], period_df["year"].values),
            "month": (["track_point"], period_df["month"].values),
            "relative_vorticity": (["track_point"], period_df["relative_vorticity"].values),
            "q_700_inner": (["track_point"], period_df["q_700_inner"].values),
            "vo_925_inner": (["track_point"], period_df["vo_925_inner"].values),
            "conv_975_inner": (["track_point"], period_df["conv_975_inner"].values),
            "q_700_outer": (["track_point"], period_df["q_700_outer"].values),
            "vo_925_outer": (["track_point"], period_df["vo_925_outer"].values),
            "conv_975_outer": (["track_point"], period_df["conv_975_outer"].values),
            "tcwv": (["track_point"], period_df["tcwv"].values),
        },
        coords={
            "track_point": np.arange(len(period_df)),
            "time_dt": (["track_point"], period_df["time_dt"].values),
        }
    )
    
    output_file = f"{output_dir}/non-developers_inner_outer_hourly_16points_{year}_{period_label}_ERA5.nc"
    ds.to_netcdf(output_file)
    print(f"Saved to {output_file}")
    ds.close()


if __name__ == "__main__":
    year = int(sys.argv[1])
    period = int(sys.argv[2])  # 1, 2, or 3
    
    df = pd.read_csv(nondeveloper_file)
    process_period(df, year, period)