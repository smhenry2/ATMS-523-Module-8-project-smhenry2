"""
Thank you to Miaorui Yu from Nanjing University for sharing the track processing code. It has been adapted for my needs - Sarah

Miaorui:
In Dr. Feng's TC seed dataset, TCG is defined as the first record in Ibtracs
This script redefines the developer dataset by identifying TCG as the first time a TC reaches ≥ 35 kt in the IBTrACS record.
-----------------------------------------------------------------------------------------------------------------------------------------
 Step 1: Read TCG data from Dr. Feng's seed dataset (ERA5_match_yes_trunc.dat)
 Step 2: Match TCG data with IBTrACS to redefine TCG as the first time  a TC reaches ≥35 kt
 Step 3: Read Developer data from Dr. Feng's seed dataset (ERA5_match_yes.dat) and and remove all developers whose IDs do not appear in the newly redefined TCG dataset.
"""

#%% Step 1 Read TCG data

import numpy as np
import pandas as pd
import os
from shapely.geometry import Point, Polygon

# Basins
BASIN_NAMES = ['WNP', 'ENP', 'NA']
BASIN_POLYGONS = [
    # Polygon([(105, 5), (200, 5), (200, 30), (105, 30)]),                       # WNP
    # Polygon([(200, 5), (295, 5), (260, 20), (260, 30), (200, 30)]),            # ENP
    Polygon([(295, 5), (360, 5), (360, 30), (260, 30), (260, 20)])             # NA
]
# settings
YEAR_START = 1980
TCG_DATA_DIR = './MATCH-NH-Xiangbo'


# Initialize dictionary to store seed DataFrames by basin
seed_df_dict = {
    name: pd.DataFrame(columns=['id', 'time', 'lat', 'lon', 'year', 'month', 'relative_vorticity'])
    for name in BASIN_NAMES
}


def process_track_data(file_path, hemisphere_label, year_offset):
    #Read tc track file 
    with open(file_path, 'r') as f:
        # Skip headers
        f.readline()
        f.readline()

        track_count = int(f.readline().split()[1])
        year = YEAR_START + year_offset
        print(f"[INFO] Processing {hemisphere_label} data for year {year}: {track_count} tracks")

        for _ in range(track_count):

            track_id = int(f.readline().split()[1])
            point_count = int(f.readline().split()[1])

            # Read track points
            track_data = np.array(
                [f.readline().split() for _ in range(point_count)],
                dtype=object
            ).T

            for i in range(point_count):
                t_val = int(track_data[0, i])
                lon = float(track_data[1, i])
                lat = float(track_data[2, i])
                rv = float(track_data[3, i])

                t_str = str(t_val)
                year = int(t_str[:4])
                month = int(t_str[4:6])

                # Point location
                p = Point(lon, lat)

                # Check which basin polygon contains the point
                for basin_idx, polygon in enumerate(BASIN_POLYGONS):
                    if p.within(polygon):
                        seed_df_dict[BASIN_NAMES[basin_idx]] = pd.concat([
                            seed_df_dict[BASIN_NAMES[basin_idx]],
                            pd.DataFrame([{
                                'id': int(str(track_id) + str(year)),
                                'time': t_val,
                                'lat': lat,
                                'lon': lon,
                                'year': year,
                                'month': month,
                                'relative_vorticity': rv
                            }])
                        ], ignore_index=True)
                        break


# Run Step 1
for year_offset in range(43):  # 1980–2022
    year = YEAR_START + year_offset
    file_path = os.path.join(TCG_DATA_DIR, f'ERA5_{year}_match_yes_trunc.dat')
    process_track_data(file_path, "NH", year_offset)

# Save original TCG dataset
DEV_OUTPUT_FILE = './Output-Miaorui-NH/TCG_origin.csv'
# with pd.ExcelWriter(DEV_OUTPUT_FILE) as writer:
#     for basin_name, df in seed_df_dict.items():
#         df.to_csv(writer, sheet_name=basin_name, index=False)
df = next(iter(seed_df_dict.values()))  # get the only basin DataFrame
df.to_csv(DEV_OUTPUT_FILE, index=False)

print(f"[INFO] TCG seed dataset saved to {DEV_OUTPUT_FILE}")


#%% Step 2: Match TCG data with IBTrACS (with hourly interpolation)
import xarray as xr

# Settings
TCG_PATH = DEV_OUTPUT_FILE
IBTRACS_PATH = './IBTrACS.since1980.v04r01.nc' # Please change to your own path and use a nc version of IBTrACS
FINAL_OUTPUT_FILE = './Output-Miaorui-NH/TCG_refined.csv'

# Matching tolerances
TIME_TOL = pd.Timedelta('6h')      # ±6 hours
LATLON_TOL = 3.0                   # ±3 degrees
WIND_THRESHOLD = 35.0              # Define TCG as first time reaching ≥35 kt

# Load TCG seed dataset from Step 1
tcg_df = pd.read_csv(TCG_PATH)
tcg_df['time'] = pd.to_datetime(tcg_df['time'].astype(str), format='%Y%m%d%H')

# Load IBTrACS 
ds = xr.open_dataset(IBTRACS_PATH)

n_storm = ds.dims['storm']
n_time = ds.dims['date_time']

lat_arr  = ds['usa_lat'].values
lon_arr  = ds['usa_lon'].values
time_arr = ds['time'].values
wind_arr = ds['usa_wind'].values

ib_time  = pd.to_datetime(time_arr.reshape(-1))
ib_lat   = lat_arr.reshape(-1)
ib_lon   = lon_arr.reshape(-1)
ib_wind  = wind_arr.reshape(-1)
ib_storm = np.repeat(np.arange(n_storm), n_time)
ib_time_index = np.tile(np.arange(n_time), n_storm)

ib_df = pd.DataFrame({
    'time': ib_time,
    'lat': ib_lat,
    'lon': ib_lon,
    'wind': ib_wind,
    'storm': ib_storm,
    'time_index': ib_time_index
})

ib_df = ib_df.dropna(subset=['lat', 'lon']).reset_index(drop=True)
ib_df = ib_df.sort_values(['storm', 'time']).reset_index(drop=True)

print(f"[INFO] Original IBTrACS records: {len(ib_df)}")

# ========== INTERPOLATION FUNCTION ==========
def interpolate_track_hourly(storm_df):
    """
    Interpolate a single storm's track from 6-hourly to hourly data.
    Linear interpolation for lat/lon, forward-fill for wind speed.
    """
    storm_df = storm_df.sort_values('time').reset_index(drop=True)
    
    if len(storm_df) < 2:
        return storm_df
    
    # Create hourly time range
    start_time = storm_df['time'].min()
    end_time = storm_df['time'].max()
    hourly_times = pd.date_range(start=start_time, end=end_time, freq='1h')
    
    # Create new dataframe with hourly times
    interp_df = pd.DataFrame({'time': hourly_times})
    
    # Merge with original data
    interp_df = interp_df.merge(storm_df, on='time', how='left')
    
    # Linear interpolation for lat/lon
    interp_df['lat'] = interp_df['lat'].interpolate(method='linear', limit_direction='both')
    interp_df['lon'] = interp_df['lon'].interpolate(method='linear', limit_direction='both')
    
    # Forward fill for wind (using last observed value until next observation)
    interp_df['wind'] = interp_df['wind'].ffill()
    
    # Forward fill storm and time_index
    interp_df['storm'] = interp_df['storm'].ffill().bfill()
    interp_df['time_index'] = interp_df['time_index'].ffill().bfill()
    
    return interp_df

# ========== APPLY INTERPOLATION TO ALL STORMS ==========
print("[INFO] Starting hourly interpolation...")
interpolated_storms = []

for storm_id in ib_df['storm'].unique():
    storm_data = ib_df[ib_df['storm'] == storm_id]
    interp_storm = interpolate_track_hourly(storm_data)
    interpolated_storms.append(interp_storm)

# Combine all interpolated storms
ib_df_hourly = pd.concat(interpolated_storms, ignore_index=True)
ib_df_hourly = ib_df_hourly.sort_values('time').reset_index(drop=True)

print(f"[INFO] Interpolated IBTrACS records: {len(ib_df_hourly)}")
print(f"[INFO] Interpolation factor: {len(ib_df_hourly) / len(ib_df):.2f}x")

# Replace original dataframe with interpolated version
ib_df = ib_df_hourly

# Numpy arrays for fast filtering (using interpolated data)
ib_time_np = ib_df['time'].values.astype('datetime64[ns]')
ib_lat_np = ib_df['lat'].values
ib_lon_np = ib_df['lon'].values
ib_wind_np = ib_df['wind'].values
ib_storm_np = ib_df['storm'].values

#Main Matching Loop 
matched_blocks = []
unmatched_ids = []

unique_ids = tcg_df['id'].unique()
n_ids = len(unique_ids)

print(f"[INFO] Total unique IDs = {n_ids}") # ID for TC

for idx, tc_id in enumerate(unique_ids, start=1):

    tc_id_df = tcg_df[tcg_df['id'] == tc_id].sort_values('time').reset_index(drop=True)
    if tc_id_df.empty:
        continue

    matched = False

    for i, row in tc_id_df.iterrows(): 
    # For each record in the original TCG dataset, search for an IBTrACS entry that:
    #   (1) falls within the specified time window,
    #   (2) lies within the spatial tolerance, and
    #   (3) has a wind speed ≥ 35 kt.
    # Once such a match is found, append the corresponding TCG record

        t_seed = row['time']
        lat_seed = float(row['lat'])
        lon_seed = float(row['lon'])

        # Time window filter
        t_start = np.datetime64((t_seed - TIME_TOL).to_datetime64())
        t_end   = np.datetime64((t_seed + TIME_TOL).to_datetime64())

        time_mask = (ib_time_np >= t_start) & (ib_time_np <= t_end)
        if not np.any(time_mask):
            continue

        candidate_idx = np.where(time_mask)[0]
        cand_lat = ib_lat_np[candidate_idx]
        cand_lon = ib_lon_np[candidate_idx]
        cand_wind = ib_wind_np[candidate_idx]

        # Spatial difference
        lat_diff = np.abs(cand_lat - lat_seed)
        lon_raw = np.abs(cand_lon - lon_seed)
        lon_diff = np.minimum(lon_raw, 360.0 - lon_raw)

        good_mask = (
            (lat_diff <= LATLON_TOL) &
            (lon_diff <= LATLON_TOL) &
            (cand_wind >= WIND_THRESHOLD)
        )

        if not np.any(good_mask):
            continue

        good_idx = np.where(good_mask)[0]
        dist2 = lat_diff[good_idx]**2 + lon_diff[good_idx]**2
        best_local = good_idx[np.argmin(dist2)]
        best_global = candidate_idx[best_local]

        best_row = ib_df.iloc[best_global]

        # Save all points after this match
        block = tc_id_df.loc[i:, ['id', 'time', 'lat', 'lon']].copy()
        block['matched_storm'] = int(best_row['storm'])
        block['matched_ib_time'] = best_row['time']
        block['matched_time_index'] = int(best_row['time_index'])
        block['match_lat_diff'] = lat_diff[best_local]
        block['match_lon_diff'] = lon_diff[best_local]

        matched_blocks.append(block)
        matched = True
        break

    if not matched:
        unmatched_ids.append(tc_id)

    if idx % 200 == 0 or idx == n_ids:
        print(f"[INFO] Processed {idx}/{n_ids} IDs | "
              f"Matched = {len(matched_blocks)} | Unmatched = {len(unmatched_ids)}")


# Save Final Output 
if matched_blocks:
    result_df = pd.concat(matched_blocks, ignore_index=True)
    result_df['time'] = pd.to_datetime(result_df['time']).dt.strftime('%Y%m%d%H')
    result_df.to_csv(FINAL_OUTPUT_FILE, index=False)
    print(f"[INFO] Final refined dataset saved to {FINAL_OUTPUT_FILE}")
else:
    print("[INFO] No matches found.")

print(f"[INFO] Unmatched ID count = {len(unmatched_ids)}")


#%% Step 3: Refine Developer Dataset using new TCG IDs

REFINED_TCG_FILE = "./Output-Miaorui-NH/TCG_refined.csv"     # output of Step 2
CLEANED_DEVELOPER_FILE = "./Output-Miaorui-NH/developer_refined_hourly.csv"
refined_tcg_df = pd.read_csv(REFINED_TCG_FILE)
refined_tcg_ids = set(refined_tcg_df["id"].unique())
print(f"[INFO] Refined TCG IDs loaded: {len(refined_tcg_ids)} IDs")

# Initialize new developer dictionary (same as Step 1)
developer_df_dict = {
    name: pd.DataFrame(columns=['id','time','lat','lon','year','month','relative_vorticity'])
    for name in BASIN_NAMES
}

def process_track_data(file_path, hemisphere_label, year_offset):
    with open(file_path, 'r') as f:
        f.readline()
        f.readline()
        track_count = int(f.readline().split()[1])

        year = YEAR_START + year_offset
        print(f"[INFO] Reading {hemisphere_label} developer file for {year}: {track_count} tracks")

        for _ in range(track_count):

            track_id = int(f.readline().split()[1])
            point_count = int(f.readline().split()[1])

            track_data = np.array(
                [f.readline().split() for _ in range(point_count)],
                dtype=object
            ).T

            for i in range(point_count):
                t_val = int(track_data[0, i])
                lon = float(track_data[1, i])
                lat = float(track_data[2, i])
                rv = float(track_data[3, i])

                t_str = str(t_val)
                year = int(t_str[:4])
                month = int(t_str[4:6])

                p = Point(lon, lat)

                for b_idx, poly in enumerate(BASIN_POLYGONS):
                    if p.within(poly):
                        developer_df_dict[BASIN_NAMES[b_idx]] = pd.concat([
                            developer_df_dict[BASIN_NAMES[b_idx]],
                            pd.DataFrame([{
                                "id": int(str(track_id) + str(year)),
                                "time": t_val,
                                "lat": lat,
                                "lon": lon,
                                "year": year,
                                "month": month,
                                "relative_vorticity": rv,
                            }])
                        ], ignore_index=True)
                        break


# Read developer files 
for year_offset in range(43):  # 1980–2022
    year = YEAR_START + year_offset
    file_path = os.path.join(TCG_DATA_DIR, f'ERA5_{year}_match_yes_trunc.dat')
    process_track_data(file_path, "NH", year_offset)

full_developer_df = pd.concat(developer_df_dict.values(), ignore_index=True)
original_dev_ids = full_developer_df["id"].nunique()
print(f"[INFO] Total developer IDs before refinement: {original_dev_ids}")

# Remove developers not appearing in refined TCG 
cleaned_developer_df = full_developer_df[
    full_developer_df["id"].isin(refined_tcg_ids)
].reset_index(drop=True)

remaining_dev_ids_before_interp = cleaned_developer_df["id"].nunique()
removed_dev_ids = original_dev_ids - remaining_dev_ids_before_interp

# ========== INTERPOLATE CLEANED DEVELOPER TRACKS TO HOURLY ==========
print("[INFO] Interpolating cleaned developer tracks to hourly resolution...")

def interpolate_developer_track_hourly(track_df):
    """
    Interpolate a single developer track from 6-hourly to hourly data.
    Linear interpolation for lat/lon, forward-fill for relative_vorticity.
    """
    track_df = track_df.sort_values('time').reset_index(drop=True)
    
    if len(track_df) < 2:
        return track_df
    
    # Convert time to datetime
    track_df['time_dt'] = pd.to_datetime(track_df['time'].astype(str), format='%Y%m%d%H')
    
    # Create hourly time range
    start_time = track_df['time_dt'].min()
    end_time = track_df['time_dt'].max()
    hourly_times = pd.date_range(start=start_time, end=end_time, freq='1h')
    
    # Create new dataframe with hourly times
    interp_df = pd.DataFrame({'time_dt': hourly_times})
    
    # Merge with original data
    interp_df = interp_df.merge(track_df, on='time_dt', how='left')
    
    # Linear interpolation for lat/lon
    interp_df['lat'] = interp_df['lat'].interpolate(method='linear', limit_direction='both')
    interp_df['lon'] = interp_df['lon'].interpolate(method='linear', limit_direction='both')
    
    # Forward fill for relative_vorticity (and other fields)
    interp_df['relative_vorticity'] = interp_df['relative_vorticity'].ffill()
    interp_df['id'] = interp_df['id'].ffill().bfill()
    interp_df['year'] = interp_df['year'].ffill().bfill()
    interp_df['month'] = interp_df['month'].ffill().bfill()
    
    # Convert time_dt back to YYYYMMDDHH format
    interp_df['time'] = interp_df['time_dt'].dt.strftime('%Y%m%d%H').astype(int)
    interp_df = interp_df.drop(columns=['time_dt'])
    
    return interp_df

# Apply interpolation to each unique track ID in cleaned_developer_df
interpolated_cleaned_developers = []
original_cleaned_dev_points = len(cleaned_developer_df)

for track_id in cleaned_developer_df['id'].unique():
    track_data = cleaned_developer_df[cleaned_developer_df['id'] == track_id]
    interp_track = interpolate_developer_track_hourly(track_data)
    interpolated_cleaned_developers.append(interp_track)

cleaned_developer_df = pd.concat(interpolated_cleaned_developers, ignore_index=True)
interpolated_cleaned_dev_points = len(cleaned_developer_df)

print(f"[INFO] Cleaned developer interpolation complete:")
print(f"       Original points: {original_cleaned_dev_points}")
print(f"       Interpolated points: {interpolated_cleaned_dev_points}")
print(f"       Interpolation factor: {interpolated_cleaned_dev_points / original_cleaned_dev_points:.2f}x")

remaining_dev_ids = cleaned_developer_df["id"].nunique()

# Save 
cleaned_developer_df.to_csv(CLEANED_DEVELOPER_FILE, index=False)
print(f"[INFO] Refined developer dataset saved to {CLEANED_DEVELOPER_FILE}")


#%% Step 4: Extract Non-Developers from Raw Vortex Tracks

print("\n[INFO] ===== Step 4: Processing Non-Developers (Full Tracks) =====")

NON_DEVELOPER_FILE = "./Output-Miaorui-NH/non_developers_hourly_16points.csv"
file_nam_n = '_tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.new'

# CONFIGURATION: Number of track points to extract
# Set to None to extract all points, or an integer to limit (e.g., 1 for seeds only, 5 for first 5 points)
MAX_TRACK_POINTS = 16  # Change this to limit track points (e.g., 1, 5, 10, or None for all)

print(f"[INFO] Track points configuration: {'All points' if MAX_TRACK_POINTS is None else f'First {MAX_TRACK_POINTS} points'}")

# Initialize dictionary for all seeds from raw vortex tracks
all_seeds_dict = {
    name: pd.DataFrame(columns=['track_id','id','time','lat','lon','year','month','relative_vorticity'])
    for name in BASIN_NAMES
}

# Initialize dictionary for FULL tracks (all points)
all_tracks_dict = {
    name: pd.DataFrame(columns=['track_id','id','time','lat','lon','year','month','relative_vorticity'])
    for name in BASIN_NAMES
}

def process_raw_vortex_tracks_full(file_path, year):
    """Process raw vortex tracks and extract ALL points from each track"""
    with open(file_path, 'r') as f:
        # Skip headers
        f.readline()
        f.readline()
        
        track_count = int(f.readline().split()[1])
        print(f"[INFO] Processing raw vortex tracks for {year}: {track_count} tracks")
        
        # Collect all tracks for this file
        seed_records = []
        track_records = []
        
        for _ in range(track_count):
            track_id = int(f.readline().split()[1])
            point_count = int(f.readline().split()[1])
            
            # Read all track points
            track_data = np.array(
                [f.readline().split() for _ in range(point_count)],
                dtype=object
            ).T
            
            # Process FIRST point (seed) for identification
            t_val_first = int(track_data[0, 0])
            lon_first = float(track_data[1, 0])
            lat_first = float(track_data[2, 0])
            rv_first = float(track_data[3, 0])
            
            t_str = str(t_val_first)
            year_val = int(t_str[:4])
            month_first = int(t_str[4:6])
            
            # Check if seed is within NH latitude bounds (0-40N)
            if not (0 < lat_first < 40):
                continue
            
            # Check which basin contains the SEED
            p_first = Point(lon_first, lat_first)
            seed_basin_idx = None
            for b_idx, poly in enumerate(BASIN_POLYGONS):
                if p_first.within(poly):
                    seed_basin_idx = b_idx
                    break
            
            if seed_basin_idx is None:
                continue  # Seed not in any basin
            
            # Store SEED information
            track_composite_id = int(str(track_id) + str(year_val))
            seed_records.append({
                "basin": BASIN_NAMES[seed_basin_idx],
                "track_id": track_id,
                "id": track_composite_id,
                "time": t_val_first,
                "lat": lat_first,
                "lon": lon_first,
                "year": year_val,
                "month": month_first,
                "relative_vorticity": rv_first,
            })
            
            # Store ALL POINTS from this track (up to MAX_TRACK_POINTS)
            num_points_to_process = point_count if MAX_TRACK_POINTS is None else min(MAX_TRACK_POINTS, point_count)
            
            for i in range(num_points_to_process):
                t_val = int(track_data[0, i])
                lon = float(track_data[1, i])
                lat = float(track_data[2, i])
                rv = float(track_data[3, i])
                
                t_str = str(t_val)
                year_pt = int(t_str[:4])
                month_pt = int(t_str[4:6])
                
                track_records.append({
                    "basin": BASIN_NAMES[seed_basin_idx],
                    "track_id": track_id,
                    "id": track_composite_id,
                    "time": t_val,
                    "lat": lat,
                    "lon": lon,
                    "year": year_pt,
                    "month": month_pt,
                    "relative_vorticity": rv,
                })
        
        # Batch append to DataFrames
        if seed_records:
            seed_df = pd.DataFrame(seed_records)
            for basin in BASIN_NAMES:
                basin_seeds = seed_df[seed_df['basin'] == basin].drop(columns=['basin'])
                if len(basin_seeds) > 0:
                    all_seeds_dict[basin] = pd.concat([all_seeds_dict[basin], basin_seeds], ignore_index=True)
        
        if track_records:
            track_df = pd.DataFrame(track_records)
            for basin in BASIN_NAMES:
                basin_tracks = track_df[track_df['basin'] == basin].drop(columns=['basin'])
                if len(basin_tracks) > 0:
                    all_tracks_dict[basin] = pd.concat([all_tracks_dict[basin], basin_tracks], ignore_index=True)

# Process all raw vortex track files
for year_offset in range(43):  # 1980–2022
    year = YEAR_START + year_offset
    file_path = os.path.join("seeds_Track/NH", f'{year}_NH{file_nam_n}')
    
    if os.path.exists(file_path):
        process_raw_vortex_tracks_full(file_path, year)
    else:
        print(f"[WARNING] File not found: {file_path}")

# Combine all seeds
all_seeds_df = pd.concat(all_seeds_dict.values(), ignore_index=True)
all_seeds_df = all_seeds_df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

# Combine all track points
all_tracks_df = pd.concat(all_tracks_dict.values(), ignore_index=True)

total_seeds = all_seeds_df["id"].nunique()
total_track_points = len(all_tracks_df)
print(f"[INFO] Total unique seeds extracted: {total_seeds}")
print(f"[INFO] Total track points extracted: {total_track_points}")

# ========== INTERPOLATE NON-DEVELOPER TRACKS TO HOURLY ==========
print("[INFO] Interpolating non-developer tracks to hourly resolution...")

def interpolate_vortex_track_hourly_with_trackid(track_df):
    """
    Interpolate a single vortex track from 6-hourly to hourly data.
    Linear interpolation for lat/lon, forward-fill for relative_vorticity.
    Preserves track_id field.
    """
    track_df = track_df.sort_values('time').reset_index(drop=True)
    
    if len(track_df) < 2:
        return track_df
    
    # Convert time to datetime
    track_df['time_dt'] = pd.to_datetime(track_df['time'].astype(str), format='%Y%m%d%H')
    
    # Create hourly time range
    start_time = track_df['time_dt'].min()
    end_time = track_df['time_dt'].max()
    hourly_times = pd.date_range(start=start_time, end=end_time, freq='1h')
    
    # Create new dataframe with hourly times
    interp_df = pd.DataFrame({'time_dt': hourly_times})
    
    # Merge with original data
    interp_df = interp_df.merge(track_df, on='time_dt', how='left')
    
    # Linear interpolation for lat/lon
    interp_df['lat'] = interp_df['lat'].interpolate(method='linear', limit_direction='both')
    interp_df['lon'] = interp_df['lon'].interpolate(method='linear', limit_direction='both')
    
    # Forward fill for relative_vorticity (and other fields)
    interp_df['relative_vorticity'] = interp_df['relative_vorticity'].ffill().infer_objects(copy=False)
    interp_df['track_id'] = interp_df['track_id'].ffill().bfill().infer_objects(copy=False)
    interp_df['id'] = interp_df['id'].ffill().bfill().infer_objects(copy=False)
    interp_df['year'] = interp_df['year'].ffill().bfill().infer_objects(copy=False)
    interp_df['month'] = interp_df['month'].ffill().bfill().infer_objects(copy=False)
    
    # Convert time_dt back to YYYYMMDDHH format
    interp_df['time'] = interp_df['time_dt'].dt.strftime('%Y%m%d%H').astype(int)
    interp_df = interp_df.drop(columns=['time_dt'])
    
    return interp_df

# Apply interpolation to each unique track ID in all_tracks_df
interpolated_tracks = []
original_track_points = len(all_tracks_df)

for track_id in all_tracks_df['id'].unique():
    track_data = all_tracks_df[all_tracks_df['id'] == track_id]
    interp_track = interpolate_vortex_track_hourly_with_trackid(track_data)
    interpolated_tracks.append(interp_track)

all_tracks_df = pd.concat(interpolated_tracks, ignore_index=True)
interpolated_track_points = len(all_tracks_df)

print(f"[INFO] Non-developer track interpolation complete:")
print(f"       Original points: {original_track_points}")
print(f"       Interpolated points: {interpolated_track_points}")
print(f"       Interpolation factor: {interpolated_track_points / original_track_points:.2f}x")

total_track_points = len(all_tracks_df)
print(f"[INFO] Total track points after interpolation: {total_track_points}")

# Get developer IDs from refined dataset
developer_ids = set(cleaned_developer_df["id"].unique())
print(f"[INFO] Total developer IDs: {len(developer_ids)}")

# Extract non-developers: seeds that are NOT in the developer list
non_developer_seeds_df = all_seeds_df[
    ~all_seeds_df["id"].isin(developer_ids)
].reset_index(drop=True)

# Extract ALL POINTS for non-developer tracks
non_developers_full_df = all_tracks_df[
    ~all_tracks_df["id"].isin(developer_ids)
].reset_index(drop=True)

non_dev_count = non_developer_seeds_df["id"].nunique()
non_dev_points = len(non_developers_full_df)

# Check for developers not found in seeds
missing_developers = developer_ids - set(all_seeds_df["id"].unique())
print(f"[INFO] Developers not found in raw seeds: {len(missing_developers)}")

# Summary
print("\n[INFO] ===== Non-Developer Extraction Summary =====")
print(f"- Track points config              : {'All points' if MAX_TRACK_POINTS is None else f'First {MAX_TRACK_POINTS} points per track'}")
print(f"- Total unique seeds (raw)         : {total_seeds}")
print(f"- Total track points (raw)         : {total_track_points}")
print(f"- Developers (refined)             : {len(developer_ids)}")
print(f"- Developers found in seeds        : {len(developer_ids & set(all_seeds_df['id'].unique()))}")
print(f"- Non-developer tracks             : {non_dev_count}")
print(f"- Non-developer total points       : {non_dev_points}")
print(f"- Avg points per non-dev track     : {non_dev_points/non_dev_count:.1f}")
print(f"- Verification (should be 0)       : {total_seeds - len(developer_ids & set(all_seeds_df['id'].unique())) - non_dev_count}")

# Save non-developers with FULL TRACKS
non_developers_full_df.to_csv(NON_DEVELOPER_FILE, index=False)
print(f"[INFO] Non-developer FULL TRACKS saved to {NON_DEVELOPER_FILE}")

# Summary 
print("[INFO] Developer Refinement Summary")
print(f"- Original developer IDs : {original_dev_ids}")
print(f"- Developer removed      : {removed_dev_ids}")
print(f"- Developer remaining    : {remaining_dev_ids}")
print(f"- Non-developers         : {NON_DEVELOPER_FILE}")
