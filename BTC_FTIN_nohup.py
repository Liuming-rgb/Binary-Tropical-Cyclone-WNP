# -*- coding: utf-8 -*-
# Exported from BTC_FTIN_review_5fold_20seed_no_zero.ipynb.
# Code cells retain their original order and contents.

# %% [markdown] [Notebook cell 1]
# # BTC-FTIN reviewer experiments: common cohort, 5 folds and 20 seeds
# 
# This notebook implements the reviewer-facing experiment set on the newly expanded environmental match.
# 
# Design fixed before training:
# 
# - Data source: `Data_BTC.csv`.
# - Analysis period: event years 1948–2019.
# - Missing-data rule: at each candidate window, all candidate predictors must be finite at the four input times and both intensities must be finite at the twelve target times. Future environmental/path values are not inputs and are therefore not used to reject a window. No zero, mean or temporal imputation is used; real measured zeros remain real zeros.
# - One common window-level complete-case cohort is built once and used by every neural experiment, including CTL.
# - Each sample uses four consecutive 6-hour input times and predicts the following twelve 6-hour intensities (6–72 h).
# - Member 1/2 follows the original SID order. There is no lifetime strong/weak classification or future-intensity reordering.
# - Training uses member-swap augmentation and symmetric inference, so exchanging the two input members exchanges the two outputs.
# - Five event-level folds are formed from connected components of shared SIDs. In round k, fold k is test, fold k+1 is validation, and the other three folds are training.
# - Each neural model runs the same 20 initialization seeds in each fold.
# - CTL contains only the two intensity histories. It has no zero padding and no dummy columns.
# 
# Neural experiments:
# 
# 1. CTL
# 2. CTL + SST
# 3. CTL + tropopause temperature
# 4. CTL + vertical wind shear
# 5. CTL + stability
# 6. CTL + mixed-layer depth
# 7. CTL + curvature
# 8. CTL + latitude/longitude
# 9. CTL + eastward translation speed u
# 10. CTL + northward translation speed v
# 11. CTL + pair separation distance
# The deterministic baselines are persistence and training-fold climatology. The full run contains 11 × 5 × 20 = 1,100 neural fits. Predictions are saved per run and training can resume safely.
# 

# %% [Notebook cell 2]
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import random
import re
import time
import warnings

import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings('ignore', category=FutureWarning)

ANALYSIS_VERSION = 'review_5fold_20seed_no_zero_v3_climatology'
try:
    TENSORFLOW_PACKAGE_VERSION = importlib.metadata.version('tensorflow')
except importlib.metadata.PackageNotFoundError:
    TENSORFLOW_PACKAGE_VERSION = 'not-installed'


# ------------------------------------------------------------------
# Paths and fixed analysis settings
# ------------------------------------------------------------------
ROOT = Path(r'./')

DATA_PATH = ROOT / 'Data_BTC.csv'

OUTPUT_ROOT = ROOT / 'BTC_out'

YEAR_MIN = 1948
YEAR_MAX = 2019
ONLY_QUALIFYING_SEGMENT = False

LOOKBACK = 4
FORECAST_STEPS = 12
STEP_HOURS = 6
WINDOW_LENGTH = LOOKBACK + FORECAST_STEPS
EXPECTED_DELTA = pd.Timedelta(hours=STEP_HOURS)
KT_TO_MS = 0.514444

FOLD_COUNT = 5
FOLD_ASSIGNMENT_SEED = 20260925
EXPERIMENT_SEEDS = [
    11, 23, 37, 42, 59, 71, 83, 97, 109, 131,
    149, 163, 179, 193, 211, 227, 241, 257, 271, 293,
]

BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 8
TRAIN_VERBOSE = 0

LSTM_UNITS = 48
DENSE_UNITS = 64
DROPOUT_RATE = 0.30
KERNEL_L2 = 1e-4
RECURRENT_L2 = 1e-5
LEARNING_RATE = 1e-3
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
EARLY_STOPPING_MIN_DELTA = 1e-5

# Curvature needs at least two genuinely moving 6-hour segments per member.
MIN_MOVING_SEGMENT_KM = 0.1
MIN_CURVATURE_TRACK_DISTANCE_KM = 1.0

# Full run is the default when the user executes the notebook.
RUN_TRAINING = True
RESUME_EXISTING_RUNS = True
SAVE_MODELS = True
FAIL_FAST = True
SMOKE_TEST = False
ALLOW_PARTIAL_RESULTS = False

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260925
LONG_LIVED_HOURS = 168.0
OPEN_OCEAN_KM = 200.0
MIN_REVIEWER_SUBGROUP_EVENTS = 20
REQUIRE_REVIEWER_SUBGROUP = False  # Optional map diagnostic; never blocks model fitting.

if not DATA_PATH.is_file():
    raise FileNotFoundError(DATA_PATH)
if DATA_PATH.name != 'Data_BTC.csv':
    raise AssertionError('Data_BTC.csv was not selected.')
if len(EXPERIMENT_SEEDS) != 20 or len(set(EXPERIMENT_SEEDS)) != 20:
    raise AssertionError('Exactly 20 unique neural-network seeds are required.')

print('Data:', DATA_PATH)
print('Output root:', OUTPUT_ROOT)
print('Seeds:', EXPERIMENT_SEEDS)
print('Training enabled:', RUN_TRAINING, '| smoke test:', SMOKE_TEST)

# %% [markdown] [Notebook cell 3]
# ## 1. Load the new match and audit missing records
# 
# Nothing is replaced by zero. The table below audits rows on which all candidate variables are complete, but it does not prematurely delete forecast-period rows because future environmental values are not model inputs. The common modeling cohort is constructed at window level in the next section: all predictors must be complete at the four input times, while only the two target intensities must be complete at the twelve forecast times.

# %% [Notebook cell 4]
def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


ID_COLUMNS = ['Pair_ID', 'Typhoon 1', 'Typhoon 2']
TIME_COLUMNS = ['Typhoon 1 Time', 'Typhoon 2 Time']
INTENSITY_COLUMNS = ['Typhoon 1 Intensity', 'Typhoon 2 Intensity']
POSITION_COLUMNS = [
    'Typhoon 1 Latitude', 'Typhoon 1 Longitude',
    'Typhoon 2 Latitude', 'Typhoon 2 Longitude',
]
DISTANCE_COLUMN = 'Spherical Distance (km)'
ENVIRONMENT_PAIRS = {
    'sst': ('sst1', 'sst2'),
    'air': ('air1', 'air2'),
    'wind_shear': ('wind_shear1', 'wind_shear2'),
    'stability': ('stabily1', 'stabily2'),
    'mld': ('mld1', 'mld2'),
}
ENVIRONMENT_COLUMNS = [
    column for pair in ENVIRONMENT_PAIRS.values() for column in pair
]
REQUIRED_NUMERIC_COLUMNS = (
    INTENSITY_COLUMNS + POSITION_COLUMNS + [DISTANCE_COLUMN]
    + ENVIRONMENT_COLUMNS + ['Full Path Elapsed Hours']
)
REQUIRED_COLUMNS = (
    ID_COLUMNS + TIME_COLUMNS + ['Year'] + REQUIRED_NUMERIC_COLUMNS
)
if ONLY_QUALIFYING_SEGMENT:
    REQUIRED_COLUMNS = REQUIRED_COLUMNS + ['in_qualifying_segment']

data_hash = sha256_file(DATA_PATH)
raw = pd.read_csv(DATA_PATH, low_memory=False)
missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
if missing_columns:
    raise KeyError(f'Missing required columns: {missing_columns}')

for column in ID_COLUMNS:
    normalized_id = raw[column].astype(str).str.strip()
    missing_id = (
        raw[column].isna()
        | normalized_id.eq('')
        | normalized_id.str.lower().isin({'nan', 'none', 'nat'})
    )
    if missing_id.any():
        raise ValueError(
            f'{column} contains {int(missing_id.sum())} missing/blank identifiers.'
        )

raw['Typhoon 1 Time'] = pd.to_datetime(raw['Typhoon 1 Time'], errors='raise')
raw['Typhoon 2 Time'] = pd.to_datetime(raw['Typhoon 2 Time'], errors='raise')
raw['Year'] = pd.to_numeric(raw['Year'], errors='raise').astype(int)
raw['Pair_ID'] = raw['Pair_ID'].astype(str)
raw['Typhoon 1'] = raw['Typhoon 1'].astype(str)
raw['Typhoon 2'] = raw['Typhoon 2'].astype(str)
for column in REQUIRED_NUMERIC_COLUMNS:
    raw[column] = pd.to_numeric(raw[column], errors='coerce')

if not raw['Typhoon 1 Time'].eq(raw['Typhoon 2 Time']).all():
    raise AssertionError('Member timestamps are not synchronized.')
if raw.duplicated(['Pair_ID', 'Typhoon 1 Time']).any():
    raise AssertionError('Duplicate Pair_ID + time rows were found.')
if not raw.groupby('Pair_ID')[['Typhoon 1', 'Typhoon 2']].nunique().le(1).all().all():
    raise AssertionError('Member SID order changes within at least one event.')
if not raw.groupby('Pair_ID')['Full Path Elapsed Hours'].nunique(dropna=False).le(1).all():
    raise AssertionError('Full Path Elapsed Hours changes within at least one event.')

period = raw.loc[raw['Year'].between(YEAR_MIN, YEAR_MAX)].copy()
if ONLY_QUALIFYING_SEGMENT:
    qualifying = period['in_qualifying_segment'].astype(str).str.lower().eq('true')
    period = period.loc[qualifying].copy()

if period.empty:
    raise ValueError('No rows remain in the requested period.')
if period['Year'].min() < YEAR_MIN or period['Year'].max() > YEAR_MAX:
    raise AssertionError('Rows outside the requested period remain.')
if not period.groupby('Pair_ID')['Year'].nunique().eq(1).all():
    raise AssertionError('Event Year changes within at least one Pair_ID.')

numeric_values = period[REQUIRED_NUMERIC_COLUMNS].to_numpy(dtype=float)
nonfinite = ~np.isfinite(numeric_values)
complete_mask = ~nonfinite.any(axis=1)

missing_reason = []
required_array = np.asarray(REQUIRED_NUMERIC_COLUMNS, dtype=object)
for flags in nonfinite:
    missing_reason.append(','.join(required_array[flags]))

excluded_records = period.loc[~complete_mask, [
    'Pair_ID', 'Typhoon 1', 'Typhoon 2', 'Year', 'Typhoon 1 Time'
]].copy()
excluded_records['exclusion_reason'] = np.asarray(missing_reason, dtype=object)[~complete_mask]

all_variable_complete_records = (
    period.loc[complete_mask]
    .sort_values(['Pair_ID', 'Typhoon 1 Time'])
    .reset_index(drop=True)
)

# Zeros are audited, not altered. No fillna(), interpolation or dummy columns are used.
zero_audit = pd.DataFrame({
    'variable': REQUIRED_NUMERIC_COLUMNS,
    'zero_count': [int(period[column].eq(0).sum()) for column in REQUIRED_NUMERIC_COLUMNS],
    'missing_or_nonfinite_count': [
        int((~np.isfinite(period[column].to_numpy(dtype=float))).sum())
        for column in REQUIRED_NUMERIC_COLUMNS
    ],
    'n_period_rows': len(period),
})

missingness_rows = []
for year, group in period.groupby('Year'):
    for column in REQUIRED_NUMERIC_COLUMNS:
        valid = np.isfinite(group[column].to_numpy(dtype=float))
        missingness_rows.append({
            'year': int(year), 'variable': column, 'n_rows': len(group),
            'missing_count': int((~valid).sum()),
            'missing_fraction': float((~valid).mean()),
        })
missingness_by_year = pd.DataFrame(missingness_rows)

print('Source SHA256:', data_hash)
print('All matched rows/events:', len(raw), raw['Pair_ID'].nunique())
print('1948–2019 rows/events:', len(period), period['Pair_ID'].nunique())
print(
    'Rows complete for every candidate variable (audit only):',
    len(all_variable_complete_records),
    all_variable_complete_records['Pair_ID'].nunique(),
)
display(zero_audit)

# %% [markdown] [Notebook cell 5]
# ## 2. Define causal path features and the experiment matrix
# 
# Each path-derived feature uses only the four observed input positions. The issue time is the fourth input time; no target-period position is read.
# 
# - u and v are the mean eastward and northward translation velocities over the three observed 6-hour segments, in m s⁻¹.
# - Curvature is the signed total heading change divided by total traveled distance, reported in radians per 100 km.
# - Longitude is encoded as sine and cosine to avoid a discontinuity at the dateline.
# - Pair distance is the observed great-circle center-to-center separation supplied by the matched table.
# - Window-level u, v and curvature summaries are repeated over the four LSTM steps. This is a representation of one issue-time history summary, not missing-value imputation.

# %% [Notebook cell 6]
EARTH_RADIUS_KM = 6371.0


def wrapped_angle_radians(values):
    return (values + np.pi) % (2 * np.pi) - np.pi


def segment_motion(latitude, longitude):
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    if len(latitude) != LOOKBACK or len(longitude) != LOOKBACK:
        raise ValueError('Path summaries require exactly four input positions.')

    phi_1 = np.radians(latitude[:-1])
    phi_2 = np.radians(latitude[1:])
    delta_phi = phi_2 - phi_1
    delta_lambda = wrapped_angle_radians(
        np.radians(longitude[1:]) - np.radians(longitude[:-1])
    )
    hav = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
    )
    distance_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    bearing = np.arctan2(
        np.sin(delta_lambda) * np.cos(phi_2),
        np.cos(phi_1) * np.sin(phi_2)
        - np.sin(phi_1) * np.cos(phi_2) * np.cos(delta_lambda),
    )
    speed_ms = distance_km * 1000.0 / (STEP_HOURS * 3600.0)
    u_ms = speed_ms * np.sin(bearing)
    v_ms = speed_ms * np.cos(bearing)
    return u_ms, v_ms, distance_km, bearing


def path_summary(latitude, longitude):
    u_ms, v_ms, distance_km, bearing = segment_motion(latitude, longitude)
    moving = np.isfinite(distance_km) & (distance_km >= MIN_MOVING_SEGMENT_KM)
    if moving.sum() < 2:
        raise ValueError(
            'Curvature needs at least two moving input segments; no zero curvature '
            'is substituted for an undefined heading.'
        )
    moving_distance = float(np.sum(distance_km[moving]))
    if moving_distance < MIN_CURVATURE_TRACK_DISTANCE_KM:
        raise ValueError('Curvature is undefined for a nearly stationary input track.')
    # Stationary segments have no bearing. They contribute physical zero velocity
    # to u/v, but are omitted rather than encoded as zero curvature.
    valid_bearing = bearing[moving]
    heading_change = wrapped_angle_radians(np.diff(valid_bearing))
    curvature_rad_per_100km = float(
        np.sum(heading_change) / moving_distance * 100.0
    )
    total_distance = float(np.sum(distance_km))
    return {
        'u_ms': float(np.mean(u_ms)),
        'v_ms': float(np.mean(v_ms)),
        'curvature_rad_per_100km': curvature_rad_per_100km,
        'input_track_distance_km': total_distance,
    }


def repeat_summary(values):
    values = np.asarray(values, dtype=float).reshape(1, -1)
    return np.repeat(values, LOOKBACK, axis=0)


def make_window_features(input_rows):
    intensity = input_rows[INTENSITY_COLUMNS].to_numpy(float) * KT_TO_MS

    environment = {
        name: input_rows[list(columns)].to_numpy(float)
        for name, columns in ENVIRONMENT_PAIRS.items()
    }

    lat_1 = input_rows['Typhoon 1 Latitude'].to_numpy(float)
    lon_1 = input_rows['Typhoon 1 Longitude'].to_numpy(float)
    lat_2 = input_rows['Typhoon 2 Latitude'].to_numpy(float)
    lon_2 = input_rows['Typhoon 2 Longitude'].to_numpy(float)
    lon_1_rad = np.radians(lon_1)
    lon_2_rad = np.radians(lon_2)
    latlon = np.column_stack([
        lat_1, np.sin(lon_1_rad), np.cos(lon_1_rad),
        lat_2, np.sin(lon_2_rad), np.cos(lon_2_rad),
    ])
    pair_distance = input_rows[[DISTANCE_COLUMN]].to_numpy(float)

    summary_1 = path_summary(lat_1, lon_1)
    summary_2 = path_summary(lat_2, lon_2)
    u = repeat_summary([summary_1['u_ms'], summary_2['u_ms']])
    v = repeat_summary([summary_1['v_ms'], summary_2['v_ms']])
    curvature = repeat_summary([
        summary_1['curvature_rad_per_100km'],
        summary_2['curvature_rad_per_100km'],
    ])

    feature_arrays = {
        'CTL': intensity,
        'CTL_SST': np.column_stack([intensity, environment['sst']]),
        'CTL_AIR': np.column_stack([intensity, environment['air']]),
        'CTL_VWS': np.column_stack([intensity, environment['wind_shear']]),
        'CTL_STABILITY': np.column_stack([intensity, environment['stability']]),
        'CTL_MLD': np.column_stack([intensity, environment['mld']]),
        'CTL_CURVATURE': np.column_stack([intensity, curvature]),
        'CTL_LATLON': np.column_stack([intensity, latlon]),
        'CTL_U': np.column_stack([intensity, u]),
        'CTL_V': np.column_stack([intensity, v]),
        'CTL_DISTANCE': np.column_stack([intensity, pair_distance]),
    }
    path_metadata = {
        'u_m1_ms': summary_1['u_ms'], 'u_m2_ms': summary_2['u_ms'],
        'v_m1_ms': summary_1['v_ms'], 'v_m2_ms': summary_2['v_ms'],
        'curvature_m1_rad_per_100km': summary_1['curvature_rad_per_100km'],
        'curvature_m2_rad_per_100km': summary_2['curvature_rad_per_100km'],
        'input_track_distance_m1_km': summary_1['input_track_distance_km'],
        'input_track_distance_m2_km': summary_2['input_track_distance_km'],
    }
    return feature_arrays, path_metadata


FEATURE_NAMES = {
    'CTL': ['intensity_m1', 'intensity_m2'],
    'CTL_SST': ['intensity_m1', 'intensity_m2', 'sst_m1', 'sst_m2'],
    'CTL_AIR': ['intensity_m1', 'intensity_m2', 'air_m1', 'air_m2'],
    'CTL_VWS': ['intensity_m1', 'intensity_m2', 'vws_m1', 'vws_m2'],
    'CTL_STABILITY': ['intensity_m1', 'intensity_m2', 'stability_m1', 'stability_m2'],
    'CTL_MLD': ['intensity_m1', 'intensity_m2', 'mld_m1', 'mld_m2'],
    'CTL_CURVATURE': [
        'intensity_m1', 'intensity_m2', 'curvature_m1', 'curvature_m2'
    ],
    'CTL_LATLON': [
        'intensity_m1', 'intensity_m2',
        'latitude_m1', 'longitude_sin_m1', 'longitude_cos_m1',
        'latitude_m2', 'longitude_sin_m2', 'longitude_cos_m2',
    ],
    'CTL_U': ['intensity_m1', 'intensity_m2', 'u_m1', 'u_m2'],
    'CTL_V': ['intensity_m1', 'intensity_m2', 'v_m1', 'v_m2'],
    'CTL_DISTANCE': ['intensity_m1', 'intensity_m2', 'pair_distance'],
}

EXPERIMENT_LABELS = {
    'CTL': 'CTL',
    'CTL_SST': 'CTL + SST',
    'CTL_AIR': 'CTL + tropopause T',
    'CTL_VWS': 'CTL + VWS',
    'CTL_STABILITY': 'CTL + stability',
    'CTL_MLD': 'CTL + MLD',
    'CTL_CURVATURE': 'CTL + curvature',
    'CTL_LATLON': 'CTL + latitude/longitude',
    'CTL_U': 'CTL + u',
    'CTL_V': 'CTL + v',
    'CTL_DISTANCE': 'CTL + pair distance',
    'PERSISTENCE': 'Persistence',
    'CLIMATOLOGY': 'Training-fold climatology',
}
ENVIRONMENT_EXPERIMENTS = [
    'CTL_SST', 'CTL_AIR', 'CTL_VWS', 'CTL_STABILITY', 'CTL_MLD'
]
PATH_EXPERIMENTS = [
    'CTL_CURVATURE', 'CTL_LATLON', 'CTL_U', 'CTL_V',
    'CTL_DISTANCE',
]
BTC_EXPERIMENTS = list(FEATURE_NAMES)
NEURAL_EXPERIMENTS = BTC_EXPERIMENTS.copy()

if FEATURE_NAMES['CTL'] != ['intensity_m1', 'intensity_m2']:
    raise AssertionError('CTL must contain only the two intensity histories.')
if any('dummy' in name.lower() for names in FEATURE_NAMES.values() for name in names):
    raise AssertionError('Dummy or zero-padding features are prohibited.')

# %% [markdown] [Notebook cell 7]
# ## 3. Build common windows within each event
# 
# Sliding is performed separately inside each `Pair_ID`. A candidate can never use a row from another double-typhoon event. Every candidate contains exactly 16 consecutive 6-hour records. The first four records provide all predictors; the following twelve provide only the two target intensities. This keeps all experiments on the same 155-event cohort without discarding a window because an unused future environmental value is missing.

# %% [Notebook cell 8]
INPUT_REQUIRED_COLUMNS = REQUIRED_NUMERIC_COLUMNS.copy()
TARGET_REQUIRED_COLUMNS = INTENSITY_COLUMNS.copy()

feature_lists = {name: [] for name in BTC_EXPERIMENTS}
target_list = []
metadata_rows = []
rejected_window_rows = []
complete_candidate_window_count = 0
complete_candidate_events = set()

for pair_id, event_rows in period.groupby('Pair_ID', sort=False):
    event_rows = event_rows.sort_values('Typhoon 1 Time').reset_index(drop=True)
    if event_rows['Typhoon 1'].nunique() != 1 or event_rows['Typhoon 2'].nunique() != 1:
        raise AssertionError(f'Member identity changes in {pair_id}.')

    for start in range(max(0, len(event_rows) - WINDOW_LENGTH + 1)):
        block = event_rows.iloc[start:start + WINDOW_LENGTH]
        issue_time = block['Typhoon 1 Time'].iloc[LOOKBACK - 1]
        candidate_id = f'{pair_id}|{issue_time.isoformat()}'

        time_deltas = block['Typhoon 1 Time'].diff().iloc[1:]
        if len(block) != WINDOW_LENGTH or not time_deltas.eq(EXPECTED_DELTA).all():
            rejected_window_rows.append({
                'Pair_ID': pair_id,
                'candidate_issue_time': issue_time,
                'reason': 'not_16_consecutive_6h_records',
            })
            continue

        input_rows = block.iloc[:LOOKBACK]
        target_rows = block.iloc[LOOKBACK:]
        input_ok = np.isfinite(
            input_rows[INPUT_REQUIRED_COLUMNS].to_numpy(dtype=float)
        ).all()
        target_ok = np.isfinite(
            target_rows[TARGET_REQUIRED_COLUMNS].to_numpy(dtype=float)
        ).all()
        if not input_ok:
            missing = []
            for column in INPUT_REQUIRED_COLUMNS:
                values = input_rows[column].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    missing.append(column)
            rejected_window_rows.append({
                'Pair_ID': pair_id,
                'candidate_issue_time': issue_time,
                'reason': 'missing_input:' + ','.join(missing),
            })
            continue
        if not target_ok:
            missing = []
            for column in TARGET_REQUIRED_COLUMNS:
                values = target_rows[column].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    missing.append(column)
            rejected_window_rows.append({
                'Pair_ID': pair_id,
                'candidate_issue_time': issue_time,
                'reason': 'missing_target:' + ','.join(missing),
            })
            continue

        complete_candidate_window_count += 1
        complete_candidate_events.add(pair_id)

        try:
            feature_arrays, path_metadata = make_window_features(input_rows)
        except ValueError as error:
            rejected_window_rows.append({
                'Pair_ID': pair_id,
                'candidate_issue_time': issue_time,
                'reason': 'invalid_path_geometry:' + str(error),
            })
            continue

        target = target_rows[INTENSITY_COLUMNS].to_numpy(float) * KT_TO_MS
        if target.shape != (FORECAST_STEPS, 2):
            raise AssertionError('Target layout is not 12 leads × 2 members.')
        if not np.isfinite(target).all():
            raise AssertionError('A non-finite target passed the eligibility check.')

        for experiment in BTC_EXPERIMENTS:
            values = feature_arrays[experiment]
            expected_shape = (LOOKBACK, len(FEATURE_NAMES[experiment]))
            if values.shape != expected_shape:
                raise AssertionError(
                    f'{experiment}: expected {expected_shape}, found {values.shape}'
                )
            if not np.isfinite(values).all():
                raise AssertionError(f'{experiment} contains NaN or Inf.')
            feature_lists[experiment].append(values.astype(np.float32))

        target_list.append(target.astype(np.float32))
        duration = float(block['Full Path Elapsed Hours'].iloc[0])
        metadata_rows.append({
            'window_id': candidate_id,
            'Pair_ID': pair_id,
            'event_year': int(block['Year'].iloc[0]),
            'sid_m1': str(block['Typhoon 1'].iloc[0]),
            'sid_m2': str(block['Typhoon 2'].iloc[0]),
            'input_start': block['Typhoon 1 Time'].iloc[0],
            'issue_time': issue_time,
            'target_start': block['Typhoon 1 Time'].iloc[LOOKBACK],
            'target_end': block['Typhoon 1 Time'].iloc[-1],
            'issue_month': int(issue_time.month),
            'event_duration_hours': duration,
            'issue_lat_m1': float(input_rows['Typhoon 1 Latitude'].iloc[-1]),
            'issue_lon_m1': float(input_rows['Typhoon 1 Longitude'].iloc[-1]),
            'issue_lat_m2': float(input_rows['Typhoon 2 Latitude'].iloc[-1]),
            'issue_lon_m2': float(input_rows['Typhoon 2 Longitude'].iloc[-1]),
            'initial_intensity_m1_ms': float(
                input_rows['Typhoon 1 Intensity'].iloc[-1] * KT_TO_MS
            ),
            'initial_intensity_m2_ms': float(
                input_rows['Typhoon 2 Intensity'].iloc[-1] * KT_TO_MS
            ),
            'path_feature_latest_source_time': issue_time,
            **path_metadata,
        })

if not metadata_rows:
    raise ValueError('No eligible windows were constructed.')

X_BY_EXPERIMENT = {
    experiment: np.stack(values).astype(np.float32)
    for experiment, values in feature_lists.items()
}
Y = np.stack(target_list).astype(np.float32)
WINDOW_META = pd.DataFrame(metadata_rows)
REJECTED_WINDOWS = pd.DataFrame(rejected_window_rows)

if WINDOW_META['window_id'].duplicated().any():
    raise AssertionError('Window IDs are not unique.')
if Y.shape != (len(WINDOW_META), FORECAST_STEPS, 2):
    raise AssertionError(f'Unexpected target shape: {Y.shape}')
if not WINDOW_META['path_feature_latest_source_time'].le(
    WINDOW_META['issue_time']
).all():
    raise AssertionError('A path feature reads after the forecast issue time.')

for experiment, values in X_BY_EXPERIMENT.items():
    if values.shape[0] != len(WINDOW_META) or values.shape[1] != LOOKBACK:
        raise AssertionError(f'{experiment} has inconsistent common windows.')
    if values.shape[2] != len(FEATURE_NAMES[experiment]):
        raise AssertionError(f'{experiment} has an incorrect feature dimension.')
    if not np.isfinite(values).all():
        raise AssertionError(f'{experiment} contains non-finite values.')

# These values identify accidental changes to the fixed source/missing-data policy.
EXPECTED_CURRENT_ELIGIBLE_EVENTS = 155
EXPECTED_INPUT_TARGET_COMPLETE_WINDOWS = 2179
EXPECTED_CURRENT_WINDOWS = 2171
if complete_candidate_window_count != EXPECTED_INPUT_TARGET_COMPLETE_WINDOWS:
    raise AssertionError(
        f'Expected {EXPECTED_INPUT_TARGET_COMPLETE_WINDOWS} input/target-complete '
        f'candidate windows, found {complete_candidate_window_count}.'
    )
if WINDOW_META['Pair_ID'].nunique() != EXPECTED_CURRENT_ELIGIBLE_EVENTS:
    raise AssertionError(
        f'Expected {EXPECTED_CURRENT_ELIGIBLE_EVENTS} eligible events, found '
        f"{WINDOW_META['Pair_ID'].nunique()}."
    )
if len(WINDOW_META) != EXPECTED_CURRENT_WINDOWS:
    raise AssertionError(
        f'Expected {EXPECTED_CURRENT_WINDOWS} windows, found {len(WINDOW_META)}.'
    )

event_manifest = (
    period.groupby('Pair_ID')
    .agg(
        event_year=('Year', 'first'),
        sid_m1=('Typhoon 1', 'first'),
        sid_m2=('Typhoon 2', 'first'),
        period_records=('Typhoon 1 Time', 'size'),
    )
    .join(
        WINDOW_META.groupby('Pair_ID').size().rename('eligible_windows'),
        how='left',
    )
    .fillna({'eligible_windows': 0})
    .reset_index()
)
event_manifest['eligible_windows'] = event_manifest['eligible_windows'].astype(int)
event_manifest['modeling_status'] = np.where(
    event_manifest['eligible_windows'].gt(0), 'eligible', 'no_eligible_window'
)

cohort_flow = pd.DataFrame([
    {'stage': 'raw matched source', 'records': len(raw), 'events': raw['Pair_ID'].nunique(), 'windows': np.nan},
    {'stage': 'event years 1948-2019', 'records': len(period), 'events': period['Pair_ID'].nunique(), 'windows': np.nan},
    {'stage': 'all-variable complete rows (audit only)', 'records': len(all_variable_complete_records), 'events': all_variable_complete_records['Pair_ID'].nunique(), 'windows': np.nan},
    {'stage': 'input/target-complete candidate windows', 'records': np.nan, 'events': len(complete_candidate_events), 'windows': complete_candidate_window_count},
    {'stage': 'common eligible modeling windows', 'records': np.nan, 'events': WINDOW_META['Pair_ID'].nunique(), 'windows': len(WINDOW_META)},
])

window_manifest_hash = hashlib.sha256(
    '\n'.join(WINDOW_META['window_id']).encode('utf-8')
).hexdigest()
print('Eligible events:', WINDOW_META['Pair_ID'].nunique())
print('Common windows:', len(WINDOW_META))
print('Common window-manifest SHA256:', window_manifest_hash)
display(cohort_flow)

# %% [markdown] [Notebook cell 9]
# ## 4. Reviewer subgroup metadata (diagnostic only)
# 
# The reviewer specifically mentioned long-lived, open-ocean, northward-moving cases. This cell creates a retrospective event-level diagnostic subgroup; none of these labels is used as a model input. Long-lived means at least seven days of simultaneous records, northward means positive median pair-mean v over eligible issue times, and open ocean means median issue-time distance of both members from 110 m Natural Earth land is at least 200 km. If the local land shapefile is unavailable, the main analysis still runs and the subgroup panel is marked unavailable.

# %% [Notebook cell 10]
def haversine_km(lat_1, lon_1, lat_2, lon_2):
    phi_1, phi_2 = np.radians([lat_1, lat_2])
    delta_phi = phi_2 - phi_1
    delta_lambda = wrapped_angle_radians(np.radians(lon_2 - lon_1))
    value = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
    )
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0, 1))))


LAND_SHAPEFILE_CANDIDATES = [
    Path(r'C:\Users\wlm\.local\share\cartopy\shapefiles\natural_earth\physical\ne_110m_land.shp'),
]
land_shapefile = next(
    (path for path in LAND_SHAPEFILE_CANDIDATES if path.is_file()), None
)
land_shapefile_hash = (
    sha256_file(land_shapefile) if land_shapefile is not None else None
)

WINDOW_META['distance_to_land_m1_km'] = np.nan
WINDOW_META['distance_to_land_m2_km'] = np.nan
land_distance_available = False

if land_shapefile is not None:
    try:
        import cartopy.io.shapereader as shapereader
        from shapely.affinity import translate
        from shapely.geometry import Point
        from shapely.ops import nearest_points, unary_union

        geometries = list(shapereader.Reader(str(land_shapefile)).geometries())
        base_land = unary_union(geometries)
        land = unary_union([
            base_land,
            translate(base_land, xoff=-360),
            translate(base_land, xoff=360),
        ])
        distance_cache = {}

        def distance_to_land_km(latitude, longitude):
            key = (round(float(latitude), 4), round(float(longitude), 4))
            if key in distance_cache:
                return distance_cache[key]
            wrapped_lon = ((float(longitude) + 180) % 360) - 180
            point = Point(wrapped_lon, float(latitude))
            if land.covers(point):
                value = 0.0
            else:
                nearest = nearest_points(point, land)[1]
                value = haversine_km(
                    float(latitude), wrapped_lon, float(nearest.y), float(nearest.x)
                )
            distance_cache[key] = value
            return value

        WINDOW_META['distance_to_land_m1_km'] = [
            distance_to_land_km(lat, lon)
            for lat, lon in zip(WINDOW_META['issue_lat_m1'], WINDOW_META['issue_lon_m1'])
        ]
        WINDOW_META['distance_to_land_m2_km'] = [
            distance_to_land_km(lat, lon)
            for lat, lon in zip(WINDOW_META['issue_lat_m2'], WINDOW_META['issue_lon_m2'])
        ]
        land_distance_available = True
    except Exception as error:
        print('Open-ocean diagnostic unavailable:', repr(error))
else:
    print('Open-ocean diagnostic unavailable: local Natural Earth shapefile not found.')

event_subgroup = (
    WINDOW_META.groupby('Pair_ID')
    .agg(
        event_duration_hours=('event_duration_hours', 'first'),
        median_pair_v_ms=(
            'v_m1_ms',
            lambda values: float(np.nanmedian(values)),
        ),
        median_land_distance_m1_km=('distance_to_land_m1_km', 'median'),
        median_land_distance_m2_km=('distance_to_land_m2_km', 'median'),
        eligible_windows=('window_id', 'size'),
    )
    .reset_index()
)
# Replace the member-1-only temporary aggregation with the true pair-mean v.
pair_v = WINDOW_META.assign(
    pair_mean_v_ms=(WINDOW_META['v_m1_ms'] + WINDOW_META['v_m2_ms']) / 2
).groupby('Pair_ID')['pair_mean_v_ms'].median()
event_subgroup['median_pair_v_ms'] = event_subgroup['Pair_ID'].map(pair_v)
event_subgroup['long_lived'] = event_subgroup['event_duration_hours'].ge(LONG_LIVED_HOURS)
event_subgroup['northward'] = event_subgroup['median_pair_v_ms'].gt(0)
event_subgroup['open_ocean'] = (
    event_subgroup[[
        'median_land_distance_m1_km', 'median_land_distance_m2_km'
    ]].min(axis=1).ge(OPEN_OCEAN_KM)
    if land_distance_available
    else False
)
event_subgroup['reviewer_subgroup'] = (
    event_subgroup['long_lived']
    & event_subgroup['northward']
    & event_subgroup['open_ocean']
)
subgroup_map = event_subgroup.set_index('Pair_ID')['reviewer_subgroup']
WINDOW_META['reviewer_subgroup'] = WINDOW_META['Pair_ID'].map(subgroup_map).fillna(False)

print('Land-distance diagnostic available:', land_distance_available)
print('Reviewer-subgroup events:', int(event_subgroup['reviewer_subgroup'].sum()))
if REQUIRE_REVIEWER_SUBGROUP and not land_distance_available:
    raise RuntimeError(
        'Formal reviewer analysis requires Cartopy, Shapely and the local '
        'Natural Earth land shapefile before training starts.'
    )
if REQUIRE_REVIEWER_SUBGROUP and int(event_subgroup['reviewer_subgroup'].sum()) == 0:
    raise RuntimeError('The formal reviewer subgroup contains zero events.')
display(event_subgroup.head())

# %% [markdown] [Notebook cell 11]
# ## 5. Five folds grouped by shared storm identity
# 
# Two different `Pair_ID` values can contain the same typhoon SID. All Pair IDs connected by either member SID are therefore assigned to the same base fold. The greedy assignment balances event count, window count and four broad eras. Fold assignments never change with neural-network seed.

# %% [Notebook cell 12]
class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


event_table = (
    WINDOW_META.groupby('Pair_ID')
    .agg(
        event_year=('event_year', 'first'),
        sid_m1=('sid_m1', 'first'),
        sid_m2=('sid_m2', 'first'),
        n_windows=('window_id', 'size'),
    )
    .reset_index()
)

union_find = UnionFind(event_table['Pair_ID'])
sid_owner = {}
for row in event_table.itertuples(index=False):
    for sid in (row.sid_m1, row.sid_m2):
        if sid in sid_owner:
            union_find.union(row.Pair_ID, sid_owner[sid])
        else:
            sid_owner[sid] = row.Pair_ID

component_members = {}
for pair_id in event_table['Pair_ID']:
    root = union_find.find(pair_id)
    component_members.setdefault(root, []).append(pair_id)
canonical_component = {
    pair_id: min(members)
    for members in component_members.values()
    for pair_id in members
}
event_table['component_id'] = event_table['Pair_ID'].map(canonical_component)
event_table['era'] = pd.cut(
    event_table['event_year'],
    bins=[1947, 1969, 1989, 2004, 2019],
    labels=['1948-1969', '1970-1989', '1990-2004', '2005-2019'],
    include_lowest=True,
).astype(str)
ERA_LABELS = ['1948-1969', '1970-1989', '1990-2004', '2005-2019']

component_rows = []
for component_id, group in event_table.groupby('component_id'):
    row = {
        'component_id': component_id,
        'n_events': len(group),
        'n_windows': int(group['n_windows'].sum()),
    }
    for era in ERA_LABELS:
        row[f'era_{era}'] = int(group['era'].eq(era).sum())
    component_rows.append(row)
components = pd.DataFrame(component_rows)

rng = np.random.default_rng(FOLD_ASSIGNMENT_SEED)
components['tie_breaker'] = rng.random(len(components))
components = components.sort_values(
    ['n_events', 'n_windows', 'tie_breaker'],
    ascending=[False, False, True],
).reset_index(drop=True)

fold_events = np.zeros(FOLD_COUNT, dtype=float)
fold_windows = np.zeros(FOLD_COUNT, dtype=float)
fold_eras = np.zeros((FOLD_COUNT, len(ERA_LABELS)), dtype=float)
target_events = len(event_table) / FOLD_COUNT
target_windows = event_table['n_windows'].sum() / FOLD_COUNT
target_eras = np.asarray([
    event_table['era'].eq(era).sum() / FOLD_COUNT for era in ERA_LABELS
], dtype=float)

component_fold = {}
for component_index, component in components.iterrows():
    if component_index < FOLD_COUNT:
        candidates = [component_index]
    else:
        candidates = list(range(FOLD_COUNT))
        rng.shuffle(candidates)

    scored = []
    for fold in candidates:
        test_events = fold_events.copy()
        test_windows = fold_windows.copy()
        test_eras = fold_eras.copy()
        test_events[fold] += component['n_events']
        test_windows[fold] += component['n_windows']
        test_eras[fold, :] += np.asarray([
            component[f'era_{era}'] for era in ERA_LABELS
        ], dtype=float)
        score = (
            np.sum(((test_events - target_events) / max(target_events, 1)) ** 2)
            + 0.5 * np.sum(((test_windows - target_windows) / max(target_windows, 1)) ** 2)
            + 0.5 * np.sum(
                ((test_eras - target_eras) / np.maximum(target_eras, 1)) ** 2
            )
        )
        scored.append((float(score), int(fold)))
    _, chosen_fold = min(scored)
    component_fold[component['component_id']] = chosen_fold
    fold_events[chosen_fold] += component['n_events']
    fold_windows[chosen_fold] += component['n_windows']
    fold_eras[chosen_fold, :] += np.asarray([
        component[f'era_{era}'] for era in ERA_LABELS
    ], dtype=float)

event_table['base_fold'] = event_table['component_id'].map(component_fold).astype(int)
if set(event_table['base_fold']) != set(range(FOLD_COUNT)):
    raise AssertionError('Not all five folds received events.')
fold_map = event_table.set_index('Pair_ID')['base_fold']
WINDOW_META['base_fold'] = WINDOW_META['Pair_ID'].map(fold_map).astype(int)
component_map = event_table.set_index('Pair_ID')['component_id']
WINDOW_META['component_id'] = WINDOW_META['Pair_ID'].map(component_map)
era_map = event_table.set_index('Pair_ID')['era']
WINDOW_META['era'] = WINDOW_META['Pair_ID'].map(era_map)
if WINDOW_META['component_id'].isna().any():
    raise AssertionError('At least one eligible window lacks a shared-SID component.')

fold_assignment_text = '\n'.join(
    event_table.sort_values('Pair_ID').apply(
        lambda row: (
            f"{row['Pair_ID']}|{row['component_id']}|{int(row['base_fold'])}"
        ),
        axis=1,
    )
)
fold_assignment_hash = hashlib.sha256(
    fold_assignment_text.encode('utf-8')
).hexdigest()

partition_summary_rows = []
overlap_rows = []
for test_fold in range(FOLD_COUNT):
    validation_fold = (test_fold + 1) % FOLD_COUNT
    partition_pairs = {
        'train': set(event_table.loc[
            ~event_table['base_fold'].isin([test_fold, validation_fold]), 'Pair_ID'
        ]),
        'validation': set(event_table.loc[
            event_table['base_fold'].eq(validation_fold), 'Pair_ID'
        ]),
        'test': set(event_table.loc[
            event_table['base_fold'].eq(test_fold), 'Pair_ID'
        ]),
    }
    partition_sids = {}
    for name, pair_ids in partition_pairs.items():
        rows = event_table[event_table['Pair_ID'].isin(pair_ids)]
        partition_sids[name] = set(rows['sid_m1']) | set(rows['sid_m2'])
        partition_summary_rows.append({
            'test_fold': test_fold,
            'validation_fold': validation_fold,
            'partition': name,
            'n_events': len(pair_ids),
            'n_sids': len(partition_sids[name]),
            'n_windows': int(WINDOW_META['Pair_ID'].isin(pair_ids).sum()),
            'year_min': int(rows['event_year'].min()),
            'year_max': int(rows['event_year'].max()),
        })

    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        pair_overlap = partition_pairs[left] & partition_pairs[right]
        sid_overlap = partition_sids[left] & partition_sids[right]
        overlap_rows.append({
            'test_fold': test_fold,
            'left_partition': left,
            'right_partition': right,
            'pair_overlap_count': len(pair_overlap),
            'sid_overlap_count': len(sid_overlap),
        })
        if pair_overlap or sid_overlap:
            raise AssertionError(
                f'Leakage in fold {test_fold}: {left} vs {right}; '
                f'pairs={pair_overlap}, sids={sid_overlap}'
            )

partition_summary = pd.DataFrame(partition_summary_rows)
partition_overlap_audit = pd.DataFrame(overlap_rows)

# Every event is test exactly once and validation exactly once across the five rounds.
test_counts = {pair_id: 0 for pair_id in event_table['Pair_ID']}
validation_counts = {pair_id: 0 for pair_id in event_table['Pair_ID']}
for test_fold in range(FOLD_COUNT):
    validation_fold = (test_fold + 1) % FOLD_COUNT
    for pair_id in event_table.loc[event_table['base_fold'].eq(test_fold), 'Pair_ID']:
        test_counts[pair_id] += 1
    for pair_id in event_table.loc[event_table['base_fold'].eq(validation_fold), 'Pair_ID']:
        validation_counts[pair_id] += 1
if set(test_counts.values()) != {1} or set(validation_counts.values()) != {1}:
    raise AssertionError('Five-fold test/validation coverage is incomplete.')

display(partition_summary)
display(partition_overlap_audit)

# %% [markdown] [Notebook cell 13]
# ## 6. Freeze configuration and write data/fold audit files
# 
# The run directory name contains a hash of the source CSV, common window manifest and fixed settings. This prevents predictions from a different cohort or configuration from being silently mixed into a resumed run.

# %% [Notebook cell 14]
UNIT_BY_FEATURE = {
    'intensity': 'm s-1', 'sst': 'deg C', 'air': 'K', 'vws': 'm s-1',
    'stability': 'deg C', 'mld': 'm', 'curvature': 'rad per 100 km',
    'latitude': 'degree north', 'longitude_sin': 'unitless',
    'longitude_cos': 'unitless', 'u': 'm s-1', 'v': 'm s-1',
    'pair_distance': 'km',
}

feature_definition_rows = []
for experiment, names in FEATURE_NAMES.items():
    for position, feature in enumerate(names):
        base = re.sub(r'_m[12]$', '', feature)
        feature_definition_rows.append({
            'experiment': experiment,
            'experiment_label': EXPERIMENT_LABELS[experiment],
            'feature_position': position,
            'feature': feature,
            'unit': UNIT_BY_FEATURE.get(base, 'see source'),
            'time_support': (
                'four dynamic input times'
                if base in {'intensity', 'sst', 'air', 'vws', 'stability', 'mld',
                            'latitude', 'longitude_sin', 'longitude_cos', 'pair_distance'}
                else 'summary of four observed input times, repeated across input steps'
            ),
            'missing_value_policy': 'window rejected; no imputation',
        })
feature_definitions = pd.DataFrame(feature_definition_rows)

experiment_config = pd.DataFrame([
    {
        'experiment': experiment,
        'label': EXPERIMENT_LABELS[experiment],
        'kind': 'BTC neural',
        'n_features': len(FEATURE_NAMES[experiment]),
        'features': ', '.join(FEATURE_NAMES[experiment]),
        'common_window_manifest_sha256': window_manifest_hash,
        'zero_padding': False,
    }
    for experiment in BTC_EXPERIMENTS
])

config_payload = {
    'analysis_version': ANALYSIS_VERSION,
    'data_path': str(DATA_PATH.resolve()),
    'data_sha256': data_hash,
    'window_manifest_sha256': window_manifest_hash,
    'year_min': YEAR_MIN,
    'year_max': YEAR_MAX,
    'only_qualifying_segment': ONLY_QUALIFYING_SEGMENT,
    'lookback': LOOKBACK,
    'forecast_steps': FORECAST_STEPS,
    'step_hours': STEP_HOURS,
    'fold_count': FOLD_COUNT,
    'fold_assignment_seed': FOLD_ASSIGNMENT_SEED,
    'fold_assignment_sha256': fold_assignment_hash,
    'experiment_seeds': EXPERIMENT_SEEDS,
    'experiments': NEURAL_EXPERIMENTS,
    'feature_names': FEATURE_NAMES,
    'batch_size': BATCH_SIZE,
    'max_epochs': MAX_EPOCHS,
    'smoke_test': SMOKE_TEST,
    'training_mode': 'smoke' if SMOKE_TEST else 'formal',
    'effective_folds': [0] if SMOKE_TEST else list(range(FOLD_COUNT)),
    'effective_seeds': EXPERIMENT_SEEDS[:2] if SMOKE_TEST else EXPERIMENT_SEEDS,
    'requested_epochs': min(MAX_EPOCHS, 2) if SMOKE_TEST else MAX_EPOCHS,
    'tensorflow_package_version': TENSORFLOW_PACKAGE_VERSION,
    'model': {
        'lstm_units': LSTM_UNITS,
        'dense_units': DENSE_UNITS,
        'dropout_rate': DROPOUT_RATE,
        'kernel_l2': KERNEL_L2,
        'recurrent_l2': RECURRENT_L2,
        'learning_rate': LEARNING_RATE,
        'loss': 'mse',
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA,
        'lr_patience': LR_PATIENCE,
        'lr_factor': LR_FACTOR,
        'min_learning_rate': MIN_LEARNING_RATE,
    },
    'path_feature_definition': {
        'version': 'causal_four_position_v2',
        'minimum_moving_segment_km': MIN_MOVING_SEGMENT_KM,
        'minimum_curvature_track_distance_km': MIN_CURVATURE_TRACK_DISTANCE_KM,
        'curvature': (
            'signed sum of heading changes among moving segments divided by '
            'moving distance, radians per 100 km'
        ),
        'u_v': 'mean across all three 6-hour segments; true zero motion retained',
    },
    'reviewer_subgroup_definition': {
        'long_lived_hours': LONG_LIVED_HOURS,
        'open_ocean_km': OPEN_OCEAN_KM,
        'northward_rule': 'positive event-median pair-mean v',
        'land_shapefile': (
            str(land_shapefile.resolve()) if land_shapefile is not None else None
        ),
        'land_shapefile_sha256': land_shapefile_hash,
        'minimum_confirmatory_events': MIN_REVIEWER_SUBGROUP_EVENTS,
        'required': REQUIRE_REVIEWER_SUBGROUP,
    },
    'inference': {
        'primary_horizon': '6-72h',
        'sensitivity_horizon': '36-72h',
        'bootstrap_iterations': BOOTSTRAP_ITERATIONS,
        'bootstrap_seed': BOOTSTRAP_SEED,
        'cluster_unit': 'connected component of shared storm SIDs',
        'seed_unit': 'one seed joined across all five OOF test folds',
    },
    'missing_policy': (
        'all candidate predictors finite at 4 input times; both intensities finite '
        'at 12 targets; no imputation and no zero padding'
    ),
}
config_hash = hashlib.sha256(
    json.dumps(config_payload, sort_keys=True).encode('utf-8')
).hexdigest()[:12]

RUN_DIR = OUTPUT_ROOT / f'run_{config_hash}'
AUDIT_DIR = RUN_DIR / 'audit'
PREDICTION_DIR = RUN_DIR / 'predictions_by_run'
HISTORY_DIR = RUN_DIR / 'training_history_by_run'
MODEL_DIR = RUN_DIR / 'saved_models'
PREPROCESSING_DIR = RUN_DIR / 'preprocessing'
TABLE_DIR = RUN_DIR / 'tables'
for directory in [
    AUDIT_DIR, PREDICTION_DIR, HISTORY_DIR, MODEL_DIR, PREPROCESSING_DIR,
    TABLE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

with (RUN_DIR / 'experiment_configuration.json').open('w', encoding='utf-8') as handle:
    json.dump(
        {**config_payload, 'config_hash': config_hash},
        handle, ensure_ascii=False, indent=2,
    )

pd.DataFrame({'seed': EXPERIMENT_SEEDS}).to_csv(
    AUDIT_DIR / 'seed_manifest.csv', index=False
)
cohort_flow.to_csv(AUDIT_DIR / 'cohort_flow_counts.csv', index=False)
event_manifest.to_csv(AUDIT_DIR / 'event_cohort_manifest.csv', index=False)
excluded_records.to_csv(AUDIT_DIR / 'excluded_records_with_reasons.csv', index=False)
REJECTED_WINDOWS.to_csv(AUDIT_DIR / 'rejected_window_candidates.csv', index=False)
WINDOW_META.to_csv(AUDIT_DIR / 'eligible_window_manifest.csv', index=False)
zero_audit.to_csv(AUDIT_DIR / 'missingness_and_zero_audit.csv', index=False)
missingness_by_year.to_csv(AUDIT_DIR / 'missingness_by_variable_year.csv', index=False)
event_subgroup.to_csv(AUDIT_DIR / 'reviewer_subgroup_event_manifest.csv', index=False)
event_table.to_csv(AUDIT_DIR / 'fold_assignments.csv', index=False)
partition_summary.to_csv(AUDIT_DIR / 'fold_partition_summary.csv', index=False)
partition_overlap_audit.to_csv(AUDIT_DIR / 'partition_overlap_audit.csv', index=False)
feature_definitions.to_csv(AUDIT_DIR / 'feature_definitions.csv', index=False)
experiment_config.to_csv(AUDIT_DIR / 'experiment_config.csv', index=False)

expected_neural_runs = len(NEURAL_EXPERIMENTS) * FOLD_COUNT * len(EXPERIMENT_SEEDS)
print('Run directory:', RUN_DIR)
print('Neural experiments:', len(NEURAL_EXPERIMENTS))
print('Expected full neural fits:', expected_neural_runs)

# %% [markdown] [Notebook cell 15]
# ## 7. Fold-local preprocessing, symmetry and deterministic baselines
# 
# Every scaler and monthly climatological intensity is fitted using only the training events of the current outer fold. Windows receive inverse-frequency event weights so a long event cannot dominate merely because it creates more overlapping windows.

# %% [Notebook cell 16]
class WeightedFeatureScaler:
    def fit(self, values, sample_weight):
        values = np.asarray(values, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float)
        if values.ndim != 3 or len(values) != len(sample_weight):
            raise ValueError('Feature scaler expects (sample, time, feature).')
        flat = values.reshape(-1, values.shape[-1])
        weights = np.repeat(sample_weight, values.shape[1])
        self.mean_ = np.average(flat, axis=0, weights=weights)
        variance = np.average((flat - self.mean_) ** 2, axis=0, weights=weights)
        self.scale_ = np.sqrt(np.maximum(variance, 0))
        self.scale_ = np.where(self.scale_ < 1e-8, 1.0, self.scale_)
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        transformed = (values - self.mean_) / self.scale_
        if not np.isfinite(transformed).all():
            raise AssertionError('Feature standardization produced NaN or Inf.')
        return transformed.astype(np.float32)

    def to_dict(self):
        return {'mean': self.mean_.tolist(), 'scale': self.scale_.tolist()}


class WeightedScalarScaler:
    def fit(self, values, sample_weight):
        values = np.asarray(values, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float)
        repeated_weight = np.repeat(sample_weight, int(values[0].size))
        flat = values.reshape(-1)
        self.mean_ = float(np.average(flat, weights=repeated_weight))
        variance = float(np.average((flat - self.mean_) ** 2, weights=repeated_weight))
        self.scale_ = math.sqrt(max(variance, 0))
        if self.scale_ < 1e-8:
            self.scale_ = 1.0
        return self

    def transform(self, values):
        transformed = (np.asarray(values, dtype=float) - self.mean_) / self.scale_
        if not np.isfinite(transformed).all():
            raise AssertionError('Target standardization produced NaN or Inf.')
        return transformed.astype(np.float32)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=float) * self.scale_ + self.mean_

    def to_dict(self):
        return {'mean': self.mean_, 'scale': self.scale_}


def event_balanced_weights(indices):
    indices = np.asarray(indices, dtype=int)
    pairs = WINDOW_META.iloc[indices]['Pair_ID']
    counts = pairs.value_counts()
    weights = pairs.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float)
    return weights / weights.mean()


def split_indices(test_fold):
    validation_fold = (test_fold + 1) % FOLD_COUNT
    fold_values = WINDOW_META['base_fold'].to_numpy(int)
    test = np.flatnonzero(fold_values == test_fold)
    validation = np.flatnonzero(fold_values == validation_fold)
    train = np.flatnonzero(
        (fold_values != test_fold) & (fold_values != validation_fold)
    )
    for name, values in [('train', train), ('validation', validation), ('test', test)]:
        if len(values) == 0:
            raise AssertionError(f'Fold {test_fold} has an empty {name} partition.')
    return train, validation, test


def swap_member_features(values, feature_names):
    values = np.asarray(values)
    output = values.copy()
    lookup = {name: index for index, name in enumerate(feature_names)}
    for index, name in enumerate(feature_names):
        if name.endswith('_m1'):
            partner = name[:-3] + '_m2'
            if partner not in lookup:
                raise KeyError(f'No member-2 partner for {name}')
            output[:, :, index] = values[:, :, lookup[partner]]
        elif name.endswith('_m2'):
            partner = name[:-3] + '_m1'
            if partner not in lookup:
                raise KeyError(f'No member-1 partner for {name}')
            output[:, :, index] = values[:, :, lookup[partner]]
    return output


def swap_targets(values):
    return np.asarray(values)[:, :, ::-1]


def persistence_prediction(indices):
    initial = X_BY_EXPERIMENT['CTL'][indices, -1, :]
    return np.repeat(initial[:, None, :], FORECAST_STEPS, axis=1)


def climatology_prediction(train_indices, test_indices):
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    weights = event_balanced_weights(train_indices)
    train_month = WINDOW_META.iloc[train_indices]['issue_month'].to_numpy(int)

    lead_fallback = np.empty(FORECAST_STEPS, dtype=float)
    month_lead = {}
    for lead in range(FORECAST_STEPS):
        values = Y[train_indices, lead, :].reshape(-1)
        member_weights = np.repeat(weights, 2)
        months = np.repeat(train_month, 2)
        lead_fallback[lead] = np.average(values, weights=member_weights)
        for month in range(1, 13):
            use = months == month
            if use.any():
                month_lead[(month, lead)] = float(
                    np.average(values[use], weights=member_weights[use])
                )

    test_month = WINDOW_META.iloc[test_indices]['issue_month'].to_numpy(int)
    prediction = np.empty((len(test_indices), FORECAST_STEPS, 2), dtype=float)
    for sample, month in enumerate(test_month):
        for lead in range(FORECAST_STEPS):
            climatological_intensity = month_lead.get(
                (month, lead), lead_fallback[lead]
            )
            prediction[sample, lead, :] = climatological_intensity
    return prediction.astype(np.float32), {
        'definition': (
            'training-fold event-balanced monthly mean absolute intensity by lead; '
            'global training-fold lead mean used only if a month is absent'
        ),
        'lead_fallback_intensity_ms': lead_fallback.tolist(),
        'month_lead_intensity_ms': {
            f'{month}_{lead + 1}': value
            for (month, lead), value in month_lead.items()
        },
    }


def prediction_long_table(prediction, indices, experiment, fold, seed, model_kind):
    indices = np.asarray(indices, dtype=int)
    prediction = np.asarray(prediction, dtype=float)
    truth = Y[indices]
    if prediction.shape != truth.shape:
        raise AssertionError(
            f'{experiment}: prediction {prediction.shape} != truth {truth.shape}'
        )
    meta = WINDOW_META.iloc[indices].reset_index(drop=True)
    n_samples = len(meta)
    values_per_window = FORECAST_STEPS * 2
    lead_pattern = np.repeat(
        np.arange(1, FORECAST_STEPS + 1) * STEP_HOURS, 2
    )
    member_pattern = np.tile([1, 2], FORECAST_STEPS)
    lead_h = np.tile(lead_pattern, n_samples)
    member = np.tile(member_pattern, n_samples)
    repeated = meta.loc[meta.index.repeat(values_per_window)].reset_index(drop=True)
    sid = np.where(member == 1, repeated['sid_m1'], repeated['sid_m2'])
    initial = np.where(
        member == 1,
        repeated['initial_intensity_m1_ms'],
        repeated['initial_intensity_m2_ms'],
    )
    result = pd.DataFrame({
        'config_hash': config_hash,
        'experiment': experiment,
        'model_label': EXPERIMENT_LABELS[experiment],
        'model_kind': model_kind,
        'fold': int(fold),
        'seed': int(seed),
        'Pair_ID': repeated['Pair_ID'].to_numpy(),
        'component_id': repeated['component_id'].to_numpy(),
        'window_id': repeated['window_id'].to_numpy(),
        'event_year': repeated['event_year'].to_numpy(int),
        'era': repeated['era'].astype(str).to_numpy(),
        'issue_time': repeated['issue_time'].to_numpy(),
        'lead_h': lead_h,
        'valid_time': repeated['issue_time'] + pd.to_timedelta(lead_h, unit='h'),
        'member': member,
        'sid': sid,
        'initial_intensity_ms': initial,
        'observed_ms': truth.reshape(-1),
        'predicted_ms': prediction.reshape(-1),
        'reviewer_subgroup': repeated['reviewer_subgroup'].to_numpy(bool),
    })
    if not np.isfinite(
        result[['observed_ms', 'predicted_ms']].to_numpy(dtype=float)
    ).all():
        raise AssertionError(f'{experiment} long predictions contain NaN or Inf.')
    return result


# Member-swap utilities must be exact involutions for every experiment.
for experiment in BTC_EXPERIMENTS:
    sample = X_BY_EXPERIMENT[experiment][:3]
    swapped_twice = swap_member_features(
        swap_member_features(sample, FEATURE_NAMES[experiment]),
        FEATURE_NAMES[experiment],
    )
    if not np.allclose(sample, swapped_twice):
        raise AssertionError(f'{experiment} member swapping is not reversible.')

# %% [markdown] [Notebook cell 17]
# ## 8. Neural-network definition
# 
# The BTC architecture follows the previous experiment: bidirectional LSTM(48), dropout 0.30, dense(64) and 24 outputs. Variable input dimensions are accepted directly; CTL is not padded.

# %% [Notebook cell 18]
tf = None
if RUN_TRAINING:
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'TensorFlow is required because RUN_TRAINING=True. Install it in '
            'this notebook kernel, or set RUN_TRAINING=False to analyze an '
            'already completed configuration-hashed run.'
        ) from error


def set_run_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_btc_model(n_features, model_name, seed):
    tf.keras.backend.clear_session()
    set_run_seed(seed)
    inputs = Input(shape=(LOOKBACK, n_features), name='input')
    hidden = Bidirectional(
        LSTM(
            LSTM_UNITS,
            return_sequences=False,
            kernel_regularizer=l2(KERNEL_L2),
            recurrent_regularizer=l2(RECURRENT_L2),
        ),
        name='bilstm',
    )(inputs)
    hidden = Dropout(DROPOUT_RATE, name='dropout')(hidden)
    hidden = Dense(
        DENSE_UNITS, activation='relu', kernel_regularizer=l2(KERNEL_L2), name='dense'
    )(hidden)
    outputs = Dense(FORECAST_STEPS * 2, name='forecast')(hidden)
    model = Model(inputs, outputs, name=model_name)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
    return model


def make_callbacks(model_path=None):
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA, restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=LR_FACTOR, patience=LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE, verbose=0,
        ),
    ]
    if model_path is not None:
        callbacks.append(ModelCheckpoint(
            model_path, monitor='val_loss', save_best_only=True, verbose=0
        ))
    return callbacks


def symmetric_btc_prediction(model, raw_x, feature_names, scaler):
    direct_x = scaler.transform(raw_x)
    swapped_x = scaler.transform(swap_member_features(raw_x, feature_names))
    direct = model.predict(direct_x, batch_size=BATCH_SIZE, verbose=0).reshape(
        -1, FORECAST_STEPS, 2
    )
    swapped = model.predict(swapped_x, batch_size=BATCH_SIZE, verbose=0).reshape(
        -1, FORECAST_STEPS, 2
    )
    swapped_back = swap_targets(swapped)
    return (direct + swapped_back) / 2

# %% [markdown] [Notebook cell 19]
# ## 9. Train 5 folds × 20 seeds with resume support
# 
# Running this cell performs the full training. Each successful run writes one compressed test-prediction file immediately. Rerunning the notebook skips existing files inside the configuration-hashed run directory. Set `SMOKE_TEST=True` only for a short pipeline check; the manuscript run must use the default full settings.

# %% [Notebook cell 20]
RUN_MANIFEST_PATH = RUN_DIR / 'run_manifest.csv'
MODEL_PARAMETER_PATH = AUDIT_DIR / 'model_parameter_counts.csv'


def load_run_manifest():
    if RUN_MANIFEST_PATH.is_file():
        return pd.read_csv(RUN_MANIFEST_PATH, dtype={'config_hash': str})
    return pd.DataFrame()


def upsert_run_manifest(record):
    current = load_run_manifest()
    key_columns = ['experiment', 'fold', 'seed']
    if not current.empty:
        keep = np.ones(len(current), dtype=bool)
        for column in key_columns:
            keep &= current[column].astype(str).eq(str(record[column]))
        current = current.loc[~keep]
    current = pd.concat([current, pd.DataFrame([record])], ignore_index=True)
    temporary = RUN_MANIFEST_PATH.with_name(RUN_MANIFEST_PATH.name + '.tmp')
    current.to_csv(temporary, index=False)
    temporary.replace(RUN_MANIFEST_PATH)


def prediction_file(experiment, fold, seed):
    return PREDICTION_DIR / f'{experiment}__fold{fold}__seed{seed}.csv.gz'


def history_file(experiment, fold, seed):
    return HISTORY_DIR / f'{experiment}__fold{fold}__seed{seed}.csv.gz'


def expected_test_window_ids(fold):
    return set(
        WINDOW_META.loc[WINDOW_META['base_fold'].eq(fold), 'window_id'].astype(str)
    )


def manifest_run_succeeded(experiment, fold, seed):
    manifest = load_run_manifest()
    if manifest.empty:
        return False
    required = {
        'config_hash', 'experiment', 'fold', 'seed', 'status',
        'training_mode', 'requested_epochs',
    }
    if not required.issubset(manifest.columns):
        return False
    match = manifest[
        manifest['config_hash'].astype(str).eq(config_hash)
        & manifest['experiment'].astype(str).eq(str(experiment))
        & pd.to_numeric(manifest['fold'], errors='coerce').eq(int(fold))
        & pd.to_numeric(manifest['seed'], errors='coerce').eq(int(seed))
        & manifest['training_mode'].astype(str).eq(
            'smoke' if SMOKE_TEST else 'formal'
        )
        & pd.to_numeric(manifest['requested_epochs'], errors='coerce').eq(
            int(min(MAX_EPOCHS, 2) if SMOKE_TEST else MAX_EPOCHS)
        )
    ]
    return (not match.empty) and match.iloc[-1]['status'] == 'success'


def validate_saved_prediction(path, experiment, fold, seed, require_success_manifest):
    path = Path(path)
    if not path.is_file():
        return False, 'file missing'
    try:
        frame = pd.read_csv(path, dtype={'config_hash': str})
        required = {
            'config_hash', 'experiment', 'fold', 'seed', 'Pair_ID',
            'component_id', 'window_id', 'event_year', 'era', 'lead_h',
            'member', 'sid', 'observed_ms', 'predicted_ms',
            'reviewer_subgroup',
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            return False, f'missing columns {missing}'
        if set(frame['config_hash'].astype(str)) != {config_hash}:
            return False, 'configuration hash differs'
        if set(frame['experiment'].astype(str)) != {str(experiment)}:
            return False, 'experiment identity differs'
        if set(pd.to_numeric(frame['fold'], errors='coerce')) != {int(fold)}:
            return False, 'fold identity differs'
        if set(pd.to_numeric(frame['seed'], errors='coerce')) != {int(seed)}:
            return False, 'seed identity differs'
        expected_windows = expected_test_window_ids(fold)
        if set(frame['window_id'].astype(str)) != expected_windows:
            return False, 'test-window manifest differs'
        expected_meta = (
            WINDOW_META.loc[
                WINDOW_META['base_fold'].eq(fold),
                [
                    'window_id', 'Pair_ID', 'component_id', 'event_year',
                    'era', 'reviewer_subgroup',
                ],
            ]
            .assign(
                window_id=lambda value: value['window_id'].astype(str),
                Pair_ID=lambda value: value['Pair_ID'].astype(str),
                component_id=lambda value: value['component_id'].astype(str),
                era=lambda value: value['era'].astype(str),
                reviewer_subgroup=lambda value: value['reviewer_subgroup'].astype(bool),
            )
            .sort_values('window_id')
            .reset_index(drop=True)
        )
        saved_meta = (
            frame[[
                'window_id', 'Pair_ID', 'component_id', 'event_year',
                'era', 'reviewer_subgroup',
            ]]
            .drop_duplicates()
            .assign(
                window_id=lambda value: value['window_id'].astype(str),
                Pair_ID=lambda value: value['Pair_ID'].astype(str),
                component_id=lambda value: value['component_id'].astype(str),
                era=lambda value: value['era'].astype(str),
                reviewer_subgroup=lambda value: (
                    value['reviewer_subgroup'].astype(str).str.lower().eq('true')
                ),
            )
            .sort_values('window_id')
            .reset_index(drop=True)
        )
        if not saved_meta.equals(expected_meta):
            return False, 'Pair/component/year/subgroup metadata differs'
        expected_rows = len(expected_windows) * FORECAST_STEPS * 2
        if len(frame) != expected_rows:
            return False, f'expected {expected_rows} rows, found {len(frame)}'
        if set(pd.to_numeric(frame['lead_h'], errors='coerce')) != set(
            np.arange(1, FORECAST_STEPS + 1) * STEP_HOURS
        ):
            return False, 'forecast leads differ'
        if set(pd.to_numeric(frame['member'], errors='coerce')) != {1, 2}:
            return False, 'member values differ'
        if frame.duplicated(['window_id', 'lead_h', 'member']).any():
            return False, 'duplicate forecast keys'
        if not np.isfinite(
            frame[['observed_ms', 'predicted_ms']].to_numpy(dtype=float)
        ).all():
            return False, 'non-finite observed/predicted value'
        if require_success_manifest and not manifest_run_succeeded(
            experiment, fold, seed
        ):
            return False, 'run manifest is not success'
    except Exception as error:
        return False, repr(error)
    return True, 'ok'


def write_prediction_atomically(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.is_file():
        temporary.unlink()
    try:
        frame.to_csv(temporary, index=False, compression='gzip')
        temporary.replace(path)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise


def save_preprocessor(experiment, fold, x_scaler, y_scaler, train_indices):
    payload = {
        'experiment': experiment,
        'fold': int(fold),
        'train_pair_ids': sorted(WINDOW_META.iloc[train_indices]['Pair_ID'].unique()),
        'train_window_count': int(len(train_indices)),
        'x_scaler': x_scaler.to_dict(),
        'y_scaler': y_scaler.to_dict(),
        'fit_scope': 'training partition only',
    }
    with (PREPROCESSING_DIR / f'{experiment}__fold{fold}.json').open(
        'w', encoding='utf-8'
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


active_folds = [0] if SMOKE_TEST else list(range(FOLD_COUNT))
active_seeds = EXPERIMENT_SEEDS[:2] if SMOKE_TEST else EXPERIMENT_SEEDS
active_experiments = NEURAL_EXPERIMENTS
epochs_this_run = min(MAX_EPOCHS, 2) if SMOKE_TEST else MAX_EPOCHS
model_parameter_records = []

if RUN_TRAINING:
    total_requested = len(active_experiments) * len(active_folds) * len(active_seeds)
    completed_counter = 0

    for fold in active_folds:
        train_indices, validation_indices, test_indices = split_indices(fold)


        for experiment in active_experiments:
            raw_x = X_BY_EXPERIMENT[experiment]
            train_weight_base = event_balanced_weights(train_indices)
            validation_weight_base = event_balanced_weights(validation_indices)

            x_train_original = raw_x[train_indices]
            x_train_swapped = swap_member_features(
                x_train_original, FEATURE_NAMES[experiment]
            )
            x_train_raw = np.concatenate(
                [x_train_original, x_train_swapped], axis=0
            )
            y_train_raw = np.concatenate(
                [Y[train_indices], swap_targets(Y[train_indices])], axis=0
            )
            train_weight = np.concatenate([
                train_weight_base / 2, train_weight_base / 2
            ])

            x_validation_original = raw_x[validation_indices]
            x_validation_swapped = swap_member_features(
                x_validation_original, FEATURE_NAMES[experiment]
            )
            x_validation_raw = np.concatenate(
                [x_validation_original, x_validation_swapped], axis=0
            )
            y_validation_raw = np.concatenate([
                Y[validation_indices], swap_targets(Y[validation_indices])
            ], axis=0)
            validation_weight = np.concatenate([
                validation_weight_base / 2, validation_weight_base / 2
            ])

            # Scalers see only training data, including its member-swapped copy.
            x_scaler = WeightedFeatureScaler().fit(x_train_raw, train_weight)
            y_scaler = WeightedScalarScaler().fit(y_train_raw, train_weight)
            x_train = x_scaler.transform(x_train_raw)
            x_validation = x_scaler.transform(x_validation_raw)
            y_train = y_scaler.transform(y_train_raw).reshape(-1, FORECAST_STEPS * 2)
            y_validation = y_scaler.transform(y_validation_raw).reshape(
                -1, FORECAST_STEPS * 2
            )
            save_preprocessor(
                experiment, fold, x_scaler, y_scaler, train_indices
            )

            for seed in active_seeds:
                completed_counter += 1
                output_path = prediction_file(experiment, fold, seed)
                if (
                    output_path.is_file() and RESUME_EXISTING_RUNS
                    and (
                        not SAVE_MODELS
                        or (MODEL_DIR / f'{experiment}__fold{fold}__seed{seed}.keras').is_file()
                    )
                ):
                    valid, reason = validate_saved_prediction(
                        output_path, experiment, fold, seed,
                        require_success_manifest=True,
                    )
                    if valid:
                        print(
                            f'[{completed_counter}/{total_requested}] reuse '
                            f'{experiment}, fold={fold}, seed={seed}'
                        )
                        continue
                    if reason == 'run manifest is not success':
                        orphan_path = output_path.with_name(
                            output_path.name + f'.orphan_{time.time_ns()}'
                        )
                        output_path.replace(orphan_path)
                        print(
                            f'Quarantined crash-interrupted prediction as '
                            f'{orphan_path.name}; retraining this run.'
                        )
                    else:
                        raise RuntimeError(
                            f'Unsafe resume for {output_path.name}: {reason}. '
                            'Use the new configuration-hashed run directory or '
                            'move the invalid file out of the run directory.'
                        )

                start_time = time.time()
                record = {
                    'config_hash': config_hash,
                    'training_mode': 'smoke' if SMOKE_TEST else 'formal',
                    'requested_epochs': epochs_this_run,
                    'experiment': experiment,
                    'fold': fold,
                    'seed': seed,
                    'status': 'started',
                    'best_epoch': np.nan,
                    'best_validation_loss': np.nan,
                    'epochs_completed': 0,
                    'train_windows': len(train_indices),
                    'validation_windows': len(validation_indices),
                    'test_windows': len(test_indices),
                    'prediction_file': str(output_path),
                    'error': '',
                }
                upsert_run_manifest(record)
                print(
                    f'[{completed_counter}/{total_requested}] train '
                    f'{experiment}, fold={fold}, seed={seed}'
                )

                try:
                    model_path = (
                        MODEL_DIR / f'{experiment}__fold{fold}__seed{seed}.keras'
                        if SAVE_MODELS else None
                    )
                    model = build_btc_model(
                        len(FEATURE_NAMES[experiment]),
                        f'{experiment.lower()}_f{fold}_s{seed}', seed,
                    )

                    trainable_parameter_count = int(sum(
                        np.prod(variable.shape) for variable in model.trainable_weights
                    ))
                    model_parameter_records.append({
                        'experiment': experiment,
                        'fold': fold,
                        'seed': seed,
                        'n_features': len(FEATURE_NAMES[experiment]),
                        'trainable_parameters': trainable_parameter_count,
                    })
                    history = model.fit(
                        x_train, y_train,
                        sample_weight=train_weight,
                        validation_data=(
                            x_validation, y_validation, validation_weight
                        ),
                        epochs=epochs_this_run,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=make_callbacks(model_path),
                        verbose=TRAIN_VERBOSE,
                    )
                    history_frame = pd.DataFrame(history.history)
                    history_frame.insert(0, 'epoch', np.arange(1, len(history_frame) + 1))
                    history_frame.insert(1, 'experiment', experiment)
                    history_frame.insert(2, 'fold', fold)
                    history_frame.insert(3, 'seed', seed)
                    history_numeric = history_frame.select_dtypes(include=[np.number])
                    if not np.isfinite(history_numeric.to_numpy(dtype=float)).all():
                        raise FloatingPointError('Training history contains NaN or Inf.')
                    history_frame.to_csv(
                        history_file(experiment, fold, seed),
                        index=False, compression='gzip',
                    )

                    scaled_prediction = symmetric_btc_prediction(
                        model,
                        X_BY_EXPERIMENT[experiment][test_indices],
                        FEATURE_NAMES[experiment],
                        x_scaler,
                    )
                    prediction = y_scaler.inverse_transform(scaled_prediction)

                    prediction_frame = prediction_long_table(
                        prediction, test_indices, experiment, fold, seed,
                        model_kind='BTC neural',
                    )

                    validation_loss = np.asarray(history.history['val_loss'], dtype=float)
                    if validation_loss.size == 0 or not np.isfinite(validation_loss).all():
                        raise FloatingPointError('Validation loss contains NaN or Inf.')
                    best_index = int(np.argmin(validation_loss))
                    write_prediction_atomically(prediction_frame, output_path)
                    record.update({
                        'status': 'success',
                        'best_epoch': best_index + 1,
                        'best_validation_loss': float(validation_loss[best_index]),
                        'epochs_completed': len(history_frame),
                        'elapsed_seconds': time.time() - start_time,
                        'trainable_parameters': trainable_parameter_count,
                    })
                    upsert_run_manifest(record)
                except Exception as error:
                    if output_path.is_file():
                        output_path.unlink()
                    record.update({
                        'status': 'failed',
                        'elapsed_seconds': time.time() - start_time,
                        'error': repr(error),
                    })
                    upsert_run_manifest(record)
                    if FAIL_FAST:
                        raise
                    print('FAILED:', repr(error))
                finally:
                    if 'model' in locals():
                        del model
                    tf.keras.backend.clear_session()

    parameter_parts = []
    if MODEL_PARAMETER_PATH.is_file():
        parameter_parts.append(pd.read_csv(MODEL_PARAMETER_PATH))
    if model_parameter_records:
        parameter_parts.append(pd.DataFrame(model_parameter_records))
    manifest_parameters = load_run_manifest()
    required_parameter_columns = {
        'experiment', 'fold', 'seed', 'status', 'trainable_parameters'
    }
    if required_parameter_columns.issubset(manifest_parameters.columns):
        parameter_parts.append(
            manifest_parameters.loc[
                manifest_parameters['status'].eq('success'),
                ['experiment', 'fold', 'seed', 'trainable_parameters'],
            ].assign(
                n_features=lambda frame: frame['experiment'].map(
                    lambda value: len(FEATURE_NAMES[value])
                )
            )
        )
    if parameter_parts:
        parameter_table = (
            pd.concat(parameter_parts, ignore_index=True)
            .sort_values(['experiment', 'fold', 'seed'])
            .drop_duplicates(['experiment', 'fold', 'seed'], keep='last')
        )
        parameter_table.to_csv(MODEL_PARAMETER_PATH, index=False)
else:
    print('RUN_TRAINING=False: training loop skipped.')
