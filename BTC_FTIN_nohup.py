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
# - Data source: `both_record_start_222_matched_environment_points.csv`.
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
# The main figure contains performance, bias, paired path effects and seed/fold uncertainty. Detailed environmental/path curves, the former mean-intensity composite, calibration, cohort/fold audits and training diagnostics are written to the supplementary-figure directory.

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

DATA_PATH = ROOT / 'both_record_start_222_matched_environment_points.csv'

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
SAVE_MODELS = False
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
if DATA_PATH.name != 'both_record_start_222_matched_environment_points.csv':
    raise AssertionError('The new both-record-start matched CSV was not selected.')
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
MAIN_FIGURE_DIR = RUN_DIR / 'main_figure'
SUPPLEMENTARY_DIR = RUN_DIR / 'supplementary_figures'
for directory in [
    AUDIT_DIR, PREDICTION_DIR, HISTORY_DIR, MODEL_DIR, PREPROCESSING_DIR,
    TABLE_DIR, MAIN_FIGURE_DIR, SUPPLEMENTARY_DIR,
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
                if output_path.is_file() and RESUME_EXISTING_RUNS:
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

# %% [markdown] [Notebook cell 21]
# ## 10. Validate prediction files and calculate event-balanced metrics
# 
# The analysis reads per-run compressed prediction files incrementally rather than loading all 1,100 runs into memory. Bias is always prediction minus observation. MAE, MSE and bias are first averaged within event and then across events; RMSE is the square root of the event-balanced mean MSE. Deterministic baselines remain one result per fold and are not falsely replicated as 20 independent runs.

# %% [Notebook cell 22]
# Evaluation-only references: no neural fitting or seed repetitions.
for fold in active_folds:
    train_indices, _, test_indices = split_indices(fold)
    # Deterministic baselines are saved once per fold, not replicated as 20 seeds.
    for baseline in ['PERSISTENCE', 'CLIMATOLOGY']:
        baseline_path = prediction_file(baseline, fold, -1)
        if baseline_path.is_file() and RESUME_EXISTING_RUNS:
            valid, reason = validate_saved_prediction(
                baseline_path, baseline, fold, -1,
                require_success_manifest=False,
            )
            if valid:
                continue
            raise RuntimeError(
                f'Unsafe baseline resume for {baseline_path.name}: {reason}. '
                'Use the new configuration-hashed run directory or move the '
                'invalid file out of the run directory.'
            )
        if baseline == 'PERSISTENCE':
            baseline_prediction = persistence_prediction(test_indices)
            baseline_payload = {'definition': 'last observed intensity repeated'}
        else:
            baseline_prediction, baseline_payload = climatology_prediction(
                train_indices, test_indices
            )
            with (PREPROCESSING_DIR / f'{baseline}__fold{fold}.json').open(
                'w', encoding='utf-8'
            ) as handle:
                json.dump(baseline_payload, handle, ensure_ascii=False, indent=2)
        baseline_frame = prediction_long_table(
            baseline_prediction, test_indices, baseline, fold, -1,
            model_kind='deterministic baseline',
        )
        write_prediction_atomically(baseline_frame, baseline_path)


def weighted_correlation(x, y, weight):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    if valid.sum() < 2:
        return np.nan
    x, y, weight = x[valid], y[valid], weight[valid]
    weight = weight / weight.sum()
    mean_x = np.sum(weight * x)
    mean_y = np.sum(weight * y)
    covariance = np.sum(weight * (x - mean_x) * (y - mean_y))
    variance_x = np.sum(weight * (x - mean_x) ** 2)
    variance_y = np.sum(weight * (y - mean_y) ** 2)
    if variance_x <= 0 or variance_y <= 0:
        return np.nan
    return float(covariance / np.sqrt(variance_x * variance_y))


def event_balanced_metrics(frame):
    work = frame.copy()
    work['error'] = work['predicted_ms'] - work['observed_ms']
    work['absolute_error'] = work['error'].abs()
    work['squared_error'] = work['error'] ** 2
    event_stats = (
        work.groupby('Pair_ID')
        .agg(
            event_bias=('error', 'mean'),
            event_mae=('absolute_error', 'mean'),
            event_mse=('squared_error', 'mean'),
            n_values=('error', 'size'),
        )
    )
    point_count = work.groupby('Pair_ID')['error'].transform('size')
    point_weight = 1.0 / point_count.to_numpy(float)
    return {
        'Bias': float(event_stats['event_bias'].mean()),
        'MAE': float(event_stats['event_mae'].mean()),
        'RMSE': float(np.sqrt(event_stats['event_mse'].mean())),
        'Pearson_R': weighted_correlation(
            work['observed_ms'], work['predicted_ms'], point_weight
        ),
        'n_events': int(event_stats.shape[0]),
        'n_windows': int(work['window_id'].nunique()),
        'n_values': int(len(work)),
    }


def metrics_from_prediction_file(frame):
    identity = frame.iloc[0]
    records = []
    for lead_h, group in frame.groupby('lead_h'):
        records.append({
            'experiment': identity['experiment'],
            'model_label': identity['model_label'],
            'model_kind': identity['model_kind'],
            'fold': int(identity['fold']),
            'seed': int(identity['seed']),
            'horizon': f'{int(lead_h)}h',
            'lead_h': int(lead_h),
            **event_balanced_metrics(group),
        })
    aggregate = frame[frame['lead_h'].between(6, 72)]
    records.append({
        'experiment': identity['experiment'],
        'model_label': identity['model_label'],
        'model_kind': identity['model_kind'],
        'fold': int(identity['fold']),
        'seed': int(identity['seed']),
        'horizon': '6-72h',
        'lead_h': np.nan,
        **event_balanced_metrics(aggregate),
    })
    late = frame[frame['lead_h'].between(36, 72)]
    records.append({
        'experiment': identity['experiment'],
        'model_label': identity['model_label'],
        'model_kind': identity['model_kind'],
        'fold': int(identity['fold']),
        'seed': int(identity['seed']),
        'horizon': '36-72h',
        'lead_h': np.nan,
        **event_balanced_metrics(late),
    })
    return records


def event_sufficient_statistics_from_file(frame):
    work = frame.copy()
    work['error'] = work['predicted_ms'] - work['observed_ms']
    work['absolute_error'] = work['error'].abs()
    work['squared_error'] = work['error'] ** 2
    identity = work.iloc[0]
    identity_columns = [
        'Pair_ID', 'component_id', 'event_year', 'era', 'reviewer_subgroup'
    ]
    parts = []
    for horizon, scoped in [
        ('6-72h', work[work['lead_h'].between(6, 72)]),
        ('36-72h', work[work['lead_h'].between(36, 72)]),
    ]:
        stats = scoped.groupby(identity_columns, as_index=False).agg(
            bias=('error', 'mean'),
            mae=('absolute_error', 'mean'),
            mse=('squared_error', 'mean'),
            n_values=('error', 'size'),
        )
        stats['horizon'] = horizon
        stats['lead_h'] = np.nan
        parts.append(stats)
    by_lead = work.groupby(identity_columns + ['lead_h'], as_index=False).agg(
        bias=('error', 'mean'),
        mae=('absolute_error', 'mean'),
        mse=('squared_error', 'mean'),
        n_values=('error', 'size'),
    )
    by_lead['horizon'] = by_lead['lead_h'].astype(int).astype(str) + 'h'
    parts.append(by_lead)
    result = pd.concat(parts, ignore_index=True)
    result.insert(0, 'experiment', str(identity['experiment']))
    result.insert(1, 'model_label', str(identity['model_label']))
    result.insert(2, 'model_kind', str(identity['model_kind']))
    result.insert(3, 'fold', int(identity['fold']))
    result.insert(4, 'seed', int(identity['seed']))
    return result


def attach_current_prediction_metadata(frame):
    """Use current audited metadata; never trust a stale label in a saved file."""
    result = frame.copy()
    current = WINDOW_META.set_index('window_id')
    for column in [
        'Pair_ID', 'component_id', 'event_year', 'era', 'reviewer_subgroup'
    ]:
        mapped = result['window_id'].map(current[column])
        if mapped.isna().any():
            raise AssertionError(f'Current metadata mapping failed for {column}.')
        result[column] = mapped.to_numpy()
    result['reviewer_subgroup'] = result['reviewer_subgroup'].astype(bool)
    return result


inventory_rows = []
expected_items = []
for fold in active_folds:
    for baseline in ['PERSISTENCE', 'CLIMATOLOGY']:
        expected_items.append((baseline, fold, -1, prediction_file(baseline, fold, -1)))
    for experiment in active_experiments:
        for seed in active_seeds:
            expected_items.append((experiment, fold, seed, prediction_file(experiment, fold, seed)))

missing_prediction_files = []
metric_rows = []
event_stat_parts = []
reference_manifest_hash = {}
reference_observation_hash = {}
for experiment, fold, seed, path in expected_items:
    exists = path.is_file()
    inventory_rows.append({
        'experiment': experiment, 'fold': fold, 'seed': seed,
        'path': str(path), 'exists': exists,
    })
    if not exists:
        missing_prediction_files.append(path)
        continue
    frame = pd.read_csv(path, dtype={'config_hash': str})
    required_prediction_columns = [
        'config_hash', 'experiment', 'fold', 'seed', 'Pair_ID', 'component_id',
        'window_id', 'event_year', 'era', 'issue_time', 'lead_h', 'member',
        'sid', 'observed_ms', 'predicted_ms', 'reviewer_subgroup'
    ]
    missing = [column for column in required_prediction_columns if column not in frame.columns]
    if missing:
        raise KeyError(f'{path.name} lacks columns: {missing}')
    valid, reason = validate_saved_prediction(
        path, experiment, fold, seed,
        require_success_manifest=seed >= 0,
    )
    if not valid:
        raise AssertionError(f'{path.name} failed validation: {reason}')
    frame = attach_current_prediction_metadata(frame)
    if frame.duplicated(['window_id', 'lead_h', 'member']).any():
        raise AssertionError(f'Duplicate predictions in {path.name}')
    key_text = '\n'.join(
        frame.sort_values(['window_id', 'lead_h', 'member'])
        .loc[:, ['window_id', 'lead_h', 'member']]
        .astype(str).agg('|'.join, axis=1)
    )
    manifest_digest = hashlib.sha256(key_text.encode('utf-8')).hexdigest()
    if fold not in reference_manifest_hash and experiment == 'CTL':
        reference_manifest_hash[fold] = manifest_digest
        observation_text = '\n'.join(
            frame.sort_values(['window_id', 'lead_h', 'member'])
            .apply(
                lambda row: (
                    f"{row['window_id']}|{int(row['lead_h'])}|{int(row['member'])}|"
                    f"{row['sid']}|{float(row['observed_ms']):.10f}"
                ),
                axis=1,
            )
        )
        reference_observation_hash[fold] = hashlib.sha256(
            observation_text.encode('utf-8')
        ).hexdigest()
    metric_rows.extend(metrics_from_prediction_file(frame))
    event_stat_parts.append(event_sufficient_statistics_from_file(frame))

prediction_inventory = pd.DataFrame(inventory_rows)
prediction_inventory.to_csv(AUDIT_DIR / 'prediction_file_manifest.csv', index=False)

if missing_prediction_files and not ALLOW_PARTIAL_RESULTS:
    raise FileNotFoundError(
        f'{len(missing_prediction_files)} expected prediction files are missing. '
        'Finish or resume training before running result cells. First missing file: '
        f'{missing_prediction_files[0]}'
    )

# Second pass: every comparable run must use the exact CTL test manifest for its fold.
for experiment, fold, seed, path in expected_items:
    if not path.is_file() or fold not in reference_manifest_hash:
        continue
    frame = pd.read_csv(
        path, usecols=['window_id', 'lead_h', 'member', 'sid', 'observed_ms']
    )
    key_text = '\n'.join(
        frame.sort_values(['window_id', 'lead_h', 'member'])
        .loc[:, ['window_id', 'lead_h', 'member']]
        .astype(str).agg('|'.join, axis=1)
    )
    digest = hashlib.sha256(key_text.encode('utf-8')).hexdigest()
    if digest != reference_manifest_hash[fold]:
        raise AssertionError(
            f'{experiment}, fold={fold}, seed={seed} does not match the CTL test cohort.'
        )
    observation_text = '\n'.join(
        frame.sort_values(['window_id', 'lead_h', 'member'])
        .apply(
            lambda row: (
                f"{row['window_id']}|{int(row['lead_h'])}|{int(row['member'])}|"
                f"{row['sid']}|{float(row['observed_ms']):.10f}"
            ),
            axis=1,
        )
    )
    observation_digest = hashlib.sha256(
        observation_text.encode('utf-8')
    ).hexdigest()
    if observation_digest != reference_observation_hash[fold]:
        raise AssertionError(
            f'{experiment}, fold={fold}, seed={seed} has different SID/observations.'
        )

metrics_all_horizons = pd.DataFrame(metric_rows)
event_stats_oof = pd.concat(event_stat_parts, ignore_index=True)
event_stats_oof.to_csv(
    TABLE_DIR / 'event_level_oof_sufficient_statistics.csv.gz',
    index=False, compression='gzip',
)
metrics_by_lead = metrics_all_horizons[
    ~metrics_all_horizons['horizon'].isin(['6-72h', '36-72h'])
].copy()
metrics_6_72 = metrics_all_horizons[metrics_all_horizons['horizon'] == '6-72h'].copy()
metrics_36_72 = metrics_all_horizons[metrics_all_horizons['horizon'] == '36-72h'].copy()
metrics_by_lead.to_csv(TABLE_DIR / 'metrics_per_run_by_lead.csv', index=False)
metrics_6_72.to_csv(TABLE_DIR / 'metrics_per_run_6_72h.csv', index=False)
metrics_36_72.to_csv(TABLE_DIR / 'metrics_per_run_36_72h.csv', index=False)

seed_oof_rows = []
for (experiment, seed, horizon, lead_h), group in event_stats_oof.groupby(
    ['experiment', 'seed', 'horizon', 'lead_h'], dropna=False
):
    seed_oof_rows.append({
        'experiment': experiment,
        'model_label': EXPERIMENT_LABELS[experiment],
        'seed': int(seed),
        'horizon': horizon,
        'lead_h': lead_h,
        'Bias': float(group['bias'].mean()),
        'MAE': float(group['mae'].mean()),
        'RMSE': float(np.sqrt(group['mse'].mean())),
        'n_events': int(group['Pair_ID'].nunique()),
        'n_components': int(group['component_id'].nunique()),
    })
seed_oof_metrics = pd.DataFrame(seed_oof_rows)
seed_oof_metrics.to_csv(TABLE_DIR / 'metrics_oof_by_seed.csv', index=False)

# Compute SS only after joining all held-out folds for each seed.
# MSESS = 1 - MSE(model)/MSE(reference); MAESS uses absolute errors.
ss_parts = []
for reference in ['PERSISTENCE', 'CLIMATOLOGY']:
    reference_metrics = seed_oof_metrics[
        seed_oof_metrics['experiment'].eq(reference)
    ][['horizon', 'MAE', 'RMSE']].rename(columns={
        'MAE': 'reference_MAE', 'RMSE': 'reference_RMSE'
    })
    result = seed_oof_metrics[seed_oof_metrics['seed'].ge(0)].merge(
        reference_metrics, on='horizon', validate='many_to_one'
    )
    result['reference'] = reference
    result['MAESS'] = np.where(
        result['reference_MAE'] > 0,
        1 - result['MAE'] / result['reference_MAE'], np.nan,
    )
    result['MSESS'] = np.where(
        result['reference_RMSE'] > 0,
        1 - (result['RMSE'] / result['reference_RMSE']) ** 2, np.nan,
    )
    ss_parts.append(result)
test_skill_scores = pd.concat(ss_parts, ignore_index=True)
test_skill_scores.to_csv(TABLE_DIR / 'test_SS_by_oof_seed_and_lead.csv', index=False)
test_skill_scores.groupby(['experiment', 'reference', 'horizon'])[
    ['MAESS', 'MSESS']
].agg(['mean', 'std', 'median', 'min', 'max']).to_csv(
    TABLE_DIR / 'test_SS_seed_summary.csv'
)

primary_seed_metrics = seed_oof_metrics[
    seed_oof_metrics['horizon'].eq('6-72h')
    & seed_oof_metrics['seed'].ge(0)
].copy()
ctl_primary = primary_seed_metrics[
    primary_seed_metrics['experiment'].eq('CTL')
][['seed', 'Bias', 'MAE', 'RMSE']].rename(columns={
    'Bias': 'CTL_Bias', 'MAE': 'CTL_MAE', 'RMSE': 'CTL_RMSE',
})
paired_differences = (
    primary_seed_metrics[~primary_seed_metrics['experiment'].eq('CTL')]
    .merge(ctl_primary, on='seed', how='left', validate='many_to_one')
)
for metric in ['Bias', 'MAE', 'RMSE']:
    paired_differences[f'{metric}_delta_vs_CTL'] = (
        paired_differences[metric] - paired_differences[f'CTL_{metric}']
    )
paired_differences.to_csv(
    TABLE_DIR / 'paired_differences_vs_ctl_oof_by_seed_6_72h.csv', index=False
)

baseline_skill_rows = []
for baseline in ['PERSISTENCE', 'CLIMATOLOGY']:
    baseline_table = metrics_all_horizons[
        metrics_all_horizons['experiment'].eq(baseline)
    ][['fold', 'horizon', 'MAE', 'RMSE']].rename(columns={
        'MAE': 'baseline_MAE', 'RMSE': 'baseline_RMSE'
    })
    comparison = metrics_all_horizons[
        metrics_all_horizons['seed'].ge(0)
    ].merge(baseline_table, on=['fold', 'horizon'], how='left', validate='many_to_one')
    comparison['reference_baseline'] = baseline
    comparison['MAE_skill'] = 1 - comparison['MAE'] / comparison['baseline_MAE']
    comparison['RMSE_skill'] = 1 - comparison['RMSE'] / comparison['baseline_RMSE']
    baseline_skill_rows.append(comparison)
baseline_skill = pd.concat(baseline_skill_rows, ignore_index=True)
baseline_skill.to_csv(TABLE_DIR / 'baseline_skill_scores.csv', index=False)

summary_rows = []
for experiment, group in primary_seed_metrics.groupby('experiment'):
    for metric in ['Bias', 'MAE', 'RMSE']:
        values = group[metric].dropna().to_numpy(float)
        summary_rows.append({
            'experiment': experiment,
            'model_label': EXPERIMENT_LABELS[experiment],
            'metric': metric,
            'n_complete_oof_seeds': len(values),
            'mean': float(np.mean(values)),
            'sd': float(np.std(values, ddof=1)),
            'median': float(np.median(values)),
            'q25': float(np.quantile(values, 0.25)),
            'q75': float(np.quantile(values, 0.75)),
            'q025': float(np.quantile(values, 0.025)),
            'q975': float(np.quantile(values, 0.975)),
        })
seed_fold_summary = pd.DataFrame(summary_rows)
seed_fold_summary.to_csv(TABLE_DIR / 'metrics_oof_seed_dispersion_6_72h.csv', index=False)

print('Prediction files found:', int(prediction_inventory['exists'].sum()))
print('Per-run/lead metric rows:', len(metrics_by_lead))
display(seed_fold_summary.head())

# %% [markdown] [Notebook cell 23]
# ## 11. Single-seed five-fold OOF inference
# 
# The primary estimand is the expected performance of one trained model, not the performance of an average of 20 predictions. Each seed is first joined across all five disjoint test folds to form one complete OOF result. The 20 complete OOF effects provide training-noise dispersion. Confidence intervals resample the shared-SID connected components and the matched seeds. The primary horizon is the predeclared full 6–72 h range; 36–72 h is retained only as a sensitivity analysis.

# %% [Notebook cell 24]
PRIMARY_HORIZON = '6-72h'
SENSITIVITY_HORIZON = '36-72h'
DETERMINISTIC_EXPERIMENTS = {'PERSISTENCE', 'CLIMATOLOGY'}
PRIMARY_MODELS = [
    'PERSISTENCE', 'CLIMATOLOGY', 'CTL'
] + PATH_EXPERIMENTS + ENVIRONMENT_EXPERIMENTS

analysis_events = event_table[
    event_table['base_fold'].isin(active_folds)
][['Pair_ID', 'component_id', 'era']].copy()
analysis_events = analysis_events.merge(
    event_subgroup[['Pair_ID', 'reviewer_subgroup']],
    on='Pair_ID', how='left', validate='one_to_one',
)
analysis_events['reviewer_subgroup'] = (
    analysis_events['reviewer_subgroup'].fillna(False).astype(bool)
)
EVENT_ORDER = analysis_events.sort_values('Pair_ID')['Pair_ID'].tolist()
reviewer_subgroup_n = int(analysis_events['reviewer_subgroup'].sum())
reviewer_subgroup_available = (
    land_distance_available and 0 < reviewer_subgroup_n < len(analysis_events)
)
if REQUIRE_REVIEWER_SUBGROUP and not land_distance_available:
    raise RuntimeError(
        'The reviewer subgroup is required, but Cartopy/Shapely/Natural Earth '
        'land distance is unavailable. Install/repair the dependency before the '
        'formal run.'
    )
if REQUIRE_REVIEWER_SUBGROUP and reviewer_subgroup_n == 0:
    raise RuntimeError('The required reviewer subgroup contains zero events.')
if reviewer_subgroup_n < MIN_REVIEWER_SUBGROUP_EVENTS:
    print(
        'WARNING: reviewer subgroup has only', reviewer_subgroup_n,
        'events; subgroup intervals are descriptive, not confirmatory.'
    )


def scoped_event_stats(experiment, seed, horizon):
    result = event_stats_oof[
        event_stats_oof['experiment'].eq(experiment)
        & event_stats_oof['seed'].eq(seed)
        & event_stats_oof['horizon'].eq(horizon)
    ].copy()
    result = result.drop_duplicates('Pair_ID')
    if set(result['Pair_ID']) != set(EVENT_ORDER):
        raise AssertionError(
            f'{experiment}, seed={seed}, horizon={horizon}: incomplete OOF events.'
        )
    return result


def absolute_event_seed_matrix(experiment, horizon):
    seeds = np.asarray(
        [-1] if experiment in DETERMINISTIC_EXPERIMENTS else active_seeds,
        dtype=int,
    )
    arrays = {}
    for metric in ['bias', 'mae', 'mse']:
        columns = []
        for seed in seeds:
            scoped = scoped_event_stats(experiment, int(seed), horizon)
            values = scoped.set_index('Pair_ID').reindex(EVENT_ORDER)[metric]
            if values.isna().any():
                raise AssertionError(f'Incomplete {experiment} {metric} matrix.')
            columns.append(values.to_numpy(float))
        arrays[metric] = np.column_stack(columns)
    return {
        'meta': analysis_events.set_index('Pair_ID').reindex(EVENT_ORDER).reset_index(),
        'seeds': seeds,
        **arrays,
    }


def absolute_metric_vector(matrix, event_indices, seed_indices):
    index = np.ix_(event_indices, seed_indices)
    bias_by_seed = matrix['bias'][index].mean(axis=0)
    mae_by_seed = matrix['mae'][index].mean(axis=0)
    rmse_by_seed = np.sqrt(matrix['mse'][index].mean(axis=0))
    return np.asarray([
        bias_by_seed.mean(), mae_by_seed.mean(), rmse_by_seed.mean()
    ])


def component_rows(meta):
    component_values = meta['component_id'].to_numpy()
    return [
        np.flatnonzero(component_values == component)
        for component in pd.unique(component_values)
    ]


def absolute_component_seed_bootstrap(experiment, horizon, rng):
    matrix = absolute_event_seed_matrix(experiment, horizon)
    all_events = np.arange(len(matrix['meta']), dtype=int)
    all_seeds = np.arange(len(matrix['seeds']), dtype=int)
    point = absolute_metric_vector(matrix, all_events, all_seeds)
    components_local = component_rows(matrix['meta'])
    draws = np.empty((BOOTSTRAP_ITERATIONS, 3), dtype=float)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sampled_components = rng.integers(
            0, len(components_local), size=len(components_local)
        )
        sampled_events = np.concatenate([
            components_local[index] for index in sampled_components
        ])
        sampled_seeds = rng.integers(
            0, len(all_seeds), size=len(all_seeds)
        )
        draws[iteration] = absolute_metric_vector(
            matrix, sampled_events, sampled_seeds
        )

    seed_values = []
    for position in all_seeds:
        seed_values.append(
            absolute_metric_vector(matrix, all_events, np.asarray([position]))
        )
    seed_values = np.asarray(seed_values)
    lead_h = (
        int(str(horizon).removesuffix('h'))
        if str(horizon).endswith('h') and '-' not in str(horizon)
        else np.nan
    )
    rows = []
    for position, metric in enumerate(['Bias', 'MAE', 'RMSE']):
        rows.append({
            'experiment': experiment,
            'model_label': EXPERIMENT_LABELS[experiment],
            'horizon': horizon,
            'lead_h': lead_h,
            'metric': metric,
            'estimate': float(point[position]),
            'ci_low': float(np.quantile(draws[:, position], 0.025)),
            'ci_high': float(np.quantile(draws[:, position], 0.975)),
            'seed_sd': (
                float(np.std(seed_values[:, position], ddof=1))
                if len(seed_values) > 1 else np.nan
            ),
            'seed_q025': float(np.quantile(seed_values[:, position], 0.025)),
            'seed_q975': float(np.quantile(seed_values[:, position], 0.975)),
            'n_seeds': len(matrix['seeds']),
            'n_events': len(matrix['meta']),
            'n_components': matrix['meta']['component_id'].nunique(),
            'bootstrap_iterations': BOOTSTRAP_ITERATIONS,
            'estimand': 'mean performance of a single five-fold OOF trained model',
        })
    return rows


bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
curve_rows = []
curve_horizons = [f'{lead}h' for lead in range(6, 73, 6)]
for experiment in PRIMARY_MODELS:
    for horizon in curve_horizons:
        curve_rows.extend(
            absolute_component_seed_bootstrap(experiment, horizon, bootstrap_rng)
        )
curve_bootstrap = pd.DataFrame(curve_rows)
curve_bootstrap.to_csv(
    TABLE_DIR / 'oof_component_seed_bootstrap_performance_curves.csv', index=False
)


def paired_event_frame(model, reference, horizon):
    parts = []
    for seed in active_seeds:
        model_stats = scoped_event_stats(model, int(seed), horizon)
        reference_seed = -1 if reference in DETERMINISTIC_EXPERIMENTS else int(seed)
        reference_stats = scoped_event_stats(reference, reference_seed, horizon)
        left = model_stats[[
            'Pair_ID', 'component_id', 'era', 'reviewer_subgroup',
            'bias', 'mae', 'mse', 'n_values',
        ]].rename(columns={
            'bias': 'bias_model', 'mae': 'mae_model', 'mse': 'mse_model',
            'n_values': 'n_values_model',
        })
        right = reference_stats[[
            'Pair_ID', 'bias', 'mae', 'mse', 'n_values'
        ]].rename(columns={
            'bias': 'bias_reference', 'mae': 'mae_reference',
            'mse': 'mse_reference', 'n_values': 'n_values_reference',
        })
        paired = left.merge(right, on='Pair_ID', how='inner', validate='one_to_one')
        if len(paired) != len(EVENT_ORDER):
            raise AssertionError(
                f'{model} vs {reference}, seed={seed}: incomplete event pairing.'
            )
        if not paired['n_values_model'].equals(paired['n_values_reference']):
            raise AssertionError(
                f'{model} vs {reference}, seed={seed}: forecast cases differ.'
            )
        paired['seed'] = int(seed)
        parts.append(paired)
    return pd.concat(parts, ignore_index=True)


def comparison_matrix(frame):
    seeds = np.asarray(active_seeds, dtype=int)
    meta = (
        frame[['Pair_ID', 'component_id', 'era', 'reviewer_subgroup']]
        .drop_duplicates('Pair_ID').set_index('Pair_ID')
        .reindex(EVENT_ORDER).reset_index()
    )
    arrays = {}
    for column in [
        'bias_model', 'bias_reference', 'mae_model', 'mae_reference',
        'mse_model', 'mse_reference',
    ]:
        pivot = frame.pivot(index='Pair_ID', columns='seed', values=column).reindex(
            index=EVENT_ORDER, columns=seeds
        )
        if pivot.isna().any().any():
            raise AssertionError(f'Incomplete paired matrix for {column}.')
        arrays[column] = pivot.to_numpy(float)
    return {'meta': meta, 'seeds': seeds, **arrays}


EFFECT_NAMES = [
    'DeltaBias', 'DeltaAbsBias', 'DeltaMAE', 'DeltaRMSE',
    'MAERatio', 'RMSERatio',
]


def effect_vector(matrix, event_indices, seed_indices):
    if len(event_indices) == 0 or len(seed_indices) == 0:
        return np.full(len(EFFECT_NAMES), np.nan)
    index = np.ix_(event_indices, seed_indices)
    bias_model = matrix['bias_model'][index].mean(axis=0)
    bias_reference = matrix['bias_reference'][index].mean(axis=0)
    mae_model = matrix['mae_model'][index].mean(axis=0)
    mae_reference = matrix['mae_reference'][index].mean(axis=0)
    rmse_model = np.sqrt(matrix['mse_model'][index].mean(axis=0))
    rmse_reference = np.sqrt(matrix['mse_reference'][index].mean(axis=0))
    if np.any(mae_reference <= 0) or np.any(rmse_reference <= 0):
        raise AssertionError('Reference error must be positive for error ratios.')
    return np.asarray([
        np.mean(bias_model - bias_reference),
        np.mean(np.abs(bias_model) - np.abs(bias_reference)),
        np.mean(mae_model - mae_reference),
        np.mean(rmse_model - rmse_reference),
        np.mean(mae_model / mae_reference),
        np.mean(rmse_model / rmse_reference),
    ])


def summarize_component_seed_bootstrap(
    paired_frame, rng, selector=None, contrast_selector=None,
):
    matrix = comparison_matrix(paired_frame)
    meta = matrix['meta']
    left_mask = (
        np.ones(len(meta), dtype=bool)
        if selector is None else np.asarray(selector(meta), dtype=bool)
    )
    right_mask = (
        None if contrast_selector is None
        else np.asarray(contrast_selector(meta), dtype=bool)
    )
    all_events = np.arange(len(meta), dtype=int)
    all_seeds = np.arange(len(matrix['seeds']), dtype=int)
    if not left_mask.any() or (right_mask is not None and not right_mask.any()):
        return [{
            'metric': metric, 'estimate': np.nan,
            'ci_low': np.nan, 'ci_high': np.nan,
            'n_events_scope': int(left_mask.sum()),
            'bootstrap_iterations': 0,
            'inference_status': 'unavailable_empty_scope',
        } for metric in EFFECT_NAMES]
    point = effect_vector(matrix, all_events[left_mask], all_seeds)
    if right_mask is not None:
        point -= effect_vector(matrix, all_events[right_mask], all_seeds)

    component_values = meta['component_id'].to_numpy()
    left_components = [
        np.flatnonzero((component_values == component) & left_mask)
        for component in pd.unique(component_values[left_mask])
    ]
    draws = np.full((BOOTSTRAP_ITERATIONS, len(EFFECT_NAMES)), np.nan)
    bootstrap_attempts = 0
    if right_mask is None:
        for iteration in range(BOOTSTRAP_ITERATIONS):
            sampled_components = rng.integers(
                0, len(left_components), size=len(left_components)
            )
            sampled_events = np.concatenate([
                left_components[index] for index in sampled_components
            ])
            sampled_seeds = rng.integers(
                0, len(all_seeds), size=len(all_seeds)
            )
            draws[iteration] = effect_vector(
                matrix, sampled_events, sampled_seeds
            )
        bootstrap_attempts = BOOTSTRAP_ITERATIONS
    else:
        # An interaction must use the same connected-component draw on both
        # sides. Rare draws containing no event from one scope are redrawn.
        all_components = component_rows(meta)
        accepted = 0
        maximum_attempts = BOOTSTRAP_ITERATIONS * 100
        while accepted < BOOTSTRAP_ITERATIONS and bootstrap_attempts < maximum_attempts:
            bootstrap_attempts += 1
            sampled_components = rng.integers(
                0, len(all_components), size=len(all_components)
            )
            sampled_events = np.concatenate([
                all_components[index] for index in sampled_components
            ])
            left_events = sampled_events[left_mask[sampled_events]]
            right_events = sampled_events[right_mask[sampled_events]]
            if len(left_events) == 0 or len(right_events) == 0:
                continue
            sampled_seeds = rng.integers(
                0, len(all_seeds), size=len(all_seeds)
            )
            draws[accepted] = (
                effect_vector(matrix, left_events, sampled_seeds)
                - effect_vector(matrix, right_events, sampled_seeds)
            )
            accepted += 1
        if accepted != BOOTSTRAP_ITERATIONS:
            raise AssertionError(
                'Could not obtain enough valid shared-component interaction draws.'
            )

    rows = []
    for position, metric in enumerate(EFFECT_NAMES):
        values = draws[:, position]
        values = values[np.isfinite(values)]
        if len(values) < int(0.95 * BOOTSTRAP_ITERATIONS):
            raise AssertionError(
                'Too many invalid component-bootstrap replicates; scope is too small.'
            )
        rows.append({
            'metric': metric,
            'estimate': float(point[position]),
            'ci_low': float(np.quantile(values, 0.025)),
            'ci_high': float(np.quantile(values, 0.975)),
            'bootstrap_probability_below_zero': float(np.mean(values < 0)),
            'bootstrap_iterations': len(values),
            'bootstrap_attempts': bootstrap_attempts,
            'n_seeds': len(matrix['seeds']),
            'n_events_scope': int(left_mask.sum()),
            'n_components_scope': int(meta.loc[left_mask, 'component_id'].nunique()),
            'n_events_contrast': (
                np.nan if right_mask is None else int(right_mask.sum())
            ),
            'n_components_contrast': (
                np.nan if right_mask is None
                else int(meta.loc[right_mask, 'component_id'].nunique())
            ),
            'inference_status': (
                'confirmatory' if left_mask.sum() >= MIN_REVIEWER_SUBGROUP_EVENTS
                and (right_mask is None or right_mask.sum() >= MIN_REVIEWER_SUBGROUP_EVENTS)
                else 'descriptive_only_small_n'
            ),
            'estimand': (
                'mean single-seed five-fold OOF effect; '
                'shared-SID-component and paired-seed bootstrap'
            ),
        })
    return rows


def per_seed_effects(paired_frame):
    matrix = comparison_matrix(paired_frame)
    event_indices = np.arange(len(matrix['meta']), dtype=int)
    rows = []
    for position, seed in enumerate(matrix['seeds']):
        values = effect_vector(matrix, event_indices, np.asarray([position]))
        for metric, value in zip(EFFECT_NAMES, values):
            rows.append({'seed': int(seed), 'metric': metric, 'estimate': float(value)})
    return rows


def add_context(rows, **context):
    for row in rows:
        row.update(context)
    return rows

# %% [Notebook cell 25]
comparison_specs = []
for experiment in PATH_EXPERIMENTS + ENVIRONMENT_EXPERIMENTS:
    comparison_specs.append(('feature_vs_CTL', experiment, 'CTL'))
for experiment in PATH_EXPERIMENTS:
    for reference in ['PERSISTENCE', 'CLIMATOLOGY']:
        comparison_specs.append(('path_vs_baseline', experiment, reference))

PAIRED_CACHE = {}
primary_rows = []
per_seed_rows = []
for family, model, reference in comparison_specs:
    key = (model, reference, PRIMARY_HORIZON)
    paired = PAIRED_CACHE.setdefault(
        key, paired_event_frame(model, reference, PRIMARY_HORIZON)
    )
    primary_rows.extend(add_context(
        summarize_component_seed_bootstrap(paired, bootstrap_rng),
        family=family, model=model, model_label=EXPERIMENT_LABELS[model],
        reference=reference, reference_label=EXPERIMENT_LABELS[reference],
        scope='all_events_6_72h', horizon=PRIMARY_HORIZON,
    ))
    per_seed_rows.extend(add_context(
        per_seed_effects(paired),
        family=family, model=model, model_label=EXPERIMENT_LABELS[model],
        reference=reference, reference_label=EXPERIMENT_LABELS[reference],
        scope='all_events_6_72h', horizon=PRIMARY_HORIZON,
    ))

primary_paired_effects = pd.DataFrame(primary_rows)
primary_paired_effects.to_csv(
    TABLE_DIR / 'paired_effects_primary_6_72h.csv', index=False
)
paired_effects_per_seed = pd.DataFrame(per_seed_rows)
paired_effects_per_seed.to_csv(
    TABLE_DIR / 'paired_effects_per_seed_6_72h.csv', index=False
)

# 36–72 h is a declared sensitivity analysis, never the primary endpoint.
sensitivity_rows = []
for model in PATH_EXPERIMENTS + ENVIRONMENT_EXPERIMENTS:
    paired = paired_event_frame(model, 'CTL', SENSITIVITY_HORIZON)
    sensitivity_rows.extend(add_context(
        summarize_component_seed_bootstrap(paired, bootstrap_rng),
        family='feature_vs_CTL_sensitivity', model=model,
        model_label=EXPERIMENT_LABELS[model], reference='CTL',
        reference_label=EXPERIMENT_LABELS['CTL'],
        scope='all_events_36_72h', horizon=SENSITIVITY_HORIZON,
    ))
pd.DataFrame(sensitivity_rows).to_csv(
    TABLE_DIR / 'paired_effects_sensitivity_36_72h.csv', index=False
)

# Reviewer subgroup: paired change and subgroup-minus-complement interaction.
subgroup_selector = lambda meta: meta['reviewer_subgroup'].to_numpy(bool)
complement_selector = lambda meta: ~meta['reviewer_subgroup'].to_numpy(bool)
subgroup_rows = []
subgroup_interaction_rows = []
for model in PATH_EXPERIMENTS:
    paired = PAIRED_CACHE[(model, 'CTL', PRIMARY_HORIZON)]
    for scope, selector in [
        ('reviewer_subgroup', subgroup_selector),
        ('complement', complement_selector),
    ]:
        rows = summarize_component_seed_bootstrap(
            paired, bootstrap_rng, selector=selector
        )
        subgroup_rows.extend(add_context(
            rows, model=model, model_label=EXPERIMENT_LABELS[model],
            reference='CTL', scope=scope, horizon=PRIMARY_HORIZON,
        ))
    interaction = summarize_component_seed_bootstrap(
        paired, bootstrap_rng, selector=subgroup_selector,
        contrast_selector=complement_selector,
    )
    subgroup_interaction_rows.extend(add_context(
        interaction, model=model, model_label=EXPERIMENT_LABELS[model],
        reference='CTL', interaction='subgroup_minus_complement',
        horizon=PRIMARY_HORIZON,
    ))

subgroup_effects = pd.DataFrame(subgroup_rows)
subgroup_effects.to_csv(
    TABLE_DIR / 'paired_effects_reviewer_subgroup_6_72h.csv', index=False
)
subgroup_interactions = pd.DataFrame(subgroup_interaction_rows)
subgroup_interactions.to_csv(
    TABLE_DIR / 'paired_interactions_reviewer_subgroup_6_72h.csv', index=False
)

ctl_subgroup_event_stats = event_stats_oof[
    event_stats_oof['experiment'].eq('CTL')
    & event_stats_oof['horizon'].eq(PRIMARY_HORIZON)
    & event_stats_oof['reviewer_subgroup'].eq(True)
    & event_stats_oof['seed'].ge(0)
].copy()
ctl_subgroup_seed_bias = (
    ctl_subgroup_event_stats.groupby('seed', as_index=False)['bias'].mean()
    .rename(columns={'bias': 'CTL_subgroup_Bias'})
)
ctl_subgroup_seed_bias.to_csv(
    TABLE_DIR / 'ctl_reviewer_subgroup_bias_by_oof_seed_6_72h.csv', index=False
)
ctl_subgroup_bias_mean = float(ctl_subgroup_seed_bias['CTL_subgroup_Bias'].mean())
ctl_subgroup_bias_sd = float(
    ctl_subgroup_seed_bias['CTL_subgroup_Bias'].std(ddof=1)
)

# Era-stratified paired effects and model×era interactions address state mixing.
era_rows = []
era_interaction_rows = []
ERA_REFERENCE = ERA_LABELS[0]
for model in PATH_EXPERIMENTS + ENVIRONMENT_EXPERIMENTS:
    paired = PAIRED_CACHE[(model, 'CTL', PRIMARY_HORIZON)]
    for era in ERA_LABELS:
        selector = lambda meta, value=era: meta['era'].eq(value).to_numpy()
        era_rows.extend(add_context(
            summarize_component_seed_bootstrap(
                paired, bootstrap_rng, selector=selector
            ),
            model=model, model_label=EXPERIMENT_LABELS[model], reference='CTL',
            era=era, horizon=PRIMARY_HORIZON,
        ))
    for era in ERA_LABELS[1:]:
        selector = lambda meta, value=era: meta['era'].eq(value).to_numpy()
        reference_selector = (
            lambda meta: meta['era'].eq(ERA_REFERENCE).to_numpy()
        )
        era_interaction_rows.extend(add_context(
            summarize_component_seed_bootstrap(
                paired, bootstrap_rng, selector=selector,
                contrast_selector=reference_selector,
            ),
            model=model, model_label=EXPERIMENT_LABELS[model], reference='CTL',
            era=era, reference_era=ERA_REFERENCE,
            interaction='model_effect_era_minus_reference_era',
            horizon=PRIMARY_HORIZON,
        ))
era_effects = pd.DataFrame(era_rows)
era_effects.to_csv(TABLE_DIR / 'paired_effects_by_era_6_72h.csv', index=False)
era_interactions = pd.DataFrame(era_interaction_rows)
era_interactions.to_csv(
    TABLE_DIR / 'paired_interactions_by_era_6_72h.csv', index=False
)

print('Primary paired-effect rows:', len(primary_paired_effects))
print('Twenty-seed OOF effect rows:', len(paired_effects_per_seed))
print('Reviewer subgroup events:', int(analysis_events['reviewer_subgroup'].sum()))
display(primary_paired_effects[
    primary_paired_effects['metric'].eq('DeltaMAE')
    & primary_paired_effects['family'].eq('feature_vs_CTL')
])

# %% [Notebook cell 26]
# Descriptive seed-ensemble predictions are created only for the old-style
# composite/calibration supplement. They are not used for primary inference.
PREDICTION_KEY = ['window_id', 'lead_h', 'member']
ensemble_parts = []
for experiment in ['PERSISTENCE', 'CLIMATOLOGY'] + active_experiments:
    for fold in active_folds:
        if experiment in DETERMINISTIC_EXPERIMENTS:
            paths = [prediction_file(experiment, fold, -1)]
        else:
            paths = [prediction_file(experiment, fold, seed) for seed in active_seeds]
        if not all(path.is_file() for path in paths):
            if ALLOW_PARTIAL_RESULTS:
                continue
            raise FileNotFoundError(paths[0])
        base = pd.read_csv(paths[0]).sort_values(PREDICTION_KEY).reset_index(drop=True)
        prediction_sum = base['predicted_ms'].to_numpy(float).copy()
        reference_keys = base[PREDICTION_KEY].astype(str).agg('|'.join, axis=1).to_numpy()
        reference_observed = base['observed_ms'].to_numpy(float)
        for path in paths[1:]:
            other = pd.read_csv(path).sort_values(PREDICTION_KEY).reset_index(drop=True)
            other_keys = other[PREDICTION_KEY].astype(str).agg('|'.join, axis=1).to_numpy()
            if not np.array_equal(reference_keys, other_keys):
                raise AssertionError(f'Descriptive seed manifests differ: {path.name}')
            if not np.allclose(reference_observed, other['observed_ms'].to_numpy(float)):
                raise AssertionError(f'Descriptive observations differ: {path.name}')
            prediction_sum += other['predicted_ms'].to_numpy(float)
        base['predicted_ms'] = prediction_sum / len(paths)
        base['seed'] = -1 if experiment in DETERMINISTIC_EXPERIMENTS else -2
        base['seed_count'] = len(paths)
        base['analysis_role'] = 'descriptive_seed_ensemble_only'
        ensemble_parts.append(base)

ensemble_predictions = pd.concat(ensemble_parts, ignore_index=True)
ensemble_predictions.to_csv(
    TABLE_DIR / 'descriptive_seed_ensemble_test_predictions.csv.gz',
    index=False, compression='gzip',
)

calibration_keys = PREDICTION_KEY + ['Pair_ID']
observation_reference = ensemble_predictions[
    ensemble_predictions['experiment'].eq('CTL')
][calibration_keys + ['observed_ms']].drop_duplicates(calibration_keys)
observation_reference['intensity_quantile'] = pd.qcut(
    observation_reference['observed_ms'].rank(method='first'), q=5,
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
)
calibration_rows = []
for experiment in [
    'PERSISTENCE', 'CLIMATOLOGY', 'CTL'
] + PATH_EXPERIMENTS:
    model = ensemble_predictions[ensemble_predictions['experiment'].eq(experiment)]
    merged = model.merge(
        observation_reference[calibration_keys + ['intensity_quantile']],
        on=calibration_keys, how='left', validate='many_to_one',
    )
    for quantile, group in merged.groupby('intensity_quantile', observed=True):
        metrics = event_balanced_metrics(group)
        calibration_rows.append({
            'experiment': experiment,
            'model_label': EXPERIMENT_LABELS[experiment],
            'intensity_quantile': str(quantile),
            'mean_observed_ms': float(group['observed_ms'].mean()),
            'mean_predicted_ms': float(group['predicted_ms'].mean()),
            **metrics,
        })
calibration = pd.DataFrame(calibration_rows)
calibration.to_csv(
    TABLE_DIR / 'descriptive_calibration_by_observed_intensity_quantile.csv',
    index=False,
)

# %% [markdown] [Notebook cell 27]
# ## 12. Main figure
# 
# The main figure prioritizes the evidence requested by the reviewer: explicit baselines, test errors rather than only mean intensity, signed bias, paired changes relative to CTL, the predefined long-lived/open-ocean/northward subgroup, and the 5-fold × 20-seed distribution. Negative ΔMAE means improvement over CTL.

# %% [Notebook cell 28]
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'PERSISTENCE': '#7f7f7f',
    'CLIMATOLOGY': '#b07c2c',
    'CTL': '#111111',
    'CTL_CURVATURE': '#1b9e77',
    'CTL_LATLON': '#d95f02',
    'CTL_U': '#7570b3',
    'CTL_V': '#e7298a',
    'CTL_DISTANCE': '#66a61e',
    'CTL_SST': '#1f78b4',
    'CTL_AIR': '#a6cee3',
    'CTL_VWS': '#fb9a99',
    'CTL_STABILITY': '#fdbf6f',
    'CTL_MLD': '#33a02c',
}


def draw_curve(ax, experiment, metric, with_ci=False, linewidth=1.5):
    data = curve_bootstrap[
        curve_bootstrap['experiment'].eq(experiment)
        & curve_bootstrap['metric'].eq(metric)
    ].sort_values('lead_h')
    if data.empty:
        return
    x = data['lead_h'].to_numpy(float)
    y = data['estimate'].to_numpy(float)
    ax.plot(
        x, y, label=EXPERIMENT_LABELS[experiment], color=COLORS[experiment],
        linewidth=linewidth, marker='o', markersize=2.7,
    )
    if with_ci:
        ax.fill_between(
            x, data['ci_low'].to_numpy(float), data['ci_high'].to_numpy(float),
            color=COLORS[experiment], alpha=0.12, linewidth=0,
        )


fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))

# a: benchmark hierarchy
ax = axes[0, 0]
for experiment in ['PERSISTENCE', 'CLIMATOLOGY', 'CTL']:
    draw_curve(ax, experiment, 'MAE', with_ci=True)
ax.set_title('a  Test MAE and benchmark models')
ax.set_xlabel('Forecast lead (h)')
ax.set_ylabel('MAE (m s$^{-1}$)')
ax.set_xticks([6, 12, 24, 36, 48, 60, 72])
ax.legend(frameon=False)

# b: requested path experiments
ax = axes[0, 1]
draw_curve(ax, 'CTL', 'MAE', with_ci=True, linewidth=2.0)
for experiment in PATH_EXPERIMENTS:
    draw_curve(ax, experiment, 'MAE')
ax.set_title('b  Path-feature MAE')
ax.set_xlabel('Forecast lead (h)')
ax.set_ylabel('MAE (m s$^{-1}$)')
ax.set_xticks([6, 12, 24, 36, 48, 60, 72])
ax.legend(frameon=False, ncol=2)

# c: signed bias
ax = axes[0, 2]
draw_curve(ax, 'CTL', 'Bias', with_ci=True, linewidth=2.0)
for experiment in PATH_EXPERIMENTS:
    draw_curve(ax, experiment, 'Bias')
ax.axhline(0, color='0.5', linewidth=0.8, linestyle='--')
ax.set_title('c  Signed bias (prediction − observation)')
ax.set_xlabel('Forecast lead (h)')
ax.set_ylabel('Bias (m s$^{-1}$)')
ax.set_xticks([6, 12, 24, 36, 48, 60, 72])

# d: primary 6–72 h paired effects, path and environment on the same scale
ax = axes[1, 0]
forest_order = PATH_EXPERIMENTS + ENVIRONMENT_EXPERIMENTS
forest = primary_paired_effects[
    primary_paired_effects['family'].eq('feature_vs_CTL')
    & primary_paired_effects['metric'].eq('DeltaMAE')
    & primary_paired_effects['model'].isin(forest_order)
].set_index('model').reindex(forest_order).reset_index()
y = np.arange(len(forest))
x = forest['estimate'].to_numpy(float)
xerr = np.vstack([
    x - forest['ci_low'].to_numpy(float),
    forest['ci_high'].to_numpy(float) - x,
])
colors = [COLORS[value] for value in forest['model']]
for position in y:
    ax.errorbar(
        x[position], position,
        xerr=xerr[:, position:position + 1],
        fmt='o', color=colors[position], capsize=2.5, markersize=4,
    )
ax.axvline(0, color='0.35', linewidth=0.9, linestyle='--')
ax.set_yticks(y)
ax.set_yticklabels([EXPERIMENT_LABELS[value] for value in forest['model']])
ax.invert_yaxis()
ax.set_xlabel('ΔMAE vs CTL, 6–72 h (m s$^{-1}$)')
ax.set_title('d  Component-cluster paired effects')

# e: direct test of whether path information changes subgroup underestimation
ax = axes[1, 1]
if reviewer_subgroup_available:
    subgroup = subgroup_effects[
        subgroup_effects['scope'].eq('reviewer_subgroup')
        & subgroup_effects['metric'].eq('DeltaBias')
    ].set_index('model').reindex(PATH_EXPERIMENTS).reset_index()
    interaction = subgroup_interactions[
        subgroup_interactions['metric'].eq('DeltaBias')
    ].set_index('model').reindex(PATH_EXPERIMENTS).reset_index()
    y = np.arange(len(PATH_EXPERIMENTS))
    for table, offset, marker, label in [
        (subgroup, -0.12, 'o', 'Subgroup ΔBias'),
        (interaction, 0.12, 's', 'Subgroup − complement'),
    ]:
        estimate = table['estimate'].to_numpy(float)
        error = np.vstack([
            estimate - table['ci_low'].to_numpy(float),
            table['ci_high'].to_numpy(float) - estimate,
        ])
        ax.errorbar(
            estimate, y + offset, xerr=error, fmt=marker, color='#333333',
            markerfacecolor=('white' if marker == 's' else '#333333'),
            capsize=2.2, markersize=3.8, label=label,
        )
    ax.axvline(0, color='0.35', linewidth=0.9, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels([EXPERIMENT_LABELS[value] for value in PATH_EXPERIMENTS])
    ax.invert_yaxis()
    ax.set_xlabel('Paired ΔBias vs CTL, 6–72 h (m s$^{-1}$)')
    status_text = (
        'descriptive, small n' if reviewer_subgroup_n < MIN_REVIEWER_SUBGROUP_EVENTS
        else 'component-bootstrap 95% CI'
    )
    ax.set_title(
        'e  Long-lived/open-ocean/northward\n'
        f'n={reviewer_subgroup_n}; CTL Bias={ctl_subgroup_bias_mean:.2f}±'
        f'{ctl_subgroup_bias_sd:.2f}; {status_text}'
    )
    ax.legend(frameon=False, fontsize=6.5)
else:
    ax.set_axis_off()
    ax.text(0.5, 0.5, 'Open-ocean subgroup unavailable: land diagnostic missing\nor no eligible subgroup/complement', ha='center', va='center', transform=ax.transAxes)

# f: 20 complete five-fold OOF seed effects for each requested path factor
ax = axes[1, 2]
path_run_delta = paired_effects_per_seed[
    paired_effects_per_seed['family'].eq('feature_vs_CTL')
    & paired_effects_per_seed['reference'].eq('CTL')
    & paired_effects_per_seed['metric'].eq('DeltaMAE')
    & paired_effects_per_seed['model'].isin(PATH_EXPERIMENTS)
]
box_values = [
    path_run_delta.loc[
        path_run_delta['model'].eq(experiment), 'estimate'
    ].dropna().to_numpy(float)
    for experiment in PATH_EXPERIMENTS
]
box = ax.boxplot(
    box_values, labels=[
        'Curvature', 'Lat/lon', 'u', 'v', 'Distance'
    ], patch_artist=True, showfliers=False,
)
for patch, experiment in zip(box['boxes'], PATH_EXPERIMENTS):
    patch.set_facecolor(COLORS[experiment])
    patch.set_alpha(0.55)
for position, (values, experiment) in enumerate(zip(box_values, PATH_EXPERIMENTS), start=1):
    jitter = np.random.default_rng(BOOTSTRAP_SEED + position).normal(0, 0.045, len(values))
    ax.scatter(
        position + jitter, values, s=7, alpha=0.28,
        color=COLORS[experiment], linewidths=0,
    )
ax.axhline(0, color='0.35', linewidth=0.9, linestyle='--')
ax.set_ylabel('ΔMAE vs CTL, 6–72 h (m s$^{-1}$)')
ax.set_title(
    f'f  {len(active_seeds)} complete {len(active_folds)}-fold OOF seeds'
)
ax.tick_params(axis='x', rotation=25)

for ax in axes.ravel():
    if ax.axison:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig.suptitle(
    f'Reviewer experiment set: N={len(EVENT_ORDER)} analyzed events; '
    f'{analysis_events.component_id.nunique()} independent SID components',
    y=1.015, fontsize=9.5,
)
fig.tight_layout()
for suffix, kwargs in [('png', {'dpi': 600}), ('pdf', {}), ('tif', {'dpi': 600})]:
    fig.savefig(
        MAIN_FIGURE_DIR / f'Figure_main_reviewer_performance_path.{suffix}',
        bbox_inches='tight', **kwargs,
    )
plt.show()

# %% [markdown] [Notebook cell 29]
# ## 13. Supplementary figures
# 
# The supplement contains the complete environmental curves, the former mean-intensity composite as a descriptive-only display, all 20 complete-OOF seed effects, calibration, cohort/fold audit, training diagnostics, direct baseline comparisons, and era/subgroup sensitivity analyses.
# 
# The calibration and parameter-count panels can diagnose regression toward the mean and capacity differences, but they cannot by themselves distinguish MSE/over-parameterization from a physical mechanism. Unless a separate loss/capacity control is added later, manuscript language must remain predictive rather than causal. Persistence and training-fold climatology are the two non-neural reference forecasts.

# %% [Notebook cell 30]
# S1: all five original environmental experiments.
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
for experiment in ['CTL'] + ENVIRONMENT_EXPERIMENTS:
    draw_curve(axes[0], experiment, 'MAE', with_ci=True)
    draw_curve(axes[1], experiment, 'Bias', with_ci=True)
axes[0].set_title('a  Environmental experiments: MAE')
axes[0].set_ylabel('MAE (m s$^{-1}$)')
axes[1].set_title('b  Environmental experiments: signed bias')
axes[1].set_ylabel('Bias (m s$^{-1}$)')
for ax in axes:
    ax.set_xlabel('Forecast lead (h)')
    ax.set_xticks([6, 12, 24, 36, 48, 60, 72])
    ax.axhline(0, color='0.6', linewidth=0.7, linestyle='--') if ax is axes[1] else None
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S1_environment_error_bias.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S1_environment_error_bias.pdf', bbox_inches='tight')
plt.show()


# S2: old-style mean-intensity composite, descriptive only and not inferential evidence.
composite_models = ['CTL'] + ENVIRONMENT_EXPERIMENTS + PATH_EXPERIMENTS
composite_rows = []
ctl_observation = ensemble_predictions[
    ensemble_predictions['experiment'].eq('CTL')
].drop_duplicates(PREDICTION_KEY + ['Pair_ID'])
for lead_h, group in ctl_observation.groupby('lead_h'):
    event_mean = group.groupby('Pair_ID')['observed_ms'].mean()
    composite_rows.append({
        'experiment': 'OBSERVED', 'model_label': 'Observed',
        'lead_h': int(lead_h), 'mean_intensity_ms': float(event_mean.mean()),
    })
for experiment in composite_models:
    model = ensemble_predictions[ensemble_predictions['experiment'].eq(experiment)]
    for lead_h, group in model.groupby('lead_h'):
        event_mean = group.groupby('Pair_ID')['predicted_ms'].mean()
        composite_rows.append({
            'experiment': experiment,
            'model_label': EXPERIMENT_LABELS[experiment],
            'lead_h': int(lead_h),
            'mean_intensity_ms': float(event_mean.mean()),
        })
composite = pd.DataFrame(composite_rows)
composite.to_csv(TABLE_DIR / 'supplementary_mean_intensity_composite.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
for ax, experiments, title in [
    (axes[0], ['CTL'] + ENVIRONMENT_EXPERIMENTS, 'Environment (descriptive only)'),
    (axes[1], ['CTL'] + PATH_EXPERIMENTS, 'Path (descriptive only)'),
]:
    observed = composite[composite['experiment'].eq('OBSERVED')]
    ax.plot(observed['lead_h'], observed['mean_intensity_ms'], color='black',
            linewidth=2.2, marker='o', markersize=3, label='Observed')
    for experiment in experiments:
        group = composite[composite['experiment'].eq(experiment)].sort_values('lead_h')
        ax.plot(group['lead_h'], group['mean_intensity_ms'],
                color=COLORS[experiment], linewidth=1.3,
                label=EXPERIMENT_LABELS[experiment])
    ax.set_title(title)
    ax.set_xlabel('Forecast lead (h)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
axes[0].set_ylabel('Mean intensity (m s$^{-1}$)')
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S2_mean_intensity_composite.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S2_mean_intensity_composite.pdf', bbox_inches='tight')
plt.show()


# S3: 20 complete five-fold OOF seed effects for every non-CTL neural experiment.
distribution_order = ENVIRONMENT_EXPERIMENTS + PATH_EXPERIMENTS
distribution_values = [
    paired_differences.loc[
        paired_differences['experiment'].eq(experiment), 'MAE_delta_vs_CTL'
    ].dropna().to_numpy(float)
    for experiment in distribution_order
]
fig, ax = plt.subplots(figsize=(10.5, 4.3))
ax.boxplot(distribution_values, labels=[
    EXPERIMENT_LABELS[value].replace('CTL + ', '') for value in distribution_order
], showfliers=False)
ax.axhline(0, color='0.35', linewidth=0.9, linestyle='--')
ax.set_ylabel('ΔMAE vs CTL, 6–72 h (m s$^{-1}$)')
ax.set_title(
    f'{len(active_seeds)} complete {len(active_folds)}-fold OOF seed effects'
)
ax.tick_params(axis='x', rotation=35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S3_all_seed_oof_distributions.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S3_all_seed_oof_distributions.pdf', bbox_inches='tight')
plt.show()


# S4: intensity-quantile calibration and regression-to-the-mean check.
fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8))
calibration_order = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
for experiment in ['PERSISTENCE', 'CLIMATOLOGY', 'CTL'] + PATH_EXPERIMENTS:
    group = calibration[calibration['experiment'].eq(experiment)].copy()
    group['intensity_quantile'] = pd.Categorical(
        group['intensity_quantile'], categories=calibration_order, ordered=True
    )
    group = group.sort_values('intensity_quantile')
    axes[0].plot(
        group['mean_observed_ms'], group['mean_predicted_ms'],
        marker='o', markersize=3, linewidth=1.2,
        color=COLORS[experiment], label=EXPERIMENT_LABELS[experiment],
    )
    axes[1].plot(
        calibration_order, group['Bias'], marker='o', markersize=3,
        linewidth=1.2, color=COLORS[experiment],
        label=EXPERIMENT_LABELS[experiment],
    )
limits = [
    min(calibration['mean_observed_ms'].min(), calibration['mean_predicted_ms'].min()),
    max(calibration['mean_observed_ms'].max(), calibration['mean_predicted_ms'].max()),
]
axes[0].plot(limits, limits, color='0.5', linestyle='--', linewidth=0.8)
axes[0].set_xlabel('Mean observed intensity (m s$^{-1}$)')
axes[0].set_ylabel('Mean predicted intensity (m s$^{-1}$)')
axes[0].set_title('a  Descriptive seed-ensemble calibration')
axes[1].axhline(0, color='0.5', linestyle='--', linewidth=0.8)
axes[1].set_xlabel('Observed-intensity quantile')
axes[1].set_ylabel('Bias (m s$^{-1}$)')
axes[1].set_title('b  Descriptive regression-to-the-mean diagnostic')
axes[0].legend(frameon=False, fontsize=6.8, ncol=2)
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S4_calibration_mean_regression.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S4_calibration_mean_regression.pdf', bbox_inches='tight')
plt.show()


# S5: cohort, missingness and fold audit.
fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
flow_plot = cohort_flow.dropna(subset=['events'])
axes[0, 0].bar(np.arange(len(flow_plot)), flow_plot['events'], color='#4c78a8')
axes[0, 0].set_xticks(np.arange(len(flow_plot)))
axes[0, 0].set_xticklabels(flow_plot['stage'], rotation=25, ha='right')
axes[0, 0].set_ylabel('Events')
axes[0, 0].set_title('a  Cohort flow')

missing_plot = zero_audit.sort_values('missing_or_nonfinite_count', ascending=False)
axes[0, 1].bar(
    np.arange(len(missing_plot)), missing_plot['missing_or_nonfinite_count'],
    color='#e45756',
)
axes[0, 1].set_xticks(np.arange(len(missing_plot)))
axes[0, 1].set_xticklabels(missing_plot['variable'], rotation=70, ha='right', fontsize=6.5)
axes[0, 1].set_ylabel('Missing/non-finite rows')
axes[0, 1].set_title('b  Missingness in 1948–2019')

fold_counts = event_table.groupby('base_fold').agg(
    events=('Pair_ID', 'size'), windows=('n_windows', 'sum')
).reset_index()
axes[1, 0].bar(fold_counts['base_fold'] - 0.17, fold_counts['events'],
               width=0.34, label='Events', color='#72b7b2')
window_scaled = fold_counts['windows'] / fold_counts['windows'].sum() * fold_counts['events'].sum()
axes[1, 0].bar(fold_counts['base_fold'] + 0.17, window_scaled,
               width=0.34, label='Windows (scaled)', color='#f2cf5b')
axes[1, 0].set_xlabel('Base fold')
axes[1, 0].set_ylabel('Count / scaled count')
axes[1, 0].set_title('c  Fold balance')
axes[1, 0].legend(frameon=False)

era_table = pd.crosstab(event_table['base_fold'], event_table['era']).reindex(
    columns=ERA_LABELS, fill_value=0
)
bottom = np.zeros(len(era_table))
for era in ERA_LABELS:
    axes[1, 1].bar(era_table.index, era_table[era], bottom=bottom, label=era)
    bottom += era_table[era].to_numpy(float)
axes[1, 1].set_xlabel('Base fold')
axes[1, 1].set_ylabel('Events')
axes[1, 1].set_title('d  Era composition by fold')
axes[1, 1].legend(frameon=False, fontsize=7)
for ax in axes.ravel():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S5_cohort_missingness_folds.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S5_cohort_missingness_folds.pdf', bbox_inches='tight')
plt.show()


# S6: training diagnostics and model capacity.
manifest = load_run_manifest()
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
successful = manifest[manifest['status'].eq('success')].copy() if not manifest.empty else pd.DataFrame()
if successful.empty:
    axes[0].text(0.5, 0.5, 'No completed neural runs', ha='center', va='center')
    axes[0].set_axis_off()
else:
    values = [
        successful.loc[successful['experiment'].eq(experiment), 'best_epoch']
        .dropna().to_numpy(float)
        for experiment in NEURAL_EXPERIMENTS
    ]
    axes[0].boxplot(values, labels=[EXPERIMENT_LABELS[value] for value in NEURAL_EXPERIMENTS],
                    showfliers=False)
    axes[0].tick_params(axis='x', rotation=70, labelsize=6.5)
    axes[0].set_ylabel('Best epoch')
    axes[0].set_title('a  Early-stopping epochs')

if MODEL_PARAMETER_PATH.is_file():
    parameters = pd.read_csv(MODEL_PARAMETER_PATH)
    parameters = parameters.groupby('experiment', as_index=False)['trainable_parameters'].first()
    parameters = parameters.set_index('experiment').reindex(NEURAL_EXPERIMENTS).reset_index()
    axes[1].bar(np.arange(len(parameters)), parameters['trainable_parameters'], color='#59a14f')
    axes[1].set_xticks(np.arange(len(parameters)))
    axes[1].set_xticklabels(
        [EXPERIMENT_LABELS[value] for value in parameters['experiment']],
        rotation=70, ha='right', fontsize=6.5,
    )
    axes[1].set_ylabel('Trainable parameters')
    axes[1].set_title('b  Model capacity (no zero padding)')
else:
    axes[1].text(0.5, 0.5, 'Parameter audit unavailable', ha='center', va='center')
    axes[1].set_axis_off()
for ax in axes:
    if ax.axison:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S6_training_diagnostics.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S6_training_diagnostics.pdf', bbox_inches='tight')
plt.show()


# S7: direct paired path-model comparisons with all requested baselines.
baseline_order = ['PERSISTENCE', 'CLIMATOLOGY']
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)
for ax, reference in zip(axes, baseline_order):
    data = primary_paired_effects[
        primary_paired_effects['family'].eq('path_vs_baseline')
        & primary_paired_effects['reference'].eq(reference)
        & primary_paired_effects['metric'].eq('DeltaMAE')
    ].set_index('model').reindex(PATH_EXPERIMENTS).reset_index()
    y = np.arange(len(data))
    estimate = data['estimate'].to_numpy(float)
    error = np.vstack([
        estimate - data['ci_low'].to_numpy(float),
        data['ci_high'].to_numpy(float) - estimate,
    ])
    ax.errorbar(estimate, y, xerr=error, fmt='o', color='#333333', capsize=2.5)
    ax.axvline(0, color='0.5', linestyle='--', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([
        EXPERIMENT_LABELS[value].replace('CTL + ', '') for value in PATH_EXPERIMENTS
    ])
    ax.set_xlabel('ΔMAE, 6–72 h (m s$^{-1}$)')
    ax.set_title(f'vs {EXPERIMENT_LABELS[reference]}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].invert_yaxis()
fig.suptitle('Direct component-cluster paired baseline comparisons', y=1.02)
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S7_path_vs_baselines.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S7_path_vs_baselines.pdf', bbox_inches='tight')
plt.show()


# S8: era-stratified effects and model×era interactions.
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
era_plot_models = PATH_EXPERIMENTS + ENVIRONMENT_EXPERIMENTS
for model in era_plot_models:
    data = era_effects[
        era_effects['model'].eq(model) & era_effects['metric'].eq('DeltaMAE')
    ].set_index('era').reindex(ERA_LABELS).reset_index()
    axes[0].plot(
        np.arange(len(ERA_LABELS)), data['estimate'], marker='o', markersize=3,
        linewidth=1.1, color=COLORS[model], label=EXPERIMENT_LABELS[model],
    )
interaction_plot = era_interactions[
    era_interactions['metric'].eq('DeltaMAE')
].copy()
interaction_summary = interaction_plot.groupby('era', as_index=False).agg(
    median_interaction=('estimate', 'median'),
    minimum=('estimate', 'min'), maximum=('estimate', 'max'),
)
x = np.arange(len(ERA_LABELS[1:]))
summary = interaction_summary.set_index('era').reindex(ERA_LABELS[1:]).reset_index()
axes[1].errorbar(
    x, summary['median_interaction'],
    yerr=np.vstack([
        summary['median_interaction'] - summary['minimum'],
        summary['maximum'] - summary['median_interaction'],
    ]), fmt='o', color='#333333', capsize=3,
)
for ax in axes:
    ax.axhline(0, color='0.5', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_xticks(np.arange(len(ERA_LABELS)))
axes[0].set_xticklabels(ERA_LABELS, rotation=25, ha='right')
axes[0].set_ylabel('ΔMAE vs CTL, 6–72 h (m s$^{-1}$)')
axes[0].set_title('a  Era-stratified paired effects')
axes[0].legend(frameon=False, fontsize=6, ncol=2)
axes[1].set_xticks(x)
axes[1].set_xticklabels(ERA_LABELS[1:], rotation=25, ha='right')
axes[1].set_ylabel('Era interaction vs 1948–1969 (m s$^{-1}$)')
axes[1].set_title('b  Median and range across single-factor models')
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S8_era_effects_interactions.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S8_era_effects_interactions.pdf', bbox_inches='tight')
plt.show()


# S9: bias-magnitude results for the reviewer subgroup and interaction.
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), sharey=True)
if reviewer_subgroup_available:
    for ax, data, title in [
        (
            axes[0],
            subgroup_effects[
                subgroup_effects['scope'].eq('reviewer_subgroup')
                & subgroup_effects['metric'].eq('DeltaAbsBias')
            ],
            'Subgroup: Δ|Bias| vs CTL',
        ),
        (
            axes[1],
            subgroup_interactions[
                subgroup_interactions['metric'].eq('DeltaAbsBias')
            ],
            'Subgroup − complement interaction',
        ),
    ]:
        data = data.set_index('model').reindex(PATH_EXPERIMENTS).reset_index()
        y = np.arange(len(data))
        estimate = data['estimate'].to_numpy(float)
        error = np.vstack([
            estimate - data['ci_low'].to_numpy(float),
            data['ci_high'].to_numpy(float) - estimate,
        ])
        ax.errorbar(estimate, y, xerr=error, fmt='o', color='#333333', capsize=2.5)
        ax.axvline(0, color='0.5', linestyle='--', linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([EXPERIMENT_LABELS[value] for value in PATH_EXPERIMENTS])
        ax.set_xlabel('m s$^{-1}$ (negative reduces bias magnitude)')
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(f'Long-lived/open-ocean/northward events: n={reviewer_subgroup_n}', y=1.02)
else:
    for ax in axes:
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'Open-ocean subgroup unavailable: land diagnostic missing\nor no eligible subgroup/complement', ha='center', va='center', transform=ax.transAxes)
fig.tight_layout()
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S9_subgroup_bias_magnitude.png', dpi=600, bbox_inches='tight')
fig.savefig(SUPPLEMENTARY_DIR / 'Figure_S9_subgroup_bias_magnitude.pdf', bbox_inches='tight')
plt.show()

figure_manifest = pd.DataFrame([
    {'figure': 'Figure_main_reviewer_performance_path', 'placement': 'main',
     'role': 'primary errors, bias, paired effects, reviewer subgroup, 20-seed OOF dispersion'},
    {'figure': 'Figure_S1_environment_error_bias', 'placement': 'supplementary',
     'role': 'all five original environmental single-factor experiments'},
    {'figure': 'Figure_S2_mean_intensity_composite', 'placement': 'supplementary',
     'role': 'descriptive-only legacy composite; not inferential evidence'},
    {'figure': 'Figure_S3_all_seed_oof_distributions', 'placement': 'supplementary',
     'role': f'{len(active_seeds)} complete {len(active_folds)}-fold OOF seed effects for all neural experiments'},
    {'figure': 'Figure_S4_calibration_mean_regression', 'placement': 'supplementary',
     'role': 'descriptive regression-to-the-mean diagnostic'},
    {'figure': 'Figure_S5_cohort_missingness_folds', 'placement': 'supplementary',
     'role': '222 to 213 to 155 cohort, missingness, folds and eras'},
    {'figure': 'Figure_S6_training_diagnostics', 'placement': 'supplementary',
     'role': 'early stopping and parameter counts'},
    {'figure': 'Figure_S7_path_vs_baselines', 'placement': 'supplementary',
     'role': 'direct persistence and training-fold climatology comparisons'},
    {'figure': 'Figure_S8_era_effects_interactions', 'placement': 'supplementary',
     'role': 'era-stratified effects and model-by-era interactions'},
    {'figure': 'Figure_S9_subgroup_bias_magnitude', 'placement': 'supplementary',
     'role': 'subgroup bias-magnitude effects and interactions'},
])
figure_manifest.to_csv(TABLE_DIR / 'figure_placement_manifest.csv', index=False)

# %% [markdown] [Notebook cell 31]
# ## 14. Final reviewer-facing assertions and output inventory
# 
# This cell fails loudly if the cohort, experiments, folds, run counts or figures do not match the preregistered design. Run it before using any values in the manuscript.

# %% [Notebook cell 32]
EXPECTED_BTC_EXPERIMENTS = {
    'CTL',
    'CTL_SST', 'CTL_AIR', 'CTL_VWS', 'CTL_STABILITY', 'CTL_MLD',
    'CTL_CURVATURE', 'CTL_LATLON', 'CTL_U', 'CTL_V', 'CTL_DISTANCE',
}
if set(BTC_EXPERIMENTS) != EXPECTED_BTC_EXPERIMENTS:
    raise AssertionError(
        f'Experiment set changed. Found: {sorted(BTC_EXPERIMENTS)}'
    )
if FEATURE_NAMES['CTL'] != ['intensity_m1', 'intensity_m2']:
    raise AssertionError('CTL contains something other than the two intensity histories.')
if len(NEURAL_EXPERIMENTS) != 11:
    raise AssertionError('Expected exactly 11 BTC neural experiments.')
if expected_neural_runs != 1100:
    raise AssertionError(f'Expected 1,100 full neural runs, found {expected_neural_runs}.')

for experiment, values in X_BY_EXPERIMENT.items():
    if not np.isfinite(values).all():
        raise AssertionError(f'{experiment} contains NaN or Inf.')
    if values.shape[0] != EXPECTED_CURRENT_WINDOWS or values.shape[1] != LOOKBACK:
        raise AssertionError(f'{experiment} does not use the common 2,171 windows.')
if not np.isfinite(Y).all() or Y.shape != (EXPECTED_CURRENT_WINDOWS, 12, 2):
    raise AssertionError('Target array is incomplete or mis-shaped.')

for text_value in (
    list(EXPERIMENT_LABELS.values())
    + [value for names in FEATURE_NAMES.values() for value in names]
):
    if re.search(r'\b(strong|weak)\b', str(text_value), flags=re.IGNORECASE):
        raise AssertionError(f'Forbidden strong/weak member label: {text_value}')

if raw['Pair_ID'].nunique() != 222:
    raise AssertionError('The source matched cohort is not the fixed 222-event cohort.')
if period['Pair_ID'].nunique() != 213:
    raise AssertionError('The 1948–2019 event count is not 213.')
if WINDOW_META['Pair_ID'].nunique() != 155:
    raise AssertionError('The common eligible modeling cohort is not 155 events.')
if complete_candidate_window_count != 2179:
    raise AssertionError('The input/target-complete candidate count is not 2,179.')
if len(WINDOW_META) != 2171:
    raise AssertionError('The common eligible window count is not 2,171.')
if partition_overlap_audit[['pair_overlap_count', 'sid_overlap_count']].to_numpy().any():
    raise AssertionError('Pair or SID leakage remains in the fold audit.')
if REQUIRE_REVIEWER_SUBGROUP and not land_distance_available:
    raise AssertionError('The required open-ocean land-distance diagnostic is unavailable.')
if REQUIRE_REVIEWER_SUBGROUP and reviewer_subgroup_n == 0:
    raise AssertionError('The required reviewer subgroup is empty.')
if any(column.lower().startswith('p_') for column in primary_paired_effects.columns):
    raise AssertionError('Invalid bootstrap p-values must not be reported.')

# Persistence must be exactly the last observed input intensity at every lead.
for fold in active_folds:
    path = prediction_file('PERSISTENCE', fold, -1)
    if path.is_file():
        persistence = pd.read_csv(path)
        if not np.allclose(
            persistence['predicted_ms'], persistence['initial_intensity_ms']
        ):
            raise AssertionError(f'Persistence definition failed in fold {fold}.')

if not SMOKE_TEST and not ALLOW_PARTIAL_RESULTS:
    manifest = load_run_manifest()
    current_success = manifest[
        manifest['config_hash'].astype(str).eq(config_hash)
        & manifest['status'].eq('success')
    ].drop_duplicates(['experiment', 'fold', 'seed'], keep='last')
    if len(current_success) != expected_neural_runs:
        raise AssertionError(
            f'Run manifest has {len(current_success)} successful neural runs; '
            f'expected {expected_neural_runs}.'
        )
    for experiment in NEURAL_EXPERIMENTS:
        experiment_files = [
            prediction_file(experiment, fold, seed)
            for fold in range(FOLD_COUNT) for seed in EXPERIMENT_SEEDS
        ]
        if sum(path.is_file() for path in experiment_files) != 100:
            raise AssertionError(f'{experiment} does not have 5 × 20 prediction files.')
    if int(prediction_inventory['exists'].sum()) != 1110:
        raise AssertionError(
            'Expected 1,100 neural prediction files plus 10 deterministic baseline files.'
        )
    oof_seed_counts = seed_oof_metrics[
        seed_oof_metrics['horizon'].eq(PRIMARY_HORIZON)
        & seed_oof_metrics['seed'].ge(0)
    ].groupby('experiment')['seed'].nunique()
    if not oof_seed_counts.reindex(NEURAL_EXPERIMENTS).eq(20).all():
        raise AssertionError('At least one neural experiment lacks 20 complete OOF seeds.')

required_figures = [
    MAIN_FIGURE_DIR / 'Figure_main_reviewer_performance_path.png',
    MAIN_FIGURE_DIR / 'Figure_main_reviewer_performance_path.pdf',
    SUPPLEMENTARY_DIR / 'Figure_S1_environment_error_bias.png',
    SUPPLEMENTARY_DIR / 'Figure_S2_mean_intensity_composite.png',
    SUPPLEMENTARY_DIR / 'Figure_S3_all_seed_oof_distributions.png',
    SUPPLEMENTARY_DIR / 'Figure_S4_calibration_mean_regression.png',
    SUPPLEMENTARY_DIR / 'Figure_S5_cohort_missingness_folds.png',
    SUPPLEMENTARY_DIR / 'Figure_S6_training_diagnostics.png',
    SUPPLEMENTARY_DIR / 'Figure_S7_path_vs_baselines.png',
    SUPPLEMENTARY_DIR / 'Figure_S8_era_effects_interactions.png',
    SUPPLEMENTARY_DIR / 'Figure_S9_subgroup_bias_magnitude.png',
]
missing_figures = [path for path in required_figures if not path.is_file()]
if missing_figures:
    raise FileNotFoundError(f'Required figure files are missing: {missing_figures}')

required_tables = [
    TABLE_DIR / 'metrics_oof_by_seed.csv',
    TABLE_DIR / 'paired_effects_primary_6_72h.csv',
    TABLE_DIR / 'paired_effects_per_seed_6_72h.csv',
    TABLE_DIR / 'paired_effects_reviewer_subgroup_6_72h.csv',
    TABLE_DIR / 'paired_interactions_reviewer_subgroup_6_72h.csv',
    TABLE_DIR / 'paired_effects_by_era_6_72h.csv',
    TABLE_DIR / 'paired_interactions_by_era_6_72h.csv',
    TABLE_DIR / 'figure_placement_manifest.csv',
]
missing_tables = [path for path in required_tables if not path.is_file()]
if missing_tables:
    raise FileNotFoundError(f'Required reviewer tables are missing: {missing_tables}')

output_inventory = []
for path in sorted(RUN_DIR.rglob('*')):
    if path.is_file():
        output_inventory.append({
            'relative_path': str(path.relative_to(RUN_DIR)),
            'size_bytes': path.stat().st_size,
        })
output_inventory = pd.DataFrame(output_inventory)
output_inventory.to_csv(RUN_DIR / 'output_inventory.csv', index=False)

completion_summary = pd.DataFrame([
    {'item': 'source matched events', 'value': 222},
    {'item': 'events in 1948-2019', 'value': 213},
    {'item': 'eligible modeling events', 'value': 155},
    {'item': 'input/target-complete candidate windows', 'value': 2179},
    {'item': 'common windows', 'value': 2171},
    {'item': 'independent shared-SID components', 'value': analysis_events['component_id'].nunique()},
    {'item': 'reviewer subgroup events', 'value': reviewer_subgroup_n},
    {'item': 'training mode', 'value': 'smoke' if SMOKE_TEST else 'formal'},
    {'item': 'base folds executed', 'value': len(active_folds)},
    {'item': 'neural seeds executed', 'value': len(active_seeds)},
    {'item': 'neural experiments', 'value': 11},
    {'item': 'neural fits requested this run',
     'value': len(active_experiments) * len(active_folds) * len(active_seeds)},
    {'item': 'formal neural fits required', 'value': 1100},
    {'item': 'primary inference horizon', 'value': PRIMARY_HORIZON},
    {'item': 'missing-value imputation used', 'value': False},
    {'item': 'zero padding used', 'value': False},
    {'item': 'member strong/weak ordering used', 'value': False},
])
completion_summary.to_csv(RUN_DIR / 'analysis_completion_summary.csv', index=False)

if SMOKE_TEST:
    print('SMOKE TEST ONLY: pipeline checks passed; do not use these outputs in the manuscript.')
else:
    print('All formal reviewer-facing assertions passed.')
print('Run directory:', RUN_DIR)
print('Main figure directory:', MAIN_FIGURE_DIR)
print('Supplementary figure directory:', SUPPLEMENTARY_DIR)
display(completion_summary)


