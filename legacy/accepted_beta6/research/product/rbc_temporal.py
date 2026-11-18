"""Input/interpretation adapter only; frozen temporal mathematics are vendored unchanged."""
from dataclasses import asdict, dataclass
import csv
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import pandas as pd

INTERFACE_VERSION = '0.1.0-beta.1'
MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_POINTS = 600
MAX_COLUMNS = 128
MAX_SLOPE_PAIRS = 2_000_000
VENDOR = Path(__file__).resolve().parents[1] / 'temporal_vendor/temporal.py'
EXPECTED_SHA256 = 'e9d7ce45e7427508e1e3bdefb851ab1044cf87c4ae2bb5cbd46adc544343953e'


def frozen_engine():
    if hashlib.sha256(VENDOR.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Frozen temporal source integrity failed. Analysis stopped.')
    name = '_microstructai_frozen_temporal_010_dev1'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, VENDOR)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class TimingContext:
    time_documented: bool = False
    event_known: bool = False
    event_offset_s: float = 0.0
    initial_state_appropriate: bool = False
    evidence: str = ''
    recording_id: str = ''
    donor_id: str = ''
    upstream_version: str = ''
    calibration: str = ''


def parse_csv(payload):
    if not payload or len(payload) > MAX_CSV_BYTES:
        raise ValueError('Temporal CSV must be nonempty and at most 2 MiB.')
    try:
        text = payload.decode('utf-8-sig')
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        if not header or len(header) > MAX_COLUMNS or any(not h.strip() for h in header) or len(set(header)) != len(header):
            raise ValueError('Require unique, nonempty column names; maximum 128 columns.')
        records = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError('Every CSV row must match the header width; no index inference or row dropping is allowed.')
            records.append(row)
            if len(records) > MAX_POINTS:
                raise ValueError('Select one recording with 2–600 observations. No rows were dropped or resampled.')
        frame = pd.DataFrame(records, columns=header)
    except (UnicodeError, csv.Error, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('Cannot read a UTF-8 comma-separated temporal table.') from exc
    if len(frame) < 2 or len(frame) > MAX_POINTS:
        raise ValueError('Select one recording with 2–600 observations. No rows were dropped or resampled.')
    if list(frame.columns) != header:
        raise ValueError('Unexpected CSV layout. Export a rectangular table with one header row.')
    return frame


def numeric(series, allow_missing=True):
    missing = {'', 'na', 'nan', 'null', 'none', 'n/a'}
    result = []
    for v in series.tolist():
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip().lower() in missing:
            if not allow_missing:
                raise ValueError('Timestamps must not contain missing values.')
            result.append(None)
            continue
        if isinstance(v, bool):
            raise ValueError('Boolean measurements are not scalar descriptor values.')
        try:
            value = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError('Selected measurement/time column contains nonnumeric text.') from exc
        if not math.isfinite(value):
            raise ValueError('Infinite values are unsupported; missing values must be explicit NA/empty.')
        result.append(value)
    return result


def qc_values(series):
    values = []
    for v in series.tolist():
        s = str(v).strip().lower()
        if s in ('true', '1', 'yes'): values.append(True)
        elif s in ('false', '0', 'no'): values.append(False)
        else: raise ValueError('QC-valid column must contain only true/false, 1/0 or yes/no; unknown is not silently accepted.')
    return values


def resource_check(times):
    # A conservative pair-count guard bounds the unchanged Theil-Sen routine.
    # No scientific parameter is changed and no observation is subsampled.
    if len(times) > MAX_POINTS or len(times) < 2:
        raise ValueError('Temporal hosting limit: 2–600 observations per curve.')
    if not all(math.isfinite(t) for t in times) or not math.isfinite(times[-1]-times[0]):
        raise ValueError('Time axis contains overflow/nonfinite support.')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Times must be strictly increasing in original order; do not sort or deduplicate silently.')
    total = times[-1]-times[0]
    windows = set()
    if total >= 5:
        for t in times:
            start = min(max(t-times[0]-2.5, 0), total-5)
            windows.add((start, start+5))
    cost = len(times)*(len(times)-1)//2
    elapsed = [t-times[0] for t in times]
    for a,b in windows:
        n = sum(a <= t <= b for t in elapsed)
        cost += n*(n-1)//2
    if cost > MAX_SLOPE_PAIRS:
        raise ValueError('Curve exceeds the hosted temporal computation budget. Use the preserved local workflow; no thinning or window changes were applied.')


def analyze_curve(frame, value_column, *, time_column=None, fps=None, qc_column=None,
                  context=None, units='a.u.', curve_kind='structure', source_sha256=''):
    context = context or TimingContext()
    if curve_kind not in ('structure', 'composite', 'optical'):
        raise ValueError('Choose structural, composite or optical interpretation.')
    if value_column not in frame:
        raise ValueError('Descriptor column not present.')
    if time_column is not None and fps is not None:
        raise ValueError('Choose timestamps OR frame rate, not both.')
    raw = numeric(frame[value_column])
    times = numeric(frame[time_column], False) if time_column else None
    if fps is not None and (isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0):
        raise ValueError('Frame rate must be positive and finite.')
    support_times = times if times is not None else [i/fps if fps else float(i) for i in range(len(raw))]
    resource_check(support_times)
    valid = qc_values(frame[qc_column]) if qc_column else [True]*len(raw)
    used = [v if ok else None for v,ok in zip(raw, valid)]
    engine = frozen_engine()
    result = engine.analyze_temporal_curve(times, used, descriptor_name=value_column,
                                          units=units, fps=fps, time_unit='s')
    reasons = []
    if result['time_unit'] != 's' or not context.time_documented:
        reasons.append('DOCUMENTED_PHYSICAL_TIME_AXIS_REQUIRED')
    if not context.event_known:
        reasons.append('EXPERIMENTAL_T0_UNKNOWN')
    if not context.initial_state_appropriate:
        reasons.append('PRE_CHANGE_OR_JUSTIFIED_INITIAL_BASELINE_NOT_CONFIRMED')
    if not context.evidence.strip(): reasons.append('TIMING_EVIDENCE_NOT_RECORDED')
    if not math.isfinite(context.event_offset_s) or not 0 <= context.event_offset_s <= support_times[-1]-support_times[0]:
        reasons.append('EVENT_OFFSET_OUTSIDE_OBSERVED_RECORD')
    if result['metrics']['baseline']['value'] is None:
        reasons.append('FROZEN_BASELINE_NOT_EVALUABLE')
    if result['plateau_status'] != 'PLATEAU_VALID':
        reasons.append('FROZEN_PLATEAU_NOT_ESTABLISHED')
    timing = {}
    for key in ('t10','t50','t90','time_to_plateau'):
        original = result['metrics'][key]
        codes = list(dict.fromkeys(reasons + original['reason_codes']))
        if context.event_known and original['value'] is not None and original['value'] < context.event_offset_s:
            codes.append('CROSSING_PRECEDES_DOCUMENTED_EVENT')
        # Any engine warning remains visible and is not promoted to qualified timing.
        qualified = not codes and original['status'] == 'OK' and original['value'] is not None
        timing[key] = {'value': original['value'] if qualified else None,
                       'units': result['time_unit'], 'status': 'QUALIFIED_STRUCTURAL_TIMING' if qualified else 'NOT_EVALUABLE',
                       'reason_codes': codes or ([] if qualified else ['COMPUTATIONAL_CROSSING_NOT_EVALUABLE']),
                       'event_relative_s': original['value']-context.event_offset_s if qualified else None,
                       'support': original['details']}
    return {'interface_version': INTERFACE_VERSION, 'engine_sha256': EXPECTED_SHA256,
            'descriptor': value_column, 'curve_kind': curve_kind, 'units': units,
            'source_sha256': source_sha256, 'context': asdict(context),
            'structural_time_course_available': any(v is not None for v in used),
            'kinetic_timing_qualified': all(timing[k]['value'] is not None for k in ('t10','t50','t90')),
            'timing_interpretation': 'Metadata-supported, recording-relative structural timing only; NOT independently validated biological aggregation timing.',
            'qualified_timing': timing, 'recording_relative_computational': result,
            'raw_values': raw, 'analysis_values': used, 'qc_valid': valid,
            'qc_column': qc_column, 'qc_assessment': 'Supplied explicit validity flag' if qc_column else 'QC UNASSESSED: scalar availability is not image/segmentation validation',
            'row_positions_preserved': True, 'sorting_resampling_smoothing': 'NONE',
            'time_axis': support_times,
            'caution': 'Aggregation-related structural composite. Interpretation depends on experimental context.' if curve_kind=='composite' else
                       'MicroStructAI optical curve; not RheoScan, LORRCA or syllectogram half-time.' if curve_kind=='optical' else
                       'Structural descriptors are not automatically validated biological biomarkers.'}


def synthetic_curve():
    # Explicit first-party software fixture: no external/public data consumed.
    return pd.DataFrame({'time_s': list(range(61)),
                         'gap_area_fraction': [0.1 if i <= 5 else 0.1+0.4*min((i-5)/25,1) for i in range(61)],
                         'valid': ['true']*61})
