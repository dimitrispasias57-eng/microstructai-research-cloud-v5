"""Independent synthetic-development temporal component. No application imports."""
from dataclasses import asdict, dataclass
import math
import statistics as st

VERSION = '0.1.0-dev.1'


@dataclass(frozen=True)
class TemporalConfig:
    baseline_mode: str = 'initial_window_median'
    baseline_window: float = 2.0
    initial_slope_window: float = 5.0
    slope_window: float = 5.0
    tail_min_window: float = 10.0
    tail_fraction: float = 0.20
    min_baseline_samples: int = 2
    min_regression_samples: int = 3
    min_tail_samples: int = 5
    min_span_fraction: float = 0.80
    max_gap_factor: float = 3.0
    absolute_range_floor: float = 1e-9
    relative_range_floor: float = 1e-8
    drift_valid: float = 0.05
    drift_uncertain: float = 0.15
    spread_valid: float = 0.10
    spread_uncertain: float = 0.30
    plateau_band_fraction: float = 0.05
    plateau_noise_multiple: float = 3.0
    reverse_variation_limit: float = 0.20
    outlier_noise_multiple: float = 6.0
    outlier_range_fraction: float = 0.25
    endpoint_amplitude_fraction: float = 0.20
    coarse_crossing_interval: float = 2.0
    min_transition_intervals: float = 4.0


METRIC_NAMES = (
    'baseline', 'final_window_value', 'final_window_delta', 'plateau', 'delta',
    't10', 't50', 't90', 'initial_slope', 'maximum_slope', 'time_to_plateau',
    'raw_auc', 'observed_segment_auc', 'integrated_duration', 'baseline_adjusted_auc',
    'normalized_auc_time', 'normalized_auc_fraction', 'reverse_variation_fraction',
    'signed_monotonicity',
)


def quantile(values, q):
    a = sorted(values)
    h = (len(a) - 1) * q
    lo = int(h)
    return a[lo] + (h - lo) * (a[min(lo + 1, len(a) - 1)] - a[lo])


def theil_sen(t, y):
    return st.median([(y[j] - y[i]) / (t[j] - t[i])
                      for i in range(len(t)) for j in range(i + 1, len(t))])


def _finite(v):
    try:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    except OverflowError:
        return False


def _json_safe_tree(value):
    """Keep diagnostic exports valid even for invalid config or overflow inputs."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_tree(v) for v in value]
    return value


def _has_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_nonfinite(v) for v in value)
    return False


def _sanitize(v):
    if _finite(v):
        return float(v)
    return None


def metric(value=None, units='', reason=None, method='', details=None, warnings=()):
    reasons = list(dict.fromkeys(([reason] if reason else []) + list(warnings)))
    if value is not None and not math.isfinite(value):
        value, reasons = None, reasons + ['NUMERICAL_OVERFLOW']
    return {'value': value, 'units': units,
            'status': 'NOT_EVALUABLE' if value is None else ('WARNING' if reasons else 'OK'),
            'reason_codes': reasons, 'method': method, 'details': details or {}}


def _validate_config(c):
    if not isinstance(c, TemporalConfig):
        return False
    if c.baseline_mode not in ('first_sample', 'initial_window_median'):
        return False
    for key, value in asdict(c).items():
        if key != 'baseline_mode' and (not _finite(value) or value <= 0):
            return False
    if not 0 < c.tail_fraction <= 0.5 or not 0 < c.min_span_fraction <= 1:
        return False
    if c.max_gap_factor <= 1 or not 0 < c.reverse_variation_limit < 1:
        return False
    if not c.drift_valid < c.drift_uncertain or not c.spread_valid < c.spread_uncertain:
        return False
    for key, minimum in [('min_baseline_samples', 2), ('min_regression_samples', 3), ('min_tail_samples', 5)]:
        v = getattr(c, key)
        if not isinstance(v, int) or v < minimum:
            return False
    return True


def analyze_temporal_curve(times, values, descriptor_name='descriptor', units='a.u.', *,
                           fps=None, time_unit='s', config=None,
                           expected_direction='UNKNOWN_DOMAIN_DEPENDENT'):
    """Return recording-relative metrics with raw values, support and explicit NA reasons.

    times=None/fps=None means frames. Timestamps are elapsed/absolute numeric values
    in the explicitly declared unit; outputs are elapsed from the first timestamp.
    """
    c = config if config is not None else TemporalConfig()
    result = {'engine_version': VERSION, 'descriptor_name': str(descriptor_name),
              'descriptor_units': str(units), 'time_unit': time_unit,
              'expected_direction': expected_direction, 'observed_direction': 'UNKNOWN',
              'config': asdict(c) if isinstance(c, TemporalConfig) else None,
              'plateau_status': 'PLATEAU_NOT_ESTABLISHED', 'qc': [],
              'recording': {}, 'raw': {}, 'derived': {'progress': [], 'local_slopes': []},
              'metrics': {k: metric(reason='NOT_EVALUATED') for k in METRIC_NAMES},
              'provenance': {'smoothing': 'NONE', 'outlier_removal': 'NONE',
                             'crossings': 'raw linear interpolation', 'auc': 'raw trapezoids',
                             'slopes': 'local Theil-Sen regression; raw retained'}}
    def fail(code):
        result['qc'] = [code]
        for key in METRIC_NAMES:
            result['metrics'][key] = metric(reason=code)
        return _json_safe_tree(result)
    if not _validate_config(c):
        return fail('CONFIG_INVALID')
    if expected_direction not in ('INCREASING', 'DECREASING', 'UNKNOWN_DOMAIN_DEPENDENT'):
        return fail('DIRECTION_METADATA_INVALID')
    try:
        vals = list(values)
    except (TypeError, ValueError):
        return fail('VALUES_INVALID')
    # Unsupported nonnumeric strings are not silently parsed as measurements.
    if any(v is not None and not isinstance(v, (int, float)) for v in vals) or any(isinstance(v, bool) for v in vals):
        return fail('VALUES_INVALID')
    x = [_sanitize(v) for v in vals]
    result['raw'] = {'values': x.copy(), 'input_value_representations': [repr(v) for v in vals]}
    n = len(x)
    if n < 2:
        return fail('INSUFFICIENT_FRAMES')
    if times is not None and fps is not None:
        return fail('TIME_VECTOR_INVALID')
    if times is None:
        if fps is None:
            t, time_unit, time_source = list(map(float, range(n))), 'frames', 'FRAME_INDEX_ONLY'
        elif _finite(fps) and fps > 0:
            t, time_unit, time_source = [i / fps for i in range(n)], 's', 'DECLARED_FPS'
        else:
            return fail('TIME_VECTOR_INVALID')
    else:
        try:
            t = list(times)
        except TypeError:
            return fail('TIME_VECTOR_INVALID')
        time_source = 'EXPLICIT_TIMESTAMPS'
        if time_unit not in ('s', 'frames'):
            return fail('TIME_VECTOR_INVALID')
    result['time_unit'] = time_unit
    result['raw']['timestamps'] = [_sanitize(v) for v in t]
    if len(t) != n or not all(_finite(v) for v in t):
        return fail('TIME_VECTOR_INVALID')
    t = [float(v) for v in t]
    if any(b <= a for a, b in zip(t, t[1:])):
        return fail('TIME_VECTOR_INVALID')
    u = [v - t[0] for v in t]
    intervals = [b - a for a, b in zip(u, u[1:])]
    if not all(math.isfinite(v) for v in u + intervals):
        return fail('TIME_VECTOR_INVALID')
    T, dt = u[-1], st.median(intervals)
    missing = [i for i, v in enumerate(x) if v is None]
    long_gaps = [i for i, d in enumerate(intervals) if d > c.max_gap_factor * dt]
    qc = result['qc']
    if missing:
        qc.append('MISSING_VALUES')
    if long_gaps:
        qc.append('TIME_GAPS_PRESENT')
    finite = [i for i in range(n) if x[i] is not None]
    result['recording'] = {'duration': T, 'number_of_frames': n, 'finite_samples': len(finite),
                           'median_interval': dt, 'minimum_interval': min(intervals),
                           'maximum_interval': max(intervals), 'time_source': time_source,
                           'time_origin': t[0], 'missing_indices': missing, 'long_gap_after_indices': long_gaps}
    result['derived']['elapsed_times'] = u
    if len(finite) < 2:
        return fail('INSUFFICIENT_FRAMES')
    floor = max(c.absolute_range_floor, c.relative_range_floor * max(abs(x[i]) for i in finite))
    result['derived']['dynamic_range_floor'] = floor
    valid_edge = [x[i] is not None and x[i + 1] is not None and i not in long_gaps for i in range(n - 1)]
    m = result['metrics']
    def supported_window(lo, hi, minimum):
        ids = [i for i in range(n) if lo - 1e-12 <= u[i] <= hi + 1e-12]
        ok = (len(ids) >= minimum and all(x[i] is not None for i in ids)
              and u[ids[-1]] - u[ids[0]] >= c.min_span_fraction * (hi - lo) - 1e-12
              and all(valid_edge[i] for i in ids[:-1]))
        return ids, ok
    def support_details(ids):
        return {'indices': ids, 'count': len(ids), 'elapsed_times': [u[i] for i in ids]}
    B = None
    if c.baseline_mode == 'first_sample':
        if x[0] is not None:
            B = x[0]
        m['baseline'] = metric(B, units, None if B is not None else 'BASELINE_NOT_ESTABLISHED',
                               'first actual sample', support_details([0]))
    else:
        ids, ok = supported_window(0, c.baseline_window, c.min_baseline_samples)
        if T >= c.baseline_window and ok:
            B = st.median(x[i] for i in ids)
        m['baseline'] = metric(B, units, None if B is not None else 'BASELINE_NOT_ESTABLISHED',
                               'median of fixed initial time window', support_details(ids))
    if B is None:
        qc.append('BASELINE_NOT_ESTABLISHED')
    residuals, raw_residuals = [], {}
    for i in range(1, n - 1):
        if valid_edge[i - 1] and valid_edge[i]:
            w = (u[i] - u[i - 1]) / (u[i + 1] - u[i - 1])
            r = x[i] - ((1 - w) * x[i - 1] + w * x[i + 1])
            raw_residuals[i] = r
            residuals.append(r / math.sqrt(1 + (1 - w)**2 + w**2))
    sigma = 0.0
    if len(residuals) >= 3:
        med = st.median(residuals)
        sigma = 1.4826 * st.median(abs(v - med) for v in residuals)
    else:
        qc.append('NOISE_ESTIMATE_UNAVAILABLE')
    result['derived']['noise_sigma_diagnostic'] = sigma
    spread_all = quantile([x[i] for i in finite], .9) - quantile([x[i] for i in finite], .1)
    threshold = max(c.outlier_noise_multiple * sigma, c.outlier_range_fraction * spread_all, floor)
    outliers = [i for i, r in raw_residuals.items() if abs(r) > threshold]
    if outliers:
        qc.append('LARGE_OUTLIER')
    result['derived']['outlier_indices'] = outliers
    W = max(c.tail_min_window, c.tail_fraction * T)
    tail_ids, tail_ok = supported_window(T - W, T, c.min_tail_samples)
    tail_finite = [i for i in tail_ids if x[i] is not None]
    C = st.median(x[i] for i in tail_finite) if tail_finite else None
    m['final_window_value'] = metric(C, units, None if C is not None else 'FINAL_WINDOW_EMPTY',
                                    'candidate tail median; not necessarily plateau', support_details(tail_finite))
    Dc = C - B if C is not None and B is not None else None
    m['final_window_delta'] = metric(Dc, units, None if Dc is not None else 'BASELINE_NOT_ESTABLISHED', 'candidate C-B')
    s = 0
    if Dc is not None and abs(Dc) > floor:
        s = 1 if Dc > 0 else -1
        result['observed_direction'] = 'INCREASING' if s > 0 else 'DECREASING'
    elif Dc is not None:
        qc.append('LOW_DYNAMIC_RANGE')
    if s and expected_direction != 'UNKNOWN_DOMAIN_DEPENDENT' and expected_direction != result['observed_direction']:
        qc.append('EXPECTED_DIRECTION_MISMATCH')
    pd = {'window_length': W, 'indices': tail_ids, 'candidate': C}
    if T < 2 * W:
        qc.append('INSUFFICIENT_DURATION')
    if not tail_ok:
        qc.append('INSUFFICIENT_TAIL_SUPPORT')
    if s and tail_ok and T >= 2 * W:
        ty, tt = [x[i] for i in tail_ids], [u[i] for i in tail_ids]
        drift = abs(theil_sen(tt, ty)) * W / abs(Dc)
        spread = (quantile(ty, .9) - quantile(ty, .1)) / abs(Dc)
        artifact = abs(ty[-1] - st.median(ty[-4:-1])) > max(c.outlier_noise_multiple * sigma, c.endpoint_amplitude_fraction * abs(Dc), floor)
        signs = [1 if b > a else -1 for a, b in zip(ty, ty[1:]) if b != a]
        reversals = sum(a != b for a, b in zip(signs, signs[1:]))
        pd.update({'relative_drift': drift, 'relative_spread': spread, 'sign_reversals': reversals, 'endpoint_artifact': artifact})
        if drift > c.drift_valid:
            qc.append('FINAL_WINDOW_DRIFT')
        if artifact:
            qc.append('FINAL_ARTIFACT')
        if reversals >= 4 and spread > .05:
            qc.append('FINAL_WINDOW_OSCILLATION')
        if not artifact and drift <= c.drift_uncertain and spread <= c.spread_uncertain:
            result['plateau_status'] = ('PLATEAU_VALID' if drift <= c.drift_valid and spread <= c.spread_valid
                                         else 'PLATEAU_UNCERTAIN')
    result['derived']['plateau_diagnostics'] = pd
    established = result['plateau_status'] == 'PLATEAU_VALID'
    if not established:
        qc.append('NO_ESTABLISHED_PLATEAU')
    m['plateau'] = metric(C if established else None, units, None if established else 'NO_ESTABLISHED_PLATEAU', 'qualified final-window median', pd)
    m['delta'] = metric(Dc if established else None, units, None if established else 'NO_ESTABLISHED_PLATEAU', 'signed P-B')
    # Raw integrals preserve artifacts, but never bridge missing observations.
    auc = sum((x[i] + x[i + 1]) * .5 * intervals[i] for i in range(n - 1) if valid_edge[i])
    covered = sum(intervals[i] for i in range(n - 1) if valid_edge[i])
    m['observed_segment_auc'] = metric(auc if covered > 0 else None, f'{units}*{time_unit}',
                                       None if covered > 0 else 'NO_SUPPORTED_INTERVALS',
                                       method='supported raw trapezoids', details={'covered_duration': covered})
    m['integrated_duration'] = metric(covered, time_unit, method='sum of supported intervals')
    full = all(valid_edge)
    m['raw_auc'] = metric(auc if full else None, f'{units}*{time_unit}', None if full else 'INCOMPLETE_COVERAGE', 'raw trapezoids')
    adj = auc - B * T if full and B is not None else None
    m['baseline_adjusted_auc'] = metric(adj, f'{units}*{time_unit}', None if adj is not None else ('INCOMPLETE_COVERAGE' if not full else 'BASELINE_NOT_ESTABLISHED'), 'raw AUC-B*T')
    norm = adj / Dc if adj is not None and established else None
    for key, value, unit in [('normalized_auc_time', norm, time_unit), ('normalized_auc_fraction', norm / T if norm is not None else None, 'dimensionless')]:
        m[key] = metric(value, unit, None if value is not None else ('NO_ESTABLISHED_PLATEAU' if not established else 'INCOMPLETE_COVERAGE'), 'baseline-adjusted AUC / signed delta' + (' / duration' if 'fraction' in key else ''))
    ids, ok = supported_window(0, c.initial_slope_window, c.min_regression_samples)
    good_initial = T >= c.initial_slope_window and ok
    slope0 = theil_sen([u[i] for i in ids], [x[i] for i in ids]) if good_initial else None
    m['initial_slope'] = metric(slope0, f'{units}/{time_unit}', None if good_initial else 'INSUFFICIENT_INITIAL_SLOPE_SUPPORT', 'Theil-Sen fixed initial window', support_details(ids))
    local = []
    if T >= c.slope_window:
        starts = sorted(set(min(max(ui - c.slope_window / 2, 0.0), T - c.slope_window) for ui in u))
        for lo in starts:
            ids, ok = supported_window(lo, lo + c.slope_window, c.min_regression_samples)
            if ok:
                slope = theil_sen([u[i] for i in ids], [x[i] for i in ids])
                if math.isfinite(slope):
                    local.append({'start': lo, 'end': lo + c.slope_window, 'center': lo + c.slope_window / 2, 'count': len(ids), 'slope': slope})
    result['derived']['local_slopes'] = local
    best = max(local, key=lambda a: s * a['slope']) if local and s else None
    m['maximum_slope'] = metric(best['slope'] if best else None, f'{units}/{time_unit}', None if best else ('DIRECTION_UNDETERMINED' if not s else 'INSUFFICIENT_LOCAL_SLOPE_SUPPORT'), 'maximum direction-aligned local Theil-Sen slope; physical sign retained', best)
    diffs = [s * (x[i + 1] - x[i]) for i in range(n - 1) if valid_edge[i]]
    V = sum(abs(d) for d in diffs)
    reverse = sum(max(0.0, -d) for d in diffs) / V if V and s else None
    if reverse is not None and reverse > c.reverse_variation_limit:
        qc.append('STRONGLY_NONMONOTONIC')
    m['reverse_variation_fraction'] = metric(reverse, 'dimensionless', None if reverse is not None else 'DIRECTION_OR_VARIATION_UNDETERMINED', 'reverse variation / total raw variation')
    m['signed_monotonicity'] = metric(1 - 2 * reverse if reverse is not None else None, 'dimensionless', None if reverse is not None else 'DIRECTION_OR_VARIATION_UNDETERMINED', '1-2*reverse fraction')
    progress = [(v - B) / Dc if v is not None else None for v in x] if established else [None] * n
    result['derived']['progress'] = progress
    for pct in (10, 50, 90):
        key, q = f't{pct}', pct / 100
        val, detail = None, {}
        reason = 'NO_ESTABLISHED_PLATEAU' if not established else f'NO_T{pct}_CROSSING'
        if established:
            previous = None
            for j, p in enumerate(progress):
                if p is None:
                    continue
                if previous is None and p >= q:
                    reason = 'LEFT_CENSORED_CROSSING'
                    break
                if previous is not None and progress[previous] < q <= p:
                    if j != previous + 1 or not valid_edge[previous]:
                        reason = 'CROSSING_IN_GAP'
                    else:
                        f = (q - progress[previous]) / (p - progress[previous])
                        val = u[previous] + f * (u[j] - u[previous])
                        detail = {'indices': [previous, j], 'times': [u[previous], u[j]], 'bracket_width': u[j] - u[previous], 'interpolation_fraction': f}
                        reason = None
                    break
                previous = j
        if val is None:
            qc.append(f'NO_T{pct}_CROSSING')
        warnings = []
        if detail.get('bracket_width', 0) > c.coarse_crossing_interval:
            warnings.append('INSUFFICIENT_TIME_RESOLUTION')
            qc.extend(warnings)
        m[key] = metric(val, time_unit, reason, 'first supported raw linear crossing', detail, warnings)
    if m['t10']['value'] is not None and m['t90']['value'] is not None and m['t90']['value'] - m['t10']['value'] < c.min_transition_intervals * dt:
        qc.append('INSUFFICIENT_TIME_RESOLUTION')
        for key in ('t10', 't50', 't90'):
            m[key]['reason_codes'].append('INSUFFICIENT_TIME_RESOLUTION')
            m[key]['status'] = 'WARNING' if m[key]['value'] is not None else 'NOT_EVALUABLE'
    tp = None
    if established:
        band = max(c.plateau_band_fraction * abs(Dc), c.plateau_noise_multiple * sigma, floor)
        suffix_start = n
        for i in range(n - 1, -1, -1):
            if x[i] is None or abs(x[i] - C) > band or (i < n - 1 and not valid_edge[i]):
                break
            suffix_start = i
        if suffix_start < n and T - u[suffix_start] >= W:
            tp = u[suffix_start]
        m['time_to_plateau'] = metric(tp, time_unit, None if tp is not None else 'NO_SUSTAINED_PLATEAU', 'first observed sustained suffix in residence band', {'band': band, 'minimum_residence': W})
    else:
        m['time_to_plateau'] = metric(reason='NO_ESTABLISHED_PLATEAU', units=time_unit)
    # Warnings do not silently exclude values; attach relevant cautions to outputs.
    for key in ('t10', 't50', 't90', 'raw_auc', 'baseline_adjusted_auc', 'maximum_slope', 'initial_slope'):
        if m[key]['value'] is not None:
            extra = [q for q in qc if q in ('LARGE_OUTLIER', 'STRONGLY_NONMONOTONIC', 'EXPECTED_DIRECTION_MISMATCH')]
            m[key]['reason_codes'] = sorted(set(m[key]['reason_codes'] + extra))
            m[key]['status'] = 'WARNING' if m[key]['reason_codes'] else 'OK'
    result['qc'] = sorted(set(qc))
    if _has_nonfinite(result):
        return fail('NUMERICAL_OVERFLOW')
    return result


def build_temporal_profile(times, descriptors, *, fps=None, time_unit='s', config=None):
    """descriptors: name -> {'values': list, 'units': str, 'expected_direction': ...}."""
    lengths = {len(d['values']) for d in descriptors.values()}
    if len(lengths) > 1:
        raise ValueError('PROFILE_LENGTH_MISMATCH')
    sections = {}
    for name, data in descriptors.items():
        sections[name] = analyze_temporal_curve(times, data['values'], name, data.get('units', 'a.u.'),
            fps=fps, time_unit=time_unit, config=config,
            expected_direction=data.get('expected_direction', 'UNKNOWN_DOMAIN_DEPENDENT'))
    return {'schema': 'TEMPORAL STRUCTURAL PROFILE', 'engine_version': VERSION,
            'general_recording': next(iter(sections.values()))['recording'] if sections else {},
            'sections': sections, 'global_quality_score': None,
            'composite_indices': 'EXPERIMENTAL RBC COMPOSITE INDICES; unchanged; not computed here',
            'optical_label': 'MICROSTRUCTAI OPTICAL CURVE', 'equivalence_claim': False}
