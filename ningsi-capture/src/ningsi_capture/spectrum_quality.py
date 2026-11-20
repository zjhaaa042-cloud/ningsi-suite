"""Descriptive band power from time-aligned, artifact-screened EEG."""

from bisect import bisect_right
from math import ceil

import numpy as np

from ningsi import config
from ningsi.signal import conditioning
from ningsi.signal.artifacts import check_channels
from ningsi.signal.bands import band_power, band_powers, relative_band_powers
from ningsi.signal.spectrum import welch_psd


POLICY = "capture-clean-segment-spectrum-v2"
SINGLE_CHANNEL_POLICY = "capture-single-channel-spectrum-v1"
FRAGMENT_SECONDS = (1.25, 1.50)
MIN_FRAGMENT_SUPPORT_SEC = 2.50
MOTION_ASSISTED_POLICY = "capture-rec-gyro-near-full-segment-v1"
MOTION_QUIET_CUTOFF = .70  # BioMulti 1.0.9-E-Release three-axis gyro norm in vendor units.
MOTION_MIN_CLEAN_SEC = 1.90
MOTION_CLOCK_MARGIN_SEC = .12


def _continuous_runs(values, stamps, rate):
    valid = np.isfinite(values).all(axis=0) & np.isfinite(stamps)
    steps = np.diff(stamps)
    joins = valid[:-1] & valid[1:] & (steps > 0) & (steps <= 1.75 / rate)
    starts = np.flatnonzero(valid & ~np.r_[False, joins])
    stops = np.flatnonzero(valid & ~np.r_[joins, False]) + 1
    return zip(starts, stops, strict=True)


def _stable_spectrum(segment, spectra, rate):
    prefix_size = segment.shape[1] - round(.10 * rate)
    for channel, whole in zip(segment, spectra, strict=True):
        prefix = welch_psd(channel[:prefix_size], rate, segment_sec=prefix_size / rate)
        before = relative_band_powers(prefix.freqs, prefix.psd)
        after = relative_band_powers(whole.freqs, whole.psd)
        prefix_power = sum(band_powers(prefix.freqs, prefix.psd).values())
        whole_power = sum(band_powers(whole.freqs, whole.psd).values())
        if (prefix_power <= 0 or abs(whole_power / prefix_power - 1) > .10
                or any(abs(after[band] - before[band]) > .02 for band in before)):
            return False
    return True


def _unmarked_candidates(mask, origin, first, segment_size):
    for start, stop in conditioning._runs(mask == 0):
        left = max(first, origin + start)
        yield left, min(origin + stop, left + segment_size)


def _fixed_candidates(mask, origin, first, stop, lengths, search_step):
    for start, end in conditioning._runs(mask == 0):
        left, right = max(first, origin + start), min(stop, origin + end)
        for length in lengths:
            last = right - length
            if left <= last:
                # Include both edges of each unmarked span, even off the search grid.
                grid_start = first + ((left - first + search_step - 1) // search_step) * search_step
                positions = {left, last, *range(grid_start, last, search_step)}
                for position in sorted(positions):
                    yield position, position + length


def _motion_quiet(motion, start, end):
    return motion is None or motion.assess(
        [start - MOTION_CLOCK_MARGIN_SEC, end + MOTION_CLOCK_MARGIN_SEC],
        threshold=MOTION_QUIET_CUTOFF)["status"] == "quiet"


def _screened_segments(raw, stamps, rate, lengths, *, near_full=False, disjoint=True, motion=None, channel=None,
                       first_sample=None):
    window_size = round(config.WINDOW_SEC * rate)
    min_size = round(MOTION_MIN_CLEAN_SEC * rate) if near_full else min(lengths)
    history = round(6 * rate) - window_size
    report_start = max(0, len(stamps) - window_size) if first_sample is None else first_sample
    search_step = max(1, round(rate * .1))

    for start, stop in _continuous_runs(raw, stamps, rate):
        first = max(report_start, start + history)
        if first > stop - min_size:
            continue
        run = raw[:, start:stop]
        wave = conditioning.filter_uv(run, rate)
        mask = conditioning._local_artifacts(wave, run, rate)
        if channel is not None:
            selected = slice(channel, channel + 1)
            # Eye, muscle and signal fault evidence can affect both frontal leads.
            shared = mask & (conditioning.EYE | conditioning.MUSCLE | conditioning.SIGNAL_FAULT)
            mask = conditioning._local_artifacts(wave[selected], run[selected], rate) | shared
        origin = start + history
        if near_full:
            candidates = _unmarked_candidates(mask, origin, first, lengths[0])
        else:
            candidates = _fixed_candidates(mask, origin, first, stop, lengths, search_step)
        next_sample = first
        for left, right in candidates:
            length = right - left
            if left < next_sample or length < min_size:
                continue
            history_duration = stamps[left] - stamps[left - history]
            segment_duration = stamps[right - 1] - stamps[left]
            if (abs(history_duration * rate / history - 1) > .05
                    or abs(segment_duration * rate / (length - 1) - 1) > .05):
                continue
            if not _motion_quiet(motion, stamps[left], stamps[right - 1] + 1 / rate):
                continue
            segment = wave[:, left - origin:right - origin]
            if channel is not None:
                if np.any(np.std(segment, axis=1) < config.QUALITY["flat_std_uv"]):
                    continue
                segment = segment[selected]
            spectra = [welch_psd(channel, rate, segment_sec=length / rate) for channel in segment]
            if not check_channels(segment, rate, spectra=spectra).ok:
                continue
            if near_full and not _stable_spectrum(segment, spectra, rate):
                continue
            yield left, right, spectra
            if disjoint:
                next_sample = right


def _clean_segments(raw, stamps, rate, *, near_full=False, motion=None, channel=None, first_sample=None):
    length = round(config.WELCH["segment_sec"] * rate)
    return _screened_segments(raw, stamps, rate, (length,), near_full=near_full, motion=motion, channel=channel,
                              first_sample=first_sample)


def _fragmented_segments(raw, stamps, rate, motion):
    lengths = tuple(ceil(seconds * rate) for seconds in FRAGMENT_SECONDS)
    candidates = sorted(_screened_segments(raw, stamps, rate, lengths, disjoint=False, motion=motion),
                        key=lambda row: row[1])
    ends = [right for _, right, _ in candidates]
    # Maximize the number of actually retained samples, then prefer fewer segments.
    best = [(0, 0, [])]
    for index, (left, right, spectra) in enumerate(candidates):
        samples, count, kept = best[bisect_right(ends, left, hi=index)]
        addition = (samples + right - left, count - 1, [*kept, (left, right, spectra)])
        best.append(max(best[-1], addition, key=lambda row: row[:2]))
    samples, _, kept = best[-1]
    return kept if samples >= ceil(MIN_FRAGMENT_SUPPORT_SEC * rate) else []


def _fragment_powers(kept):
    # Integrate each spectrum on its own frequency grid. Never join time samples across artifacts or gaps.
    absolute = []
    totals = []
    weights = []
    for left, right, spectra in kept:
        for spectrum in spectra:
            absolute.append(band_powers(spectrum.freqs, spectrum.psd))
            totals.append(band_power(spectrum.freqs, spectrum.psd, *config.TOTAL_BAND))
            weights.append(right - left)
    powers = {band: float(np.average([row[band] for row in absolute], weights=weights)) for band in config.BANDS}
    total = float(np.average(totals, weights=weights))
    return powers, {band: power / total if total > 0 else 0. for band, power in powers.items()}


def _unavailable_spectrum():
    return {"status": "unavailable", "support_sec": 0., "segments": 0,
            "band_powers": {}, "relative_bands": {}, "intervals": [], "method": "unavailable"}


def _estimate_segments(values, stamps, rate, scale_to_uv, *, near_full=False, motion=None, channel=None,
                       first_sample=None):
    if scale_to_uv is None or not np.isfinite(scale_to_uv) or scale_to_uv <= 0 or rate < 100:
        return _unavailable_spectrum()
    if np.any(np.diff(stamps[np.isfinite(stamps)]) <= 0):
        return _unavailable_spectrum()
    with np.errstate(over="ignore", invalid="ignore"):
        raw = values * scale_to_uv
    kept = list(_clean_segments(raw, stamps, rate, near_full=near_full, motion=motion, channel=channel,
                                first_sample=first_sample))
    fragmented = False
    if not kept and not near_full and channel is None and first_sample is None:
        kept = _fragmented_segments(raw, stamps, rate, motion)
        fragmented = bool(kept)
    if not kept:
        return _unavailable_spectrum()
    if fragmented:
        powers, relative = _fragment_powers(kept)
        method = "short_fragments"
    else:
        freqs = kept[0][2][0].freqs
        spectra_psd = [spectrum.psd for _, _, spectra in kept for spectrum in spectra]
        weights = ([right - left for left, right, spectra in kept for _ in spectra]
                   if near_full or first_sample is not None else None)
        psd = np.average(spectra_psd, axis=0, weights=weights)
        powers, relative = band_powers(freqs, psd), relative_band_powers(freqs, psd)
        method = "near_full_gyro" if near_full else "continuous_segments"
    intervals = []
    for left, right, _ in kept:
        end = stamps[right - 1] + 1 / rate
        if first_sample is not None and right < len(stamps) and np.isfinite(stamps[right]):
            if stamps[right] > stamps[right - 1]:
                end = min(end, stamps[right])
        intervals.append((float(stamps[left]), float(end)))
    return {"status": "partial", "support_sec": sum(right - left for left, right, _ in kept) / rate,
            "segments": len(kept), "band_powers": powers, "relative_bands": relative, "method": method,
            "segment_lengths_sec": [(right - left) / rate for left, right, _ in kept],
            "intervals": intervals}


def describe_spectrum(values, stamps, rate, scale_to_uv, full_window, *, motion=None):
    """Use whole windows or time-local support for descriptive frequency bands."""
    width = round(config.WINDOW_SEC * rate)
    if full_window.usable and _motion_quiet(motion, stamps[-width], full_window.t_end):
        return {"status": "full", "support_sec": width / rate,
                "segments": full_window.spectrum["segments"], "band_powers": full_window.powers,
                "relative_bands": full_window.rel, "method": "full_window",
                "intervals": [(float(stamps[-width]), full_window.t_end)]}
    return _estimate_segments(values, stamps, rate, scale_to_uv, motion=motion)


def describe_interval_spectrum(values, stamps, rate, scale_to_uv, *, first, motion=None):
    """Pool nonoverlapping clean two-second EEG segments in the requested interval.

    Samples before `first` provide filtering history. The caller sets the end by
    slicing the input. After a gap, each continuous run needs its own history.
    """
    values = np.asarray(values, dtype=float)
    stamps = np.asarray(stamps, dtype=float)
    if (stamps.ndim != 1 or values.shape != (2, len(stamps))
            or not isinstance(first, (int, np.integer)) or not 0 <= first < len(stamps)
            or not np.isfinite(rate)):
        return _unavailable_spectrum()
    return _estimate_segments(values, stamps, rate, scale_to_uv, motion=motion, first_sample=first)


def describe_single_channels(values, stamps, rate, scale_to_uv, dual_spectrum, motion, labels):
    """Keep independently screened single-lead bands if no dual-lead estimate is available."""
    if dual_spectrum["status"] != "unavailable" or motion is None or values.shape[0] != 2:
        return []
    kept = []
    for channel in range(2):
        spectrum = _estimate_segments(values, stamps, rate, scale_to_uv, motion=motion, channel=channel)
        if spectrum["status"] == "partial":
            label = labels[channel] if channel < len(labels) and labels[channel] else f"Ch{channel + 1}"
            kept.append({"channel_index": channel + 1, "channel_label": label, **spectrum})
    return kept


def describe_motion_assisted_spectrum(values, stamps, rate, scale_to_uv, full_window, original_spectrum, motion):
    """Allow near-full clean support with quiet IMU and only a motion-like EEG rejection."""
    width = round(config.WINDOW_SEC * rate)
    span = stamps[-width:]
    if len(span) > 1:
        span = np.r_[span[0] - MOTION_CLOCK_MARGIN_SEC, span, span[-1] + MOTION_CLOCK_MARGIN_SEC]
    window_imu = motion.assess(span, threshold=MOTION_QUIET_CUTOFF)
    report = {"imu_peak": window_imu.get("peak"), "imu_status": window_imu["status"],
              "imu_reason": window_imu.get("reason")}
    if original_spectrum["status"] != "unavailable":
        return {**original_spectrum, **report, "decision": "standard"}
    if rate == 250 and full_window.quality.reasons == ("motion_transient",):
        if window_imu["status"] == "motion":
            return {**original_spectrum, **report, "decision": "imu_motion"}
        if window_imu["status"] != "quiet":
            return {**original_spectrum, **report, "decision": "imu_coverage_incomplete"}
        recovered = _estimate_segments(values, stamps, rate, scale_to_uv, near_full=True, motion=motion)
        if recovered["status"] == "partial":
            return {**recovered, **report, "decision": "quiet_imu_near_full_clean_segment"}
    return {**original_spectrum, **report, "decision": "no_safe_segment"}
