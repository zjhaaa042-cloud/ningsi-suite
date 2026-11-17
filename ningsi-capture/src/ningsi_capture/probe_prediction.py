"""Blind predictions using only EEG committed by the software probe onset."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import math

import numpy as np

from .bootstrap import ensure_engine
ensure_engine()
from ningsi.acquisition.eeg import source_type_conflict, unit_conflict
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, BIOMULTI_MAX_DELAY, BIOMULTI_REC_TIMING, PACKET_TIMING
from ningsi.models import attlapse, attlapse_shadow
from ningsi.models.capture_protocols import RESEARCH_PROTOCOLS
from ningsi.models.research_features import aggregate_probe
from ningsi.signal.baseline import build_baseline

from .export import analysis_windows
from .storage import open_readonly


def lookup_margin(source):
    return {PACKET_TIMING: .05, BIOMULTI_LSL_TIMING: BIOMULTI_MAX_DELAY}.get(source.get("timestamp_policy"), 0.)


class ProbePredictions:
    def __init__(self, directory, metadata, save_result):
        self.directory = directory
        self.metadata = metadata
        self.save_result = save_result
        self.write_failures = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture-probe")
        # Warm the worker before acquisition so the first probe only enqueues its snapshot.
        self.executor.submit(lambda: None).result()
        self.phase_starts = {}
        self.calibration = None
        self.baseline_valid = None
        self.model_error = None
        try:
            self.model = attlapse.load_candidate()
        except (OSError, TypeError, AttributeError, KeyError, ValueError, OverflowError):
            self.model = None
            self.model_error = "model_invalid"
        self.shadow_error = None
        try:
            self.shadow = attlapse_shadow.load_candidates()
        except (OSError, TypeError, AttributeError, KeyError, ValueError, OverflowError):
            self.shadow = None
            self.shadow_error = "model_invalid"

    def shadow_metadata(self):
        return {"model_bundle_id": self.shadow["bundle_id"] if self.shadow else None,
                "model_ids": self.shadow["model_ids"] if self.shadow else {},
                "rate_hz": 250, "lookback_sec": [-18, -2], "montage": "Fp1-Fp2",
                "mode": "blind_at_onset", "quality": "physical", "locally_calibrated": False,
                "task_elapsed_origin": "first_formal_sart_phase_start_including_breaks",
                "protocol_elapsed_origin": "protocol_start",
                "probe_ordinal_basis": "one_based_formal_probe_onsets_across_blocks"}

    def observe(self, kind, payload, stamp, event_id, sample_count):
        phase = payload.get("phase")
        if kind == "phase_start":
            self.phase_starts[phase] = (stamp.lsl, sample_count)
        elif kind == "phase_end" and phase == "baseline_calibration" and not payload.get("incomplete"):
            self.calibration = (self.phase_starts.get(phase), stamp.lsl, sample_count)
        elif kind == "probe_onset":
            self.executor.submit(self._record, event_id, bool(payload.get("practice")), phase,
                                 stamp, sample_count, self.phase_starts.get(phase), self.calibration)

    def close(self):
        self.executor.shutdown(wait=True)

    def _record(self, event_id, practice, phase, stamp, sample_count, phase_start, calibration):
        base = {"model_id": self.model["model_id"] if self.model else None,
                "sample_count_at_onset": sample_count, "window_start_lsl": stamp.lsl - 12,
                "window_end_lsl": stamp.lsl - 2}
        try:
            result = self._predict(practice, phase, stamp, sample_count, phase_start, calibration)
        except Exception as exc:
            # A background model failure cannot interrupt the protocol or discard the answer.
            result = {"reason": "prediction_error", "error_type": type(exc).__name__}
        result["status"] = "unavailable" if "reason" in result else "scored"
        # Keep the two research candidates independent of any legacy inference failure.
        shadow = {"window_start_lsl": stamp.lsl - 18, "window_end_lsl": stamp.lsl - 2,
                  "sample_count_at_onset": sample_count, "logistic_score": None, "rbf_score": None,
                  "model_bundle_id": self.shadow["bundle_id"] if self.shadow else None,
                  "model_ids": self.shadow["model_ids"] if self.shadow else {}}
        try:
            with closing(open_readonly(self.directory)) as reader:
                shadow.update(self._shadow_predict(reader, practice, phase, stamp, sample_count, phase_start))
        except Exception as exc:
            shadow.update(reason="prediction_error", error_type=type(exc).__name__)
        shadow["status"] = "unavailable" if "reason" in shadow else "scored"
        # The ordinary QC window can include more recent samples than the model;
        # compute it separately and never pass its state or baseline into scoring.
        try:
            with closing(open_readonly(self.directory)) as reader:
                shadow.update(self._input_quality(reader, stamp, sample_count))
                shadow["baseline_valid"] = self._baseline(reader, calibration, stamp, sample_count)
        except Exception as exc:
            shadow.setdefault("live_input_window_status", "unavailable")
            shadow.setdefault("live_input_window_reasons", ["quality_error"])
            shadow["quality_error_type"] = type(exc).__name__
            shadow.setdefault("baseline_valid", None)
        result["shadow_250"] = shadow
        try:
            self.save_result(event_id, {**base, **result})
        except Exception as exc:
            self.write_failures.append({"onset_event_id": event_id, "error_type": type(exc).__name__})

    def _input_quality(self, reader, stamp, sample_count):
        source = self.metadata["source"]
        if source.get("timestamp_policy") == BIOMULTI_REC_TIMING:
            return {"live_input_window_status": "unavailable", "live_input_window_reasons": ["native_clock_pending"]}
        rate = source["srate"]
        size, hop = round(4 * rate), max(1, round(2 * rate))
        latest = reader.execute("SELECT sample_index, received_perf FROM samples WHERE sample_index < ? AND received_perf <= ? "
                                "ORDER BY sample_index DESC LIMIT 1", (sample_count, stamp.perf)).fetchone()
        if latest is None:
            return {"live_input_window_status": "unavailable", "live_input_window_reasons": ["signal_too_short"]}
        latest_age = stamp.perf - latest["received_perf"]
        if latest["sample_index"] + 1 < size:
            reasons = ["signal_too_short"]
            if latest_age >= 3:
                reasons.append("signal_stale")
            return {"live_input_window_status": "unavailable", "live_input_window_reasons": reasons,
                    "live_input_last_packet_age_sec": latest_age}
        end = size + ((latest[0] + 1 - size) // hop) * hop
        row, = analysis_windows(reader, self.metadata, first_index=end - 1, stop_index=end,
                                received_by=stamp.perf, sample_limit=sample_count)
        flags = list(row["window"].quality.reasons)
        received = reader.execute("SELECT received_perf FROM samples WHERE sample_index=?", (end - 1,)).fetchone()[0]
        window_age = stamp.perf - received
        if latest_age >= 3:
            flags.append("signal_stale")
        if window_age >= 3:
            flags.append("quality_window_stale")
        if row["end"] is not None and abs(stamp.lsl - row["end"]) >= 3:
            flags.append("window_clock_difference_over_3s")
        # end is the exclusive edge of a sample interval, up to one period beyond its timestamp.
        if row["end"] is not None and row["end"] - 1 / rate > stamp.lsl + 1e-8:
            flags.append("timestamp_after_probe")
        return {"live_input_window_status": "failed" if flags else "passed", "live_input_window_reasons": flags,
                "live_input_window_start_lsl": row["start"], "live_input_window_end_lsl": row["end"],
                "live_input_window_sample_end_index": row["sample_end_index"],
                "live_input_last_packet_age_sec": latest_age, "live_input_window_received_age_sec": window_age}

    def _baseline(self, reader, calibration, stamp, sample_count):
        if self.metadata["source"].get("timestamp_policy") == BIOMULTI_REC_TIMING:
            return None
        if not calibration or not calibration[0] or calibration[1] > stamp.lsl:
            return None
        if self.baseline_valid is None:
            (begin, first), end, stop = calibration
            source = self.metadata["source"]
            # The frozen public candidate's baseline is part of its original feature contract.
            candidate_source = {**source, "preprocessing_policy": None}
            calibration_rows = analysis_windows(reader, {**self.metadata, "source": candidate_source}, first_index=first,
                                                stop_index=min(sample_count, stop), received_by=stamp.perf,
                                                sample_limit=sample_count)
            windows = [row["window"] for row in calibration_rows if row["start"] is not None
                       and row["end"] is not None and row["start"] >= begin and row["end"] <= end]
            self.baseline_valid = build_baseline(windows, device=source["name"], srate=source["srate"]).valid
        return self.baseline_valid

    def _shadow_predict(self, reader, practice, phase, stamp, sample_count, phase_start):
        source = self.metadata["source"]
        reason = self._eligibility(practice, phase)
        if reason:
            return {"reason": reason}
        if self.shadow is None:
            return {"reason": self.shadow_error or "model_unavailable"}
        if float(source["srate"]) != 250.:
            return {"reason": "rate_mismatch"}
        reason = self._source_problem(source)
        if reason:
            return {"reason": reason}
        try:
            indices = attlapse.channel_indices(source.get("channel_labels") or ())
        except ValueError:
            return {"reason": "electrode_mismatch"}
        left, right = stamp.lsl - 18, stamp.lsl - 2
        margin = lookup_margin(source)
        if phase_start is None or phase_start[0] > left:
            return {"reason": "lookback_crosses_phase"}
        bounds = reader.execute(
            "SELECT MIN(sample_index), MAX(sample_index) FROM samples WHERE corrected_timestamp "
            "BETWEEN ? AND ? AND sample_index < ? AND received_perf <= ?",
            (left - margin - 1e-8, right + margin + 1e-8, sample_count, stamp.perf)).fetchone()
        windows = []
        expected = None
        if bounds[0] is not None:
            rate = source["srate"]
            anchor = reader.execute("SELECT corrected_timestamp FROM samples WHERE sample_index=?",
                                    (bounds[0],)).fetchone()[0]
            # Candidate ends lie on the recording-origin 2 s grid. Also count windows lost to
            # a dead stream at the tail. Near a jittered boundary the aggregator keeps at
            # least as many candidates as the corrected clock actually places in the interval.
            grid_origin = anchor - bounds[0] / rate
            first_end = math.ceil((left + 4 - grid_origin) / 2 - 1e-8)
            last_end = math.floor((right - grid_origin) / 2 + 1e-8)
            expected = last_end - first_end + 1

            def analyze(context, source):
                values = np.stack([item[0] for item in context]).T
                times = np.asarray([item[1] for item in context], dtype=float)
                return attlapse_shadow.compute_window(values, times, source["srate"], source["scale_to_uv"], indices)

            windows = analysis_windows(reader, self.metadata, first_index=bounds[0], stop_index=bounds[1] + 1,
                                       analyze=analyze, received_by=stamp.perf, sample_limit=sample_count)
        return attlapse_shadow.aggregate_and_score(self.shadow, windows, stamp.lsl, phase_start[0],
                                                  expected_windows=expected)

    def _eligibility(self, practice, phase):
        source = self.metadata["source"]
        if practice:
            return "practice"
        if source.get("kind") != "lsl" or source.get("data_origin") != "real":
            return "simulation_unverified"
        if self.metadata["protocol"].get("name") not in RESEARCH_PROTOCOLS or not (phase or "").startswith("sart_"):
            return "nonresearch_protocol"
        return None

    @staticmethod
    def _source_problem(source):
        if source.get("timestamp_policy") == BIOMULTI_REC_TIMING:
            return "native_clock_pending"
        if source_type_conflict(source):
            return "source_type_mismatch"
        if unit_conflict(source):
            return "unit_conflict"
        if source.get("scale_to_uv") is None:
            return "unit_unconfirmed"
        return None

    def _predict(self, practice, phase, stamp, sample_count, phase_start, calibration):
        source = self.metadata["source"]
        reason = self._eligibility(practice, phase)
        if reason:
            return {"reason": reason}
        if self.model is None:
            return {"reason": self.model_error or "model_unavailable"}
        reason = self._source_problem(source)
        if reason:
            return {"reason": reason}
        try:
            indices = attlapse.channel_indices(source.get("channel_labels") or ())
        except ValueError:
            return {"reason": "electrode_mismatch"}
        if phase_start is None or phase_start[0] > stamp.lsl - 12:
            return {"reason": "lookback_crosses_phase"}
        if not calibration or not calibration[0] or calibration[1] > stamp.lsl:
            return {"reason": "baseline_invalid"}
        with closing(open_readonly(self.directory)) as reader:
            if not self._baseline(reader, calibration, stamp, sample_count):
                return {"reason": "baseline_invalid"}
            # Get the exact range of candidate ends using the corrected source clock.
            # The capture's timestamp index bounds the lookup independently of session length.
            margin = lookup_margin(source)
            bounds = reader.execute(
                "SELECT MIN(sample_index), MAX(sample_index) FROM samples WHERE corrected_timestamp "
                "BETWEEN ? AND ? AND sample_index < ? AND received_perf <= ?",
                (stamp.lsl - 12 - margin, stamp.lsl - 2 + margin, sample_count, stamp.perf)).fetchone()
            if bounds[0] is None:
                return {"reason": "insufficient_usable_eeg"}

            def bipolar(context, source):
                values = np.stack([item[0] for item in context]).T
                stamps = np.asarray([item[1] for item in context], dtype=float)
                return attlapse.bipolar_window(values, stamps, source["srate"], source["scale_to_uv"], indices)

            windows = analysis_windows(reader, self.metadata, first_index=bounds[0], stop_index=bounds[1] + 1,
                                       analyze=bipolar, received_by=stamp.perf, sample_limit=sample_count)
        result = aggregate_probe(windows, stamp.lsl, phase_start=phase_start[0], baseline_valid=True)
        details = {key: result[key] for key in ("windows_total", "windows_usable", "usable_seconds")}
        if result["reasons"]:
            return {**details, "reason": "|".join(result["reasons"])}
        score = attlapse.score_probe(self.model, result["features"])
        return {**details, "features": result["features"], "mw_score": score, "threshold": self.model["threshold"],
                "predicted_label": int(score >= self.model["threshold"])}
