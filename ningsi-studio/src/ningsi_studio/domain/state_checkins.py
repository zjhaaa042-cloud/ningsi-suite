"""Time-stamped participant ratings. The EEG snapshot, if present, precedes the answer."""

import json
from statistics import fmean

from ningsi.monitoring.history import aggregate
from ningsi_studio.db import repository as repo
from ningsi_studio.db.sqlite_store import utcnow


CHECKIN_PHASES = ("scales", "monitor", "training")
HIGH_STRESS = 7
HIGH_MESSAGE = "你刚才自评的压力较高，可以暂停任务并稍作休息。"


def public(row):
    return {key: row[key] for key in ("request_id", "phase", "stress", "valence", "recorded_at",
                                      "t_sec", "source_kind")} | {
        "eeg_context": json.loads(row["eeg_context"]) if row["eeg_context"] else None}


def save(conn, session, request_id, stress, valence, *, source_kind, elapsed, eeg_context):
    # The caller holds a write transaction. Retry and concurrent submissions share one ID.
    key = (session["id"], request_id)
    old = conn.execute("SELECT * FROM state_checkins WHERE session_id=? AND request_id=?", key).fetchone()
    if old:
        if old["stress"] != stress or old["valence"] != valence:
            raise ValueError("这次提交编号已经对应另一份作答")
        return public(old), None
    if not source_kind or session["status"] != "running" or session["phase"] not in CHECKIN_PHASES:
        raise ValueError("请在运行中的量表、监测或训练阶段记录当下感受")
    stamp = utcnow()
    payload = json.dumps(eeg_context, ensure_ascii=False, allow_nan=False) if eeg_context else None
    conn.execute("""INSERT INTO state_checkins
                    (session_id,request_id,phase,stress,valence,source_kind,recorded_at,t_sec,eeg_context)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                 (*key, session["phase"], stress, valence, source_kind, stamp, elapsed, payload))
    alert = None
    if stress is not None and stress >= HIGH_STRESS:
        alert = {"kind": "self_reported_high_stress", "state": "triggered", "value": stress,
                 "t_sec": elapsed, "message": HIGH_MESSAGE}
        repo.add_alert(conn, session["id"], **alert)
    row = conn.execute("SELECT * FROM state_checkins WHERE session_id=? AND request_id=?", key).fetchone()
    return public(row), alert


def summary(conn, session_id):
    rows = [public(row) for row in conn.execute(
        "SELECT * FROM state_checkins WHERE session_id=? ORDER BY id", (session_id,))]
    values = [r["stress"] for r in rows if r["stress"] is not None]
    return {"records": rows, "stress_mean": round(fmean(values), 2) if values else None,
            "high_stress_reports": sum(value >= HIGH_STRESS for value in values)}


def stress_trend(conn, participant, period):
    # Average each visit first so frequently reporting participants cannot overweight a visit.
    sql = """SELECT s.uuid, s.created_at AS recorded_at, AVG(c.stress) AS stress
             FROM state_checkins c JOIN sessions s ON s.id = c.session_id
             JOIN subjects p ON p.id = s.subject_id
             WHERE s.status='done' AND c.source_kind='lsl' AND c.stress IS NOT NULL"""
    args = []
    if participant:
        sql += " AND p.public_id=?"
        args.append(participant)
    records = [dict(r) for r in conn.execute(sql + " GROUP BY s.id", args)]
    return {"field": "stress", "period": period, "points": aggregate(records, "stress", period),
            "sessions": len(records), "comparable": len(records), "rejected": 0,
            "rejected_reasons": {}, "ledger_points": [], "unit": "0–10 分",
            "note": ("所选被试。" if participant else "当前汇总所有被试。")
                    + "压力来自本人当下自评（0–10 分）。仅统计已完成的现场 LSL 会话。先计算各会话均值，再按周期平均。"}


def report_data(conn, session_id, diagnostics):
    result = summary(conn, session_id)
    records = result["records"]
    lines = []
    if records:
        lines.append("当下感受由本人主动选择时间提交，可能偏向压力较高或较低的时刻。")
        lines.append(f"本人提交当下感受 {len(records)} 次，压力达到提醒线 {result['high_stress_reports']} 次。")
        if result["stress_mean"] is not None:
            lines.append(f"自报压力均值 {result['stress_mean']}/10。")
        for r in records:
            values = []
            if r["stress"] is not None:
                values.append(f"压力 {r['stress']}/10")
            if r["valence"] is not None:
                values.append(f"愉悦程度 {r['valence']}/10")
            lines.append(f"{r['recorded_at']}　" + "，".join(values))
        if result["high_stress_reports"]:
            lines.append(HIGH_MESSAGE)
    else:
        lines.append("本次未收到当下压力或愉悦程度自评。")
    eligible = [r for r in diagnostics if r.get("phase") == "monitor" and r.get("affect_model")]
    for name, label in (("stress", "压力"), ("valence", "情绪效价")):
        scored = [r["affect_model"][name] for r in eligible if r["affect_model"][name]["available"]]
        if scored:
            model = scored[-1]
            classification = f"，模型类别 {model['label']}" if model.get("display_class") else ""
            lines.append(f"独立监测阶段有 {len(scored)}/{len(eligible)} 窗获得脑电{label}研究分数，"
                         f"最近一次为 {model['score']:.3f}{classification}；本机真人判别能力待验证。")
        else:
            lines.append(f"独立监测阶段未获得可用的脑电{label}模型分数。")
    result["summary_lines"] = lines
    return result
