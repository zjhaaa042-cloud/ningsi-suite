"""Read-only history of archived tests alongside sessions in the current Studio database."""

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import stat

from ningsi_studio.api import schemas
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.http.router import ApiError, NotFound, Response


_UUID = re.compile(r"[0-9a-f]{32}")
_FIELDS = ("uuid", "label", "participant", "device", "source", "status", "phase", "progress",
           "started_at", "created_at", "ended_at", "history_kind", "origin_label", "source_ref",
           "summary", "protocol", "protocol_label", "phase_label", "time_scale", "alert_count",
           "error", "original_status", "sample_count", "duration_seconds", "hide_participant")
_TYPES = {".html": "text/html", ".md": "text/plain", ".json": "application/json"}
_OFFLINE_HEADERS = {"Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'; "
                                              "img-src data:; base-uri 'none'; form-action 'none'"}


def _archive_file(archive, relative):
    """Allow only regular files reached without crossing a symlink or Windows reparse point."""
    current = archive
    try:
        for part in ("", *Path(relative).parts):
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return None
        return current if stat.S_ISREG(info.st_mode) else None
    except (FileNotFoundError, NotADirectoryError):
        return None


def _load(archive):
    index = _archive_file(archive, "index.json")
    if index is None:
        return {}
    try:
        data = json.loads(index.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("items"), list):
            raise ValueError("unknown archive format")
        rows = {}
        for item in data["items"]:
            if not isinstance(item, dict) or not isinstance(item.get("uuid"), str) or not _UUID.fullmatch(item["uuid"]):
                raise ValueError("invalid archive record")
            if not all(item.get(k) is None or isinstance(item[k], str) for k in ("label", "participant", "status")):
                raise ValueError("invalid archive metadata")
            if item["uuid"] in rows:
                raise ValueError("duplicate archive UUID")
            rows[item["uuid"]] = dict(item, archived=True)
        return rows
    except (ValueError, OSError) as error:
        raise ApiError("无法读取本机历史测试索引") from error


def _report(archive, item):
    name = item.get("report_file")
    if not isinstance(name, str) or name not in {f"reports/{item['uuid']}{ext}" for ext in _TYPES}:
        return None
    return _archive_file(archive, name)


def _report_data(archive, item):
    relative = f"report_data/{item['uuid']}.json"
    if item.get("report_data_file") == relative:
        return _archive_file(archive, relative)
    return None


def contextual_report(report, row):
    """Enrich a view of a saved report without rewriting its original evidence."""
    extras = report.get("extras") or {}
    context = dict(extras.get("report_context") or {})
    for name, key in (("title", "label"), ("uuid", "uuid"), ("started_at", "started_at"),
                      ("channels", "channels"), ("srate", "srate"), ("source_label", "origin_label"),
                      ("signal_origin", "signal_origin"), ("response_mode", "response_mode"),
                      ("recording_ref", "recording_ref"), ("hide_participant", "hide_participant")):
        if row.get(key) is not None:
            context[name] = row[key]
    # Older built-in Studio demos saved the device and speed before they saved the
    # answer mode. Fast sim-bsense sessions use the runtime's automatic task answers.
    speed = extras.get("time_scale")
    if (not context.get("response_mode") and extras.get("device") == "sim-bsense"
            and extras.get("source_kind") == "sim" and isinstance(speed, (float, int))
            and not isinstance(speed, bool) and 0 < speed < 0.2):
        context["response_mode"] = "automated"
    return dict(report, extras=dict(extras, report_context=context))


def presentation_response(report):
    from ningsi.assessment.diagnostics import render_html
    return Response(body=render_html(report).encode("utf-8"), content_type="text/html; charset=utf-8",
                    headers=_OFFLINE_HEADERS)


def _public(archive, item):
    row = {key: item.get(key) for key in _FIELDS if isinstance(item.get(key), (str, float, int))}
    route = "/api/test-history/" + row["uuid"]
    row.update(archived=True, evidence_url=route + "/evidence",
               presentation_url=route + "/presentation" if _report_data(archive, item) else None,
               report_url=route + "/report" if _report(archive, item) else None)
    return row


def _on_date(item):
    value = item.get("created_at") or item.get("started_at")
    try:
        instant = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return instant.replace(tzinfo=timezone.utc) if instant.tzinfo is None else instant.astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _evidence(row, *, link_report):
    from ningsi.assessment.report_presentation import report_header

    status = {"done": "已完成", "failed": "失败", "cancelled": "已取消", "unfinished": "未记录结束"}
    overview = [("状态", status.get(row.get("status"), row.get("status"))),
                ("时长（秒）", row.get("duration_seconds")), ("采样点数", row.get("sample_count"))]
    metrics = "".join(f'<div class="metric"><dt>{key}</dt><dd>{escape(str(value))}</dd></div>'
                      for key, value in overview if value is not None)
    facts = [("会话编号", row["uuid"]), ("采集设备", row.get("device")), ("原始记录", row.get("source_ref")),
             ("结束时间", row.get("ended_at")), ("阶段", row.get("phase_label") or row.get("phase")),
             ("原始状态", row.get("original_status")), ("错误信息", row.get("error"))]
    fields = "".join(f"<dt>{key}</dt><dd>{escape(str(value))}</dd>" for key, value in facts
                     if value is not None and value != "")
    context = {"title": row.get("label") or row["uuid"], "source_label": row.get("origin_label") or row.get("source"),
               "started_at": row.get("started_at") or row.get("created_at"), "hide_participant": row.get("hide_participant")}
    header = report_header({"participant": row.get("participant"), "extras": {
        "report_context": context, "protocol_label": row.get("protocol_label") or "历史记录"}})
    links = (("打开评估展示报告", row.get("presentation_url")), ("打开原始报告", row.get("report_url")))
    actions = "".join(f'<p><a href="{escape(url)}">{label}</a></p>' for label, url in links if link_report and url)
    summary = f'<p>{escape(str(row["summary"]))}</p>' if row.get("summary") else ""
    missing = ('<p class="note">这条记录没有保存完整评估报告。</p>'
               if not row.get("report_url") and not row.get("presentation_url") else "")
    page = (header + '<section class="overview"><h2>记录概览</h2>'
            + (f'<dl class="measurements">{metrics}</dl>' if metrics else "")
            + summary + missing + actions + '</section><details class="appendix"><summary>查看记录明细</summary>'
            + f'<dl style="overflow-wrap:anywhere">{fields}</dl></details></main></body></html>')
    return Response(body=page.encode(), content_type="text/html; charset=utf-8", headers=_OFFLINE_HEADERS)


def register(router, settings, session_public):
    archive = Path(settings.data_dir).resolve() / "history_archive"

    @router.get("/api/test-history")
    def list_history(request):
        limit, offset = schemas.pagination(request)
        participant = request.query.get("participant", "").strip()
        status = request.query.get("status")
        records = _load(archive)
        with store.read_only(settings.db_path) as conn:
            conn.execute("BEGIN")
            live = conn.execute("""SELECT uuid, sessions.created_at, started_at, status, public_id AS participant
                                   FROM sessions JOIN subjects ON subjects.id = sessions.subject_id""")
            # Live records win even when the archive copy has different status or participant.
            for item in live:
                records[item["uuid"]] = dict(item, archived=False)
            people = sorted({r["participant"] for r in records.values() if r.get("participant")},
                            key=lambda code: (code.casefold(), code))
            visible_codes = {r["participant"] for r in records.values() if r.get("participant") and not r.get("hide_participant")}
            live_codes = {r["participant"] for r in records.values() if not r["archived"] and r.get("participant")}
            live_people = [code for code in people if code in live_codes]
            if participant and participant not in people:
                participant = participant.removeprefix("sub-") or participant
                if participant not in people:
                    candidates = [code for code in people if code.casefold() == participant.casefold()]
                    if len(candidates) == 1:
                        participant = candidates[0]
            matches = [r for r in records.values()
                       if (not participant or r.get("participant") == participant)
                       and (not status or r.get("status") == status)]
            matches.sort(key=lambda r: (_on_date(r), r["uuid"]), reverse=True)
            page = matches[offset:offset + limit]
            live_ids = [r["uuid"] for r in page if r.get("archived") is False]
            rows, _ = repo.list_sessions(conn, uuids=live_ids, limit=limit)
            rendered = {row["uuid"]: session_public(row, extra={
                "archived": False, "alert_count": len(repo.list_alerts(conn, row["id"]))}) for row in rows}
        items = [rendered[r["uuid"]] if r["uuid"] in rendered else _public(archive, r) for r in page]
        return Response.json({"items": items, "participants": [p for p in people if p in visible_codes],
                              "live_participants": live_people, "total": len(matches),
                              "limit": limit, "page": offset // limit + 1})

    def archived_record(request):
        uuid = schemas.ensure_uuid(request.params["uuid"])
        with store.read_only(settings.db_path) as conn:
            if repo.get_session(conn, uuid) is not None:
                raise NotFound("请在当前会话中查看这条记录")
        row = _load(archive).get(uuid)
        if row is None:
            raise NotFound("未找到该归档测试")
        return row

    @router.get("/api/test-history/{uuid}/evidence")
    def evidence(request):
        return _evidence(_public(archive, archived_record(request)), link_report=not settings.api_token)

    @router.get("/api/test-history/{uuid}/report")
    def report(request):
        path = _report(archive, archived_record(request))
        if path is None:
            raise NotFound("该归档测试没有可打开的原始报告")
        return Response(body=path.read_bytes(), content_type=_TYPES[path.suffix] + "; charset=utf-8",
                        headers=_OFFLINE_HEADERS)

    @router.get("/api/test-history/{uuid}/presentation")
    def presentation(request):
        item = archived_record(request)
        path = _report_data(archive, item)
        if path is None:
            raise NotFound("这次测试未保存结构化报告")
        return presentation_response(contextual_report(json.loads(path.read_text(encoding="utf-8-sig")), item))
