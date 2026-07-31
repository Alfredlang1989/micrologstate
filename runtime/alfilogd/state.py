from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import FacilityConfig, MainConfig
from .ontology import Classification, REASON_TEXT


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value is not None else None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass(slots=True)
class LogEvent:
    text: str
    source: str
    observed_at: datetime
    metadata: dict[str, Any]


@dataclass(slots=True)
class ActiveError:
    key: str
    first_seen: datetime
    last_seen: datetime
    count: int
    reason: str | None
    confidence: float
    source: str
    last_text: str


class FacilityState:
    def __init__(self, config: FacilityConfig) -> None:
        self.config = config
        self.active_errors: deque[ActiveError] = deque(maxlen=config.max_active_errors)
        self.initialized_at = utc_now()
        self.last_event_at: datetime | None = None
        self.last_accepted_at: datetime | None = None
        self.last_ok_at: datetime | None = None
        self.last_bad_at: datetime | None = None
        self.last_unknown_at: datetime | None = None
        self.last_classification: Classification | None = None
        self.last_source: str | None = None
        self.last_text: str | None = None
        self.events_total = 0
        self.accepted_total = 0
        self.bad_total = 0
        self.ok_total = 0
        self.unknown_total = 0

    def _error_key(self, result: Classification, source: str) -> str:
        payload = f"{result.domain}\0{result.reason or 'bad'}\0{source}\0{result.text}"
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:24]

    def apply(self, event: LogEvent, result: Classification) -> None:
        now = event.observed_at
        self.events_total += 1
        self.last_event_at = now
        self.last_classification = result
        self.last_source = event.source
        self.last_text = event.text

        if result.abstain or result.health is None:
            self.unknown_total += 1
            self.last_unknown_at = now
            return

        self.accepted_total += 1
        self.last_accepted_at = now
        if result.health == 0:
            self.ok_total += 1
            self.last_ok_at = now
            if self.config.recover_on_ok:
                self.active_errors.clear()
            return

        self.bad_total += 1
        self.last_bad_at = now
        key = self._error_key(result, event.source)
        for item in self.active_errors:
            if item.key == key:
                item.last_seen = now
                item.count += 1
                item.confidence = max(item.confidence, result.accept_score)
                item.last_text = event.text
                return
        self.active_errors.append(
            ActiveError(
                key=key,
                first_seen=now,
                last_seen=now,
                count=1,
                reason=result.reason,
                confidence=result.accept_score,
                source=event.source,
                last_text=event.text,
            )
        )

    def expire(self, now: datetime | None = None) -> int:
        now = now or utc_now()
        keep: deque[ActiveError] = deque(maxlen=self.config.max_active_errors)
        removed = 0
        for item in self.active_errors:
            age = (now - item.last_seen).total_seconds()
            if age <= self.config.recovery_time:
                keep.append(item)
            else:
                removed += 1
        self.active_errors = keep
        return removed

    def _status(self) -> tuple[int, str, str, float, str | None]:
        active_count = sum(item.count for item in self.active_errors)
        if active_count:
            newest = max(self.active_errors, key=lambda item: item.last_seen)
            reason_text = REASON_TEXT.get(newest.reason or "", newest.reason or "unhealthy")
            if active_count >= self.config.critical_after:
                return 2, "CRITICAL", reason_text, newest.confidence, newest.reason
            if self.config.warning_after == 0 or active_count >= self.config.warning_after:
                return 1, "WARNING", reason_text, newest.confidence, newest.reason

        if self.last_accepted_at is not None:
            message = f"no active errors in the last {int(self.config.recovery_time)} seconds"
            confidence = self.last_classification.accept_score if self.last_classification else 1.0
            return 0, "OK", message, confidence, "recovered"

        if self.config.unknown_overrides and self.last_unknown_at is not None:
            confidence = self.last_classification.accept_score if self.last_classification else 0.0
            return 3, "UNKNOWN", "events seen, but no reliable health state found", confidence, None

        return 3, "UNKNOWN", "facility initialized, no accepted health event received", 0.0, None

    def restore(self, payload: dict[str, Any]) -> None:
        self.initialized_at = parse_datetime(payload.get("initialized_at")) or self.initialized_at
        self.last_event_at = parse_datetime(payload.get("last_event_at"))
        self.last_accepted_at = parse_datetime(payload.get("last_accepted_at"))
        self.last_ok_at = parse_datetime(payload.get("last_ok_at"))
        self.last_bad_at = parse_datetime(payload.get("last_bad_at"))
        self.last_unknown_at = parse_datetime(payload.get("last_unknown_at"))
        self.last_source = payload.get("last_source")
        self.last_text = payload.get("last_text")
        counters = payload.get("counters", {})
        self.events_total = int(counters.get("events_total", 0))
        self.accepted_total = int(counters.get("accepted_total", 0))
        self.bad_total = int(counters.get("bad_total", 0))
        self.ok_total = int(counters.get("ok_total", 0))
        self.unknown_total = int(counters.get("unknown_total", 0))
        restored: deque[ActiveError] = deque(maxlen=self.config.max_active_errors)
        for item in payload.get("active_errors", []):
            first_seen = parse_datetime(item.get("first_seen"))
            last_seen = parse_datetime(item.get("last_seen"))
            if first_seen is None or last_seen is None:
                continue
            restored.append(
                ActiveError(
                    key=str(item.get("key", "")),
                    first_seen=first_seen,
                    last_seen=last_seen,
                    count=max(1, int(item.get("count", 1))),
                    reason=item.get("reason"),
                    confidence=float(item.get("confidence", 0.0)),
                    source=str(item.get("source", "restored")),
                    last_text=str(item.get("last_text", "")),
                )
            )
        self.active_errors = restored
        self.expire()

    def persisted(self) -> dict[str, Any]:
        payload = self.snapshot()
        payload.pop("daemon_updated_at", None)
        payload.pop("nagios", None)
        return payload

    def snapshot(self, daemon_updated_at: datetime | None = None) -> dict[str, Any]:
        daemon_updated_at = daemon_updated_at or utc_now()
        code, status, message, confidence, reason = self._status()
        active = sorted(self.active_errors, key=lambda item: item.last_seen, reverse=True)
        return {
            "version": 1,
            "facility": self.config.name,
            "output": self.config.output,
            "domain": self.config.domain,
            "status": status,
            "nagios_code": code,
            "message": message,
            "reason": reason,
            "confidence": confidence,
            "active_error_count": sum(item.count for item in active),
            "active_error_groups": len(active),
            "recovery_time_seconds": self.config.recovery_time,
            "critical_after": self.config.critical_after,
            "warning_after": self.config.warning_after,
            "initialized_at": iso(self.initialized_at),
            "last_event_at": iso(self.last_event_at),
            "last_accepted_at": iso(self.last_accepted_at),
            "last_ok_at": iso(self.last_ok_at),
            "last_bad_at": iso(self.last_bad_at),
            "last_unknown_at": iso(self.last_unknown_at),
            "daemon_updated_at": iso(daemon_updated_at),
            "last_source": self.last_source,
            "last_text": self.last_text,
            "counters": {
                "events_total": self.events_total,
                "accepted_total": self.accepted_total,
                "bad_total": self.bad_total,
                "ok_total": self.ok_total,
                "unknown_total": self.unknown_total,
            },
            "active_errors": [
                {
                    **asdict(item),
                    "first_seen": iso(item.first_seen),
                    "last_seen": iso(item.last_seen),
                }
                for item in active[:20]
            ],
            "nagios": f"{status} - {self.config.output.upper()}: {message} (confidence={confidence:.3f})",
        }


class StateManager:
    def __init__(self, main: MainConfig, facilities: dict[str, FacilityConfig]) -> None:
        self.main = main
        self.states = {name: FacilityState(config) for name, config in facilities.items() if config.enabled}
        self.main.output_base.mkdir(parents=True, exist_ok=True)
        self.main.state_base.mkdir(parents=True, exist_ok=True)
        self.persistence_base = self.main.state_base / "facilities"
        self.persistence_base.mkdir(parents=True, exist_ok=True)
        self._restore_all()

    def _persistence_path(self, state: FacilityState) -> Path:
        return self.persistence_base / f"{state.config.name}.json"

    def _restore_all(self) -> None:
        for state in self.states.values():
            try:
                payload = json.loads(self._persistence_path(state).read_text(encoding="utf-8"))
                state.restore(payload)
            except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
                continue

    def apply(self, facility_name: str, event: LogEvent, result: Classification) -> None:
        state = self.states.get(facility_name)
        if state is None:
            return
        state.apply(event, result)

    def expire_all(self) -> int:
        now = utc_now()
        return sum(state.expire(now) for state in self.states.values())

    def status_path(self, state: FacilityState) -> Path:
        return self.main.output_base / state.config.output / "status.json"

    def write_all(self) -> None:
        now = utc_now()
        for state in self.states.values():
            output_dir = self.main.output_base / state.config.output
            output_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(output_dir, 0o755)
            path = output_dir / "status.json"
            payload = state.snapshot(now)
            atomic_write_json(path, payload, mode=0o644)
            atomic_write_json(self._persistence_path(state), state.persisted(), mode=0o640)

    def get_snapshot(self, facility_name: str) -> dict[str, Any] | None:
        state = self.states.get(facility_name)
        return state.snapshot() if state is not None else None


def atomic_write_json(path: Path, payload: dict[str, Any], mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
