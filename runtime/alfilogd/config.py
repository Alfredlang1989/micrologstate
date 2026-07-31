from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DOMAIN_ALIASES = {
    "FS": "FILESYSTEM",
    "DISK": "FILESYSTEM",
    "FILESYSTEM": "FILESYSTEM",
    "CPU": "CPU",
    "LOAD": "CPU",
    "NETWORK": "NETWORK",
    "NET": "NETWORK",
    "SECURITY": "SECURITY",
    "AUTH": "SECURITY",
    "MEMORY": "MEMORY",
    "MEM": "MEMORY",
    "RAM": "MEMORY",
    "SERVICE": "SERVICE",
    "APPLICATION": "SERVICE",
    "APP": "SERVICE",
    "HTTPD": "SERVICE",
    "APACHE": "SERVICE",
    "NGINX": "SERVICE",
    "ANY": "ANY",
    "*": "ANY",
}

POLLING_ALIASES = {
    "INOTIFY": "inotify",
    "EVENT": "inotify",
    "FILE_EVENT": "inotify",
    "POLL": "poll",
    "POLLING": "poll",
    "TIMER": "poll",
    "SYSTEMD_EVENT": "systemd_event",
    "SYSTEMD": "systemd_event",
    "JOURNAL": "systemd_event",
    "JOURNAL_EVENT": "systemd_event",
    "JOURNALD": "systemd_event",
}

LOGTYPE_ALIASES = {
    "FILE": "file",
    "SYSTEMD": "systemd",
    "JOURNAL": "systemd",
    "JOURNALD": "systemd",
}

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.I)


def parse_duration(value: str | int | float | None, default: float) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    match = _DURATION_RE.match(str(value))
    if not match:
        raise ValueError(f"invalid duration: {value!r}; use seconds or suffix s/m/h/d")
    number = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[suffix]
    return number * multiplier


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "ja"}:
        return True
    if normalized in {"0", "false", "no", "off", "nein"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def first_value(section: configparser.SectionProxy, names: Iterable[str], default: str | None = None) -> str | None:
    for name in names:
        if name in section:
            return section.get(name)
    return default


def normalize_domain(value: str) -> str:
    key = value.strip().upper()
    try:
        return DOMAIN_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported domain {value!r}") from exc


def normalize_polling(value: str) -> str:
    key = value.strip().upper()
    try:
        return POLLING_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported polling backend {value!r}") from exc


def normalize_logtype(value: str) -> str:
    key = value.strip().upper()
    try:
        return LOGTYPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported logtype {value!r}") from exc


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError(f"invalid empty/special-only name: {value!r}")
    return cleaned


@dataclass(slots=True)
class MainConfig:
    path: Path
    embedding_model: Path
    classifier: Path
    device: str = "auto"
    config_dir: Path = Path("/etc/alfilogd/conf.d")
    output_base: Path = Path("/var/alfilogd")
    state_base: Path = Path("/var/lib/alfilogd")
    runtime_dir: Path = Path("/run/alfilogd")
    socket_path: Path = Path("/run/alfilogd/alfilogd.sock")
    batch_size: int = 64
    batch_wait_ms: int = 25
    queue_size: int = 10000
    polling_rate: float = 300.0
    polling_lines: int = 500
    recovery_time: float = 3600.0
    status_write_interval: float = 10.0
    dedupe_time: float = 7200.0
    start_position: str = "end"
    max_line_bytes: int = 65536
    socket_mode: int = 0o660
    socket_group: str = "alfilogd"


@dataclass(slots=True)
class NrpeConfig:
    enabled: bool = True
    include_dir: Path = Path("/etc/nagios/nrpe.d")
    main_config: Path = Path("/etc/nagios/nrpe.cfg")
    service_name: str = "nrpe"
    generated_name: str = "alfilogd.cfg"
    validate_command: str = "/usr/sbin/nrpe -c /etc/nagios/nrpe.cfg -n"
    reload_command: str = "/usr/bin/systemctl reload nrpe"


@dataclass(slots=True)
class FacilityConfig:
    name: str
    enabled: bool
    logtype: str
    polling: str
    domain: str
    output: str
    logfile: Path | None = None
    units: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    priority: str | None = None
    polling_rate: float = 300.0
    polling_lines: int = 500
    recovery_time: float = 3600.0
    dedupe_time: float = 7200.0
    start_position: str = "end"
    critical_after: int = 1
    warning_after: int = 0
    recover_on_ok: bool = True
    unknown_overrides: bool = False
    max_active_errors: int = 256
    multiline: str = "none"
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def source_key(self) -> tuple[object, ...]:
        if self.logtype == "file":
            return (
                "file",
                str(self.logfile),
                self.polling,
                self.polling_rate,
                self.polling_lines,
                self.start_position,
                self.multiline,
            )
        return (
            "systemd",
            self.polling,
            self.units,
            self.identifiers,
            self.priority,
        )


@dataclass(slots=True)
class RuntimeConfig:
    main: MainConfig
    nrpe: NrpeConfig
    facilities: dict[str, FacilityConfig]


def _read_parser(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str.lower
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser


def load_config(path: Path) -> RuntimeConfig:
    path = path.resolve()
    parser = _read_parser(path)
    if "main" not in parser:
        raise ValueError(f"missing [main] in {path}")
    main_sec = parser["main"]

    embedding_raw = first_value(main_sec, ("embeddingmodel", "embedding_model", "encoder"))
    classifier_raw = first_value(main_sec, ("classifier", "classifiert", "classifier_model", "artifact"))
    if not embedding_raw or not classifier_raw:
        raise ValueError("[main] requires embeddingmodel= and classifier=")

    def resolve_config_path(raw: str) -> Path:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate
        return (path.parent / candidate).resolve()

    default_polling_rate = parse_duration(
        first_value(main_sec, ("polling_rate", "pollingrate")), 300.0
    )
    default_polling_lines = int(
        first_value(main_sec, ("polling_lines", "pollinglines", "plolling_lines"), "500") or 500
    )
    default_recovery = parse_duration(
        first_value(main_sec, ("recovery_time", "recoverytime", "wiederherstellungszeit")), 3600.0
    )
    state_base = resolve_config_path(first_value(main_sec, ("state_base", "statebase"), "/var/lib/alfilogd") or "/var/lib/alfilogd")
    runtime_dir = resolve_config_path(first_value(main_sec, ("runtime_dir", "runtimedir"), "/run/alfilogd") or "/run/alfilogd")
    socket_raw = first_value(main_sec, ("socket", "socket_path"), str(runtime_dir / "alfilogd.sock"))

    main = MainConfig(
        path=path,
        embedding_model=resolve_config_path(embedding_raw),
        classifier=resolve_config_path(classifier_raw),
        device=main_sec.get("device", "auto").strip(),
        config_dir=resolve_config_path(main_sec.get("config_dir", "/etc/alfilogd/conf.d")),
        output_base=resolve_config_path(main_sec.get("output_base", "/var/alfilogd")),
        state_base=state_base,
        runtime_dir=runtime_dir,
        socket_path=resolve_config_path(socket_raw or str(runtime_dir / "alfilogd.sock")),
        batch_size=max(1, int(main_sec.get("batch_size", "64"))),
        batch_wait_ms=max(0, int(main_sec.get("batch_wait_ms", "25"))),
        queue_size=max(32, int(main_sec.get("queue_size", "10000"))),
        polling_rate=max(1.0, default_polling_rate),
        polling_lines=max(1, default_polling_lines),
        recovery_time=max(1.0, default_recovery),
        status_write_interval=max(1.0, parse_duration(main_sec.get("status_write_interval"), 10.0)),
        dedupe_time=max(1.0, parse_duration(main_sec.get("dedupe_time"), max(7200.0, default_recovery * 2))),
        start_position=main_sec.get("start_position", "end").strip().lower(),
        max_line_bytes=max(1024, int(main_sec.get("max_line_bytes", "65536"))),
        socket_mode=int(main_sec.get("socket_mode", "0660"), 8),
        socket_group=main_sec.get("socket_group", "alfilogd").strip(),
    )
    if main.start_position not in {"beginning", "end", "cursor"}:
        raise ValueError("start_position must be beginning, end or cursor")

    nrpe_sec = parser["nrpe"] if "nrpe" in parser else None
    nrpe = NrpeConfig()
    if nrpe_sec is not None:
        nrpe = NrpeConfig(
            enabled=parse_bool(nrpe_sec.get("enabled"), True),
            include_dir=Path(nrpe_sec.get("include_dir", "/etc/nagios/nrpe.d")).expanduser(),
            main_config=Path(nrpe_sec.get("main_config", "/etc/nagios/nrpe.cfg")).expanduser(),
            service_name=nrpe_sec.get("service_name", "nrpe").strip(),
            generated_name=sanitize_name(nrpe_sec.get("generated_name", "alfilogd.cfg")),
            validate_command=nrpe_sec.get("validate_command", "/usr/sbin/nrpe -c /etc/nagios/nrpe.cfg -n").strip(),
            reload_command=nrpe_sec.get("reload_command", "/usr/bin/systemctl reload nrpe").strip(),
        )

    facility_parser = configparser.ConfigParser(interpolation=None, strict=False)
    facility_parser.optionxform = str.lower
    config_files = sorted(main.config_dir.glob("*.conf")) if main.config_dir.exists() else []
    facility_parser.read([str(item) for item in config_files], encoding="utf-8")

    facilities: dict[str, FacilityConfig] = {}
    for section_name in facility_parser.sections():
        sec = facility_parser[section_name]
        name = sanitize_name(section_name)
        enabled = parse_bool(sec.get("enabled"), True)
        logtype = normalize_logtype(sec.get("logtype", "file"))
        polling_default = "systemd_event" if logtype == "systemd" else "inotify"
        polling = normalize_polling(sec.get("polling", polling_default))
        if logtype == "systemd" and polling != "systemd_event":
            raise ValueError(f"[{section_name}] systemd sources require polling=systemd_event")
        if logtype == "file" and polling not in {"inotify", "poll"}:
            raise ValueError(f"[{section_name}] file source requires inotify or poll")

        domain = normalize_domain(sec.get("domain", section_name))
        output = sanitize_name(sec.get("output", name))
        logfile: Path | None = None
        units: tuple[str, ...] = ()
        identifiers: tuple[str, ...] = ()
        if logtype == "file":
            raw_logfile = first_value(sec, ("logfile", "file", "path"))
            if not raw_logfile:
                raise ValueError(f"[{section_name}] file source requires logfile=")
            logfile = Path(raw_logfile).expanduser()
        else:
            unit_values = first_value(sec, ("units", "unit"), "") or ""
            units = tuple(item.strip() for item in unit_values.split(",") if item.strip())
            identifier_values = first_value(sec, ("identifiers", "identifier"), "") or ""
            identifiers = tuple(item.strip() for item in identifier_values.split(",") if item.strip())
            if not units and not identifiers:
                raise ValueError(f"[{section_name}] systemd source requires unit(s)= or identifier(s)=")

        polling_rate = max(
            1.0,
            parse_duration(first_value(sec, ("polling_rate", "pollingrate")), main.polling_rate),
        )
        polling_lines = max(
            1,
            int(first_value(sec, ("polling_lines", "pollinglines", "plolling_lines"), str(main.polling_lines)) or main.polling_lines),
        )
        recovery_time = max(
            1.0,
            parse_duration(first_value(sec, ("recovery_time", "recoverytime", "wiederherstellungszeit")), main.recovery_time),
        )
        dedupe_time = max(
            1.0,
            parse_duration(sec.get("dedupe_time"), max(main.dedupe_time, recovery_time * 2)),
        )
        start_position = sec.get("start_position", main.start_position).strip().lower()
        if start_position not in {"beginning", "end", "cursor"}:
            raise ValueError(f"[{section_name}] invalid start_position={start_position}")

        known = {
            "enabled", "logtype", "polling", "domain", "output", "logfile", "file", "path",
            "unit", "units", "identifier", "identifiers", "priority", "polling_rate", "pollingrate",
            "polling_lines", "pollinglines", "plolling_lines", "recovery_time", "recoverytime",
            "wiederherstellungszeit", "dedupe_time", "start_position", "critical_after",
            "warning_after", "recover_on_ok", "unknown_overrides", "max_active_errors", "multiline",
        }
        facilities[name] = FacilityConfig(
            name=name,
            enabled=enabled,
            logtype=logtype,
            polling=polling,
            domain=domain,
            output=output,
            logfile=logfile,
            units=units,
            identifiers=identifiers,
            priority=sec.get("priority"),
            polling_rate=polling_rate,
            polling_lines=polling_lines,
            recovery_time=recovery_time,
            dedupe_time=dedupe_time,
            start_position=start_position,
            critical_after=max(1, int(sec.get("critical_after", "1"))),
            warning_after=max(0, int(sec.get("warning_after", "0"))),
            recover_on_ok=parse_bool(sec.get("recover_on_ok"), True),
            unknown_overrides=parse_bool(sec.get("unknown_overrides"), False),
            max_active_errors=max(1, int(sec.get("max_active_errors", "256"))),
            multiline=sec.get("multiline", "none").strip().lower(),
            extras={key: value for key, value in sec.items() if key not in known},
        )

    return RuntimeConfig(main=main, nrpe=nrpe, facilities=facilities)
