from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import random
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

DOMAINS = ("FILESYSTEM", "CPU", "NETWORK", "SECURITY", "MEMORY", "SERVICE")

RULES: list[dict[str, Any]] = [
    {"domain": "FILESYSTEM", "health": 1, "reason": "full", "templates": ["filesystem {mount} full", "ERROR disk {mount} is full", "No space left on device {mount}"]},
    {"domain": "FILESYSTEM", "health": 0, "reason": "not_full", "templates": ["filesystem {mount} not full", "disk {mount} has free capacity", "filesystem {mount} healthy and writable"]},
    {"domain": "FILESYSTEM", "health": 1, "reason": "not_writable", "templates": ["filesystem {mount} not writable", "write failed on {path}", "filesystem remounted read-only: {mount}"]},
    {"domain": "FILESYSTEM", "health": None, "reason": "mention", "templates": ["filesystem {mount}", "filesystem statistics for {mount}", "checking filesystem configuration"]},
    {"domain": "CPU", "health": 1, "reason": "high_load", "templates": ["high load", "CPU load {load} above threshold", "load average is high: {load}", "processor saturated at {percent}%"]},
    {"domain": "CPU", "health": 0, "reason": "normal_load", "templates": ["CPU load normal", "load average {load} below threshold", "processor utilization healthy at {percent}%"]},
    {"domain": "CPU", "health": None, "reason": "mention", "templates": ["CPU information", "load statistics", "show processor metrics"]},
    {"domain": "MEMORY", "health": 1, "reason": "oom", "templates": ["Out of memory: killed process {pid}", "memory exhausted", "allocation failed for {bytes} bytes"]},
    {"domain": "MEMORY", "health": 1, "reason": "memory_pressure", "templates": ["memory pressure high", "available memory low: {percent}%", "swap usage critical"]},
    {"domain": "MEMORY", "health": 0, "reason": "memory_normal", "templates": ["memory pressure cleared", "memory healthy", "available memory {percent}%"]},
    {"domain": "MEMORY", "health": None, "reason": "mention", "templates": ["memory information", "RAM statistics", "memory configuration loaded"]},
    {"domain": "NETWORK", "health": 1, "reason": "timeout", "templates": ["connection timed out to {dst_ip}:{port}", "network timeout", "upstream {dst_ip}:{port} timed out"]},
    {"domain": "NETWORK", "health": 1, "reason": "unreachable", "templates": ["destination {dst_ip} unreachable", "network is unreachable", "interface {iface} down"]},
    {"domain": "NETWORK", "health": 0, "reason": "up", "templates": ["network link {iface} up", "connection established to {dst_ip}:{port}", "route restored"]},
    {"domain": "NETWORK", "health": None, "reason": "mention", "templates": ["network configuration", "interface {iface}", "routing table follows"]},
    {"domain": "SECURITY", "health": 1, "reason": "auth_failed", "templates": ["Failed password for {user} from {src_ip} port {port}", "authentication failed for {user}", "access denied for account {user}"]},
    {"domain": "SECURITY", "health": 1, "reason": "brute_force", "templates": ["repeated failed logins for {user} from {src_ip}", "possible brute force from {src_ip}", "{count} authentication failures detected"]},
    {"domain": "SECURITY", "health": 0, "reason": "auth_success", "templates": ["Accepted password for {user} from {src_ip} port {port}", "authentication successful for {user}", "security audit success event {event_id}"]},
    {"domain": "SECURITY", "health": None, "reason": "mention", "templates": ["security audit configuration", "account {user}", "authentication subsystem loaded"]},
    {"domain": "SERVICE", "health": 1, "reason": "service_down", "templates": ["{service}.service is down", "service {service} not running", "health check failed for {service}"]},
    {"domain": "SERVICE", "health": 1, "reason": "failed", "templates": ["{service}.service entered failed state", "process {service} exited with code {exit_code}", "watchdog timeout for {service}"]},
    {"domain": "SERVICE", "health": 0, "reason": "service_up", "templates": ["{service}.service active running", "service {service} started successfully", "health check passed for {service}"]},
    {"domain": "SERVICE", "health": None, "reason": "mention", "templates": ["service {service}", "service configuration for {service}", "dependency graph for {service}"]},
]

WRAPPERS = [
    "{message}",
    "{level}: {message}",
    "{timestamp} {host} {program}[{pid}]: {message}",
    "{program}: {message}",
    "domain={domain} level={level} message=\"{message}\"",
]

PATTERNS = {
    "src_ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "user": re.compile(r"\b(?:user|account|for)[:= ]+([A-Za-z0-9_.@-]{2,64})", re.I),
    "path": re.compile(r"(?:[A-Za-z]:\\[^\s\"']+|/(?:[^\s\"']+/)*[^\s\"']*)"),
    "service": re.compile(r"\b([A-Za-z0-9_.@-]+)\.service\b"),
    "host": re.compile(r"\b[A-Za-z0-9][A-Za-z0-9.-]{2,}\.(?:local|internal|example|com|net|org)\b"),
}

DEFAULT_POOLS: dict[str, list[str]] = {
    "mount": ["/", "/var", "/var/lib/mysql", "/srv", "/home", "C:", "D:"],
    "path": ["/var/lib/mysql/binlog", "/srv/data/archive.dat", "/tmp/cache.bin", r"C:\Windows\Temp\setup.log"],
    "user": ["root", "admin", "deploy", "oracle", "www-data", "svc_backup"],
    "service": ["nginx", "httpd", "apache2", "mysql", "postgresql", "sshd", "php-fpm"],
    "host": ["web01.example.internal", "db02.example.internal", "app17.local"],
    "program": ["kernel", "systemd", "sshd", "nginx", "httpd", "winlogbeat"],
    "iface": ["eth0", "ens192", "bond0", "enp3s0"],
}


def random_ip(rng: random.Random) -> str:
    network = rng.choice([ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("192.168.0.0/16"), ipaddress.ip_network("203.0.113.0/24")])
    return str(network.network_address + rng.randrange(1, max(2, network.num_addresses - 1)))


def harvest(paths: Iterable[Path], limit: int = 250000) -> dict[str, list[str]]:
    pools: dict[str, set[str]] = defaultdict(set)
    seen = 0
    for root in paths:
        files = [root] if root.is_file() else list(root.rglob("*.jsonl")) + list(root.rglob("*.ndjson")) + list(root.rglob("*.log"))
        for path in files:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        if seen >= limit:
                            break
                        seen += 1
                        text = line
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict):
                                text = str(row.get("text") or row.get("message") or line)
                        except json.JSONDecodeError:
                            pass
                        for name, pattern in PATTERNS.items():
                            for match in pattern.finditer(text):
                                value = match.group(1) if match.lastindex else match.group(0)
                                if 1 < len(value) < 300:
                                    pools[name].add(value)
            except OSError:
                continue
    return {name: sorted(values) for name, values in pools.items()}


def fields(rng: random.Random, pools: dict[str, list[str]]) -> dict[str, str]:
    now = datetime.now(timezone.utc) - timedelta(seconds=rng.randrange(0, 365 * 86400))
    value = {name: rng.choice(values) for name, values in pools.items() if values}
    value.update({
        "src_ip": rng.choice(pools.get("src_ip", [])) if pools.get("src_ip") else random_ip(rng),
        "dst_ip": random_ip(rng),
        "port": str(rng.randrange(1, 65536)),
        "pid": str(rng.randrange(2, 65000)),
        "count": str(rng.randrange(2, 500)),
        "percent": str(rng.randrange(1, 101)),
        "load": f"{rng.uniform(0.01, 64.0):.2f}",
        "bytes": str(rng.choice([4096, 65536, 1048576, 67108864, rng.randrange(128, 10_000_000)])),
        "event_id": str(rng.randrange(1000, 9999)),
        "exit_code": str(rng.choice([1, 2, 3, 127, 255])),
        "timestamp": now.isoformat(timespec="seconds"),
        "level": rng.choice(["INFO", "NOTICE", "WARNING", "ERROR", "CRITICAL"]),
        "guid": str(uuid.UUID(int=rng.getrandbits(128))),
    })
    return value


def group_id(domain: str, reason: str, template: str) -> str:
    digest = hashlib.sha256(template.encode()).hexdigest()[:16]
    return f"factory:{domain}:{reason}:{digest}"


def generate(per_template: int, seed: int, harvested: dict[str, list[str]]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pools = {name: list(values) for name, values in DEFAULT_POOLS.items()}
    for name, values in harvested.items():
        pools.setdefault(name, []).extend(values)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in RULES:
        for template in rule["templates"]:
            group = group_id(rule["domain"], rule["reason"], template)
            for _ in range(per_template):
                values = fields(rng, pools)
                message = template.format_map(defaultdict(str, values))
                wrapper = rng.choice(WRAPPERS)
                text = wrapper.format(message=message, domain=rule["domain"], **values)
                if text in seen:
                    continue
                seen.add(text)
                abstain = 1 if rule["health"] is None else 0
                rows.append({
                    "text": text,
                    "domain": rule["domain"],
                    "health": rule["health"],
                    "abstain": abstain,
                    "reason": rule["reason"],
                    "group": group,
                    "source": "local-template-factory",
                    "sample_weight": 1.0,
                })
    rng.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic local log dataset mixer")
    parser.add_argument("--output", type=Path, default=Path("data/processed/log_states_v3.jsonl"))
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--per-template", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    harvested = harvest(args.corpus)
    rows = generate(max(1, args.per_template), args.seed, harvested)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "domains": dict(defaultdict(int, {domain: sum(row["domain"] == domain for row in rows) for domain in DOMAINS})),
        "harvested": {name: len(values) for name, values in harvested.items()},
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
