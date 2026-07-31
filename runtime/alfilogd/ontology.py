from __future__ import annotations

from dataclasses import dataclass

DOMAINS = ["FILESYSTEM", "CPU", "NETWORK", "SECURITY", "MEMORY", "SERVICE", "UNKNOWN"]

DOMAIN_DESCRIPTIONS = {
    "FILESYSTEM": "filesystem",
    "CPU": "CPU/load",
    "NETWORK": "network",
    "SECURITY": "security",
    "MEMORY": "memory",
    "SERVICE": "service/application",
    "UNKNOWN": "unclassified source",
}

REASON_TEXT = {
    "full": "filesystem full",
    "capacity": "filesystem capacity exhausted",
    "pressure": "filesystem capacity under pressure",
    "inode_full": "filesystem inodes exhausted",
    "read_only": "filesystem read-only",
    "not_writable": "filesystem not writable",
    "io_error": "filesystem I/O error",
    "mount_failed": "filesystem mount failed",
    "recovered": "filesystem recovered",
    "normal": "filesystem healthy",
    "not_full": "filesystem not full",
    "high_load": "high CPU load",
    "cpu_saturation": "CPU saturated",
    "iowait": "CPU I/O wait high",
    "steal": "CPU steal time high",
    "throttling": "CPU throttling detected",
    "normal_load": "CPU load normal",
    "down": "network unavailable",
    "timeout": "network timeout",
    "refused": "connection refused",
    "unreachable": "destination unreachable",
    "dns_failure": "DNS resolution failed",
    "packet_loss": "packet loss high",
    "link_down": "network link down",
    "up": "network healthy",
    "auth_failed": "authentication failed",
    "brute_force": "repeated authentication failures",
    "access_denied": "access denied",
    "suspicious": "suspicious security activity",
    "malicious": "malicious security activity",
    "auth_success": "authentication successful",
    "audit_success": "security audit success",
    "oom": "out of memory",
    "memory_pressure": "memory pressure high",
    "swap_full": "swap exhausted",
    "allocation_failed": "memory allocation failed",
    "memory_normal": "memory healthy",
    "service_down": "service down",
    "failed": "service failed",
    "restart_loop": "service restart loop",
    "watchdog": "service watchdog timeout",
    "health_failed": "service health check failed",
    "http_server_error": "HTTP service error",
    "http_success": "HTTP service healthy",
    "service_up": "service running",
    "healthy": "healthy",
    "bad": "unhealthy",
    "mention": "state not stated",
    "irrelevant": "unclassified event",
    "sequence_only": "state requires sequence context",
}


@dataclass(slots=True)
class Classification:
    text: str
    domain: str
    health: int | None
    abstain: bool
    reason: str | None
    reason_confidence: float | None
    domain_confidence: float
    health_confidence: float
    abstain_probability: float
    accept_score: float
    nagios_code: int
    nagios: str


def render_nagios(domain: str, health: int | None, abstain: bool, confidence: float, reason: str | None) -> tuple[int, str]:
    confidence = max(0.0, min(1.0, float(confidence)))
    if abstain or health is None or domain == "UNKNOWN":
        message = "unclassified log event" if domain == "UNKNOWN" else f"{DOMAIN_DESCRIPTIONS.get(domain, domain)} recognized, but no reliable health state found"
        return 3, f"UNKNOWN - {message} (confidence={confidence:.3f})"
    reason_text = REASON_TEXT.get(reason or "")
    if reason_text is None:
        reason_text = "healthy" if health == 0 else "unhealthy"
    if health == 0:
        return 0, f"OK - {domain}: {reason_text} (confidence={confidence:.3f})"
    return 2, f"CRITICAL - {domain}: {reason_text} (confidence={confidence:.3f})"
