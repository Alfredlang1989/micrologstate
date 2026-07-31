from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import FacilityConfig, MainConfig
from .state import LogEvent, atomic_write_json

LOGGER = logging.getLogger(__name__)
EmitCallback = Callable[[LogEvent, tuple[str, ...]], Awaitable[None]]


def source_digest(source_key: tuple[object, ...]) -> str:
    raw = json.dumps(source_key, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class CursorStore:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)

    def path_for(self, source_key: tuple[object, ...]) -> Path:
        return self.base / f"{source_digest(source_key)}.json"

    def load(self, source_key: tuple[object, ...]) -> dict[str, Any]:
        path = self.path_for(source_key)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def save(self, source_key: tuple[object, ...], payload: dict[str, Any]) -> None:
        atomic_write_json(self.path_for(source_key), payload, mode=0o640)


class DedupeCache:
    def __init__(self, path: Path, ttl: float, max_entries: int = 10000) -> None:
        self.path = path
        self.ttl = ttl
        self.max_entries = max_entries
        self.items: OrderedDict[str, float] = OrderedDict()
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            now = time.time()
            for key, timestamp in payload.get("items", []):
                timestamp = float(timestamp)
                if now - timestamp <= self.ttl:
                    self.items[str(key)] = timestamp
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            self.items.clear()

    def _prune(self) -> None:
        cutoff = time.time() - self.ttl
        while self.items:
            first_key = next(iter(self.items))
            if self.items[first_key] >= cutoff and len(self.items) <= self.max_entries:
                break
            self.items.popitem(last=False)

    def seen(self, value: str) -> bool:
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        self._prune()
        if digest in self.items:
            self.items.move_to_end(digest)
            return True
        self.items[digest] = time.time()
        self._prune()
        return False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, {"items": list(self.items.items())}, mode=0o640)


class PathWakeHandler(FileSystemEventHandler):
    def __init__(self, target: Path, callback: Callable[[], None]) -> None:
        self.target = str(target.resolve())
        self.callback = callback

    def on_any_event(self, event: FileSystemEvent) -> None:
        paths = [getattr(event, "src_path", ""), getattr(event, "dest_path", "")]
        if any(path and str(Path(path).resolve()) == self.target for path in paths):
            self.callback()


class FileInotifySource:
    def __init__(
        self,
        main: MainConfig,
        template: FacilityConfig,
        targets: tuple[str, ...],
        emit: EmitCallback,
        cursor_store: CursorStore,
    ) -> None:
        if template.logfile is None:
            raise ValueError("file source without logfile")
        self.main = main
        self.template = template
        self.targets = targets
        self.emit = emit
        self.cursor_store = cursor_store
        self.path = template.logfile
        self.source_key = template.source_key
        self._file: Any | None = None
        self._inode: int | None = None
        self._offset = 0
        self._cursor_loaded = False

    def _load_cursor(self) -> dict[str, Any]:
        return self.cursor_store.load(self.source_key)

    def _save_cursor(self) -> None:
        self.cursor_store.save(
            self.source_key,
            {
                "type": "file",
                "path": str(self.path),
                "inode": self._inode,
                "offset": self._offset,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _open_current(self, force_beginning: bool = False) -> None:
        if not self.path.exists():
            return
        stat = self.path.stat()
        handle = self.path.open("rb", buffering=0)
        cursor = self._load_cursor() if not self._cursor_loaded else {}
        self._cursor_loaded = True
        if force_beginning:
            offset = 0
        elif cursor and int(cursor.get("inode", -1)) == int(stat.st_ino):
            offset = min(int(cursor.get("offset", 0)), int(stat.st_size))
        elif self.template.start_position == "beginning":
            offset = 0
        else:
            offset = int(stat.st_size)
        handle.seek(offset)
        self._file = handle
        self._inode = int(stat.st_ino)
        self._offset = offset
        LOGGER.info("watching %s inode=%s offset=%s", self.path, self._inode, self._offset)

    async def _emit_line(self, raw: bytes) -> None:
        text = raw.rstrip(b"\r\n")[: self.main.max_line_bytes].decode("utf-8", errors="replace")
        if not text.strip():
            return
        event = LogEvent(
            text=text,
            source=str(self.path),
            observed_at=datetime.now(timezone.utc),
            metadata={"logtype": "file", "path": str(self.path)},
        )
        await self.emit(event, self.targets)

    async def _read_to_eof(self) -> None:
        if self._file is None:
            return
        while True:
            position = int(self._file.tell())
            raw = self._file.readline(self.main.max_line_bytes + 1)
            if not raw:
                break
            if not raw.endswith(b"\n"):
                self._file.seek(position)
                break
            self._offset = int(self._file.tell())
            await self._emit_line(raw)
        self._save_cursor()

    async def _refresh_file(self) -> None:
        if self._file is None:
            self._open_current()
            return
        try:
            current = self.path.stat()
        except FileNotFoundError:
            await self._read_to_eof()
            return
        if int(current.st_ino) != self._inode:
            await self._read_to_eof()
            self._file.close()
            self._file = None
            self._inode = None
            self._offset = 0
            self._open_current(force_beginning=True)
            return
        if int(current.st_size) < self._offset:
            LOGGER.info("file truncated: %s", self.path)
            self._file.seek(0)
            self._offset = 0

    async def run(self, stop: asyncio.Event) -> None:
        if not self.path.parent.exists():
            raise FileNotFoundError(f"watch directory does not exist: {self.path.parent}")
        loop = asyncio.get_running_loop()
        wakeup = asyncio.Event()

        def wake() -> None:
            loop.call_soon_threadsafe(wakeup.set)

        observer = Observer()
        observer.schedule(PathWakeHandler(self.path, wake), str(self.path.parent), recursive=False)
        observer.start()
        try:
            self._open_current()
            await self._read_to_eof()
            while not stop.is_set():
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                wakeup.clear()
                await self._refresh_file()
                if self._file is None:
                    self._open_current(force_beginning=True)
                await self._read_to_eof()
        finally:
            observer.stop()
            observer.join(timeout=5)
            if self._file is not None:
                await self._read_to_eof()
                self._file.close()


def tail_last_lines(path: Path, count: int, max_bytes: int = 16 * 1024 * 1024) -> list[str]:
    if count <= 0:
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        block_size = 65536
        data = bytearray()
        while position > 0 and data.count(b"\n") <= count and len(data) < max_bytes:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            data[:0] = handle.read(read_size)
        lines = bytes(data).splitlines()[-count:]
    return [line.decode("utf-8", errors="replace") for line in lines]


class FilePollSource:
    def __init__(
        self,
        main: MainConfig,
        template: FacilityConfig,
        targets: tuple[str, ...],
        emit: EmitCallback,
        cursor_store: CursorStore,
    ) -> None:
        if template.logfile is None:
            raise ValueError("file source without logfile")
        self.main = main
        self.template = template
        self.targets = targets
        self.emit = emit
        self.path = template.logfile
        cache_path = cursor_store.path_for(template.source_key).with_suffix(".dedupe.json")
        self.dedupe = DedupeCache(cache_path, template.dedupe_time)

    async def scan(self) -> None:
        try:
            lines = await asyncio.to_thread(
                tail_last_lines, self.path, self.template.polling_lines
            )
        except FileNotFoundError:
            LOGGER.warning("poll source missing: %s", self.path)
            return
        except PermissionError:
            LOGGER.exception("cannot read poll source: %s", self.path)
            return
        changed = False
        for text in lines:
            text = text[: self.main.max_line_bytes]
            fingerprint_value = f"{self.path}\0{text}"
            if not text.strip() or self.dedupe.seen(fingerprint_value):
                continue
            changed = True
            await self.emit(
                LogEvent(
                    text=text,
                    source=str(self.path),
                    observed_at=datetime.now(timezone.utc),
                    metadata={"logtype": "file", "path": str(self.path), "polling": True},
                ),
                self.targets,
            )
        if changed:
            self.dedupe.save()

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.scan()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.template.polling_rate)
            except asyncio.TimeoutError:
                pass
        self.dedupe.save()


class JournalSource:
    def __init__(
        self,
        main: MainConfig,
        template: FacilityConfig,
        targets: tuple[str, ...],
        emit: EmitCallback,
        cursor_store: CursorStore,
    ) -> None:
        self.main = main
        self.template = template
        self.targets = targets
        self.emit = emit
        self.cursor_store = cursor_store
        self.source_key = template.source_key

    def command(self, cursor: str | None) -> list[str]:
        command = ["journalctl", "--follow", "--output=json", "--no-pager"]
        if cursor:
            command.append(f"--after-cursor={cursor}")
        else:
            command.extend(["--since=now", "-n", "0"])
        for unit in self.template.units:
            command.extend(["-u", unit])
        for identifier in self.template.identifiers:
            command.extend(["-t", identifier])
        if self.template.priority:
            command.extend(["-p", self.template.priority])
        return command

    async def run_once(self, stop: asyncio.Event) -> None:
        saved = self.cursor_store.load(self.source_key)
        cursor = saved.get("cursor") if saved else None
        command = self.command(str(cursor) if cursor else None)
        LOGGER.info("starting journal source: %s", " ".join(command))
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdout is not None
        try:
            while not stop.is_set():
                raw = await process.stdout.readline()
                if not raw:
                    break
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                message = item.get("MESSAGE")
                if isinstance(message, list):
                    message = " ".join(str(part) for part in message)
                if not isinstance(message, str) or not message.strip():
                    continue
                event_cursor = item.get("__CURSOR")
                if event_cursor:
                    self.cursor_store.save(
                        self.source_key,
                        {
                            "type": "journal",
                            "cursor": event_cursor,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                timestamp = datetime.now(timezone.utc)
                realtime = item.get("__REALTIME_TIMESTAMP")
                if realtime:
                    try:
                        timestamp = datetime.fromtimestamp(int(realtime) / 1_000_000, tz=timezone.utc)
                    except (ValueError, TypeError, OSError):
                        pass
                metadata = {
                    "logtype": "systemd",
                    "unit": item.get("_SYSTEMD_UNIT"),
                    "identifier": item.get("SYSLOG_IDENTIFIER"),
                    "priority": item.get("PRIORITY"),
                    "pid": item.get("_PID"),
                    "cursor": event_cursor,
                }
                source = f"journal:{metadata.get('unit') or metadata.get('identifier') or 'system'}"
                await self.emit(
                    LogEvent(
                        text=message[: self.main.max_line_bytes],
                        source=source,
                        observed_at=timestamp,
                        metadata=metadata,
                    ),
                    self.targets,
                )
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            stderr = b""
            if process.stderr is not None:
                stderr = await process.stderr.read()
            if stderr:
                LOGGER.warning("journalctl stderr: %s", stderr.decode("utf-8", errors="replace").strip())

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.run_once(stop)
            except FileNotFoundError:
                LOGGER.exception("journalctl not installed")
                return
            except Exception:
                LOGGER.exception("journal source crashed; retrying")
            if not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
