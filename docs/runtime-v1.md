# alfilogd runtime v1

`alfilogd` keeps the local BGE encoder and the v3 fixed-head classifier warm in one systemd service. It reads configured files or journald streams, batches log messages, classifies them once, routes results to facilities, expires old errors, and writes Nagios-readable state files.

## Runtime path

```text
file / journald / fallback poll
             |
             v
       shared source reader
             |
             v
       micro-batch queue
             |
             v
 BGE encoder + v3 classifier
             |
             v
 domain routing + facility state
             |
             v
 /var/alfilogd/<output>/status.json
 /var/alfilogd/<output>/get.sh
```

Identical sources are deduplicated. If CPU, memory and filesystem facilities all read `/var/log/messages`, each line is encoded once and then routed by the predicted domain.

## Main defaults

```ini
[main]
embeddingmodel=/opt/alfilogd/models/bge-small-en-v1.5
classifier=/opt/alfilogd/models/micro-log-state-v3.pt
pollingrate=5m
polling_lines=500
recovery_time=60m
```

Every facility can override `pollingrate`, `polling_lines`, `recovery_time`, `critical_after`, `warning_after`, `recover_on_ok`, and `dedupe_time`.

## File source with inotify

```ini
[cpu]
logtype=file
logfile=/var/log/messages
polling=inotify
domain=cpu
output=cpu
```

The reader stores inode and byte offset under `/var/lib/alfilogd/cursors`. On rotation, it drains the old file descriptor and starts the new inode at byte zero.

## Journald source

```ini
[apache]
logtype=systemd
polling=systemd_event
units=httpd.service,apache2.service
domain=httpd
output=httpd
```

`httpd`, `apache`, `nginx`, `app`, and `application` are aliases for the trained `SERVICE` domain. Journald cursors are persisted.

## Dumb fallback polling

```ini
[legacy]
logtype=file
logfile=/srv/legacy/app.log
polling=poll
pollingrate=5m
polling_lines=800
domain=service
output=legacy
```

The poll backend reads the last N lines and uses a persistent TTL dedupe cache so the same tail does not refresh an old incident forever.

## Recovery semantics

A BAD result creates or refreshes an active error. An accepted OK result clears active errors when `recover_on_ok=true`. Independently, every active error disappears after `recovery_time`. Once no active errors remain, the facility returns OK.

The daemon refreshes `daemon_updated_at` in every status file. `get.sh` returns UNKNOWN if the status heartbeat is stale. Active errors, counters and timestamps are also persisted under `/var/lib/alfilogd/facilities`, so a daemon restart does not magically erase a current incident.

## NRPE

On startup the daemon creates:

```text
/var/lib/alfilogd/generated/alfilogd.cfg
/var/lib/alfilogd/generated/install-nrpe-addon.sh
```

The generated NRPE file points only to the tiny status readers. NRPE never loads PyTorch.

```bash
sudo /var/lib/alfilogd/generated/install-nrpe-addon.sh
```

## CLI

```bash
alfilogctl ping --json
alfilogctl classify --json "high load"
alfilogctl classify --stdin --json < sample.log
alfilogctl status --facility cpu --nagios
/var/alfilogd/cpu/get.sh
```

## Configuration changes

After adding or changing a file under `/etc/alfilogd/conf.d`, restart the daemon:

```bash
sudo systemctl restart alfilogd
```

The NRPE assets are regenerated on every daemon start. Run the generated admin installer again after facilities change.
