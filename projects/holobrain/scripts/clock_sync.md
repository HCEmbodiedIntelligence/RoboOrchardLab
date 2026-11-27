# Robot/client clock synchronization

ROS feedback and control ownership timestamps require a common clock. Keep the
robot's wired NTP server as the reference; do not mix independent time servers.
Only one time synchronization service should adjust the inference computer.

For systemd-timesyncd, install a local drop-in under
`/etc/systemd/timesyncd.conf.d/robot-ntp.conf`:

```ini
[Time]
NTP=
NTP=192.168.2.33
FallbackNTP=
PollIntervalMinSec=16
PollIntervalMaxSec=32
ConnectionRetrySec=5
```

Use the actual robot wired IP if different. Close robot output and stop the
inference client before restarting synchronization or making any large clock
correction. Do not step clocks while controlling hardware. Keep a backup of the
previous configuration. A displayed synchronized flag alone is not sufficient;
verify measured offset with the checker and ROS source timestamp diagnostics.

`start_hc_tj_ros_local.sh` runs the read-only clock checker before loading the
model and again immediately before starting the client. It requires at least
three consecutive good samples spanning two seconds, with absolute NTP offset
at most 10 ms and round-trip time at most 20 ms. After 30 seconds without stable
results it fails closed. `HC_TJ_NTP_SERVER` selects the robot's numeric wired
IPv4 address. The check neither adjusts time nor enables robot output.

Manual check:

```sh
python projects/holobrain/scripts/hc_tj_clock_check.py --server 192.168.2.33
timedatectl timesync-status
```

The launch check is not a replacement for runtime safety. The client logs
`ownership_timing` and `feedback_timing` with source age and timestamps. After
an AUTO grant, ownership timing violations pause execution and clear old
trajectory state, using the same recovery path as feedback timing faults.
Ordinary heartbeats cannot resume execution: after feedback is stable for one
second, manually take over and release to a new AUTO grant. Replan from fresh
observations. Other non-timing safety faults remain latched.

Directly launching the Python client bypasses the shell preflight, but does not
bypass runtime freshness checks. Use the standard launcher for model control.
