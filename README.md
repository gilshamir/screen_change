# Four-circle screen monitor

This Windows script watches all connected displays for four similarly sized
circles arranged in a cross. It searches across different pattern sizes and
intensities, including single-textured and all-textured Gabor patterns, then
sends `Big`, `Small`, or `Tiny` when the pattern first appears. Detection
matches the circle boundaries and also recognizes four Gabor circle blobs.

## Setup

Use Python 3.10 or later:

```powershell
py -m pip install -r requirements.txt
```

## Run

Replace the example address and port with the TCP server that should receive
the notification:

```powershell
py screen_monitor.py --host 127.0.0.1 --port 5000
```

Each detection opens a TCP connection and sends
`E;1;Konan;1;;;;Gabor;<PatternId>;<PatternName>\r\n`. IDs are stable by type:
`1` is `Tiny`, `2` is `Small`, and `3` is `Big`. The connection then closes.
`Big` means the detected
circle radius is at least 10% of the shorter screen dimension, `Small` is at
least 3.5%, and smaller patterns are named `Tiny`. TCP is a byte stream, so the receiving server
should use the newline as the message delimiter rather than expecting one
network packet per message.

The script sends only once while a pattern remains visible. It becomes ready
to notify again after one consecutive scan without a match. Press Ctrl+C to
stop it.

Options:

- `--threshold VALUE`: adjust boundary-match strictness (default `0.82`, range `(0, 1]`).
- `--interval SECONDS`: minimum delay between scans (default `0.02`).
- `--rearm-frames COUNT`: consecutive misses required before another
  notification (default `1`).

The detector checks only the exact known radii: 28 (Tiny), 56 (Small), and
196 (Big).  It scales those radii once to match `search_scale`; it does not
search a range of intermediate sizes.  Custom radii can be passed to
`detect_pattern(pattern_radii=(...))` in Python.

If the intended pattern is not detected, try a lower `--threshold`; if other
screen content triggers notifications, try a higher one.

## Test

```powershell
py -m unittest
```
