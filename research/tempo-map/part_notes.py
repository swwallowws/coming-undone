"""Where each part's notes fall in a frozen demo: notes per 5 s and the pitch range.
Usage: part_notes.py <try-dist>/try/data.json"""
import json
import sys

data = json.load(open(sys.argv[1]))
for p in data["parts"]:
    counts = [0] * (int(data["duration"] // 5) + 1)
    for s, e, q, v in p["notes"]:
        counts[int(s // 5)] += 1
    ps = [n[2] for n in p["notes"]] or [0]
    print(f"{p['name']:>8} {len(p['notes']):4d} notes, pitch {min(ps)}-{max(ps)}, per 5 s: {counts}")
    if "--pitches" in sys.argv:
        print("         pitches:", sorted(ps))
