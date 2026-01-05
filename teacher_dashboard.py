import json
import csv
from collections import defaultdict

def parse_mmss(mmss):
    if not mmss:
        return None
    try:
        m, s = mmss.split(':')
        return int(m) * 60 + int(s)
    except Exception:
        return None

def main():
    events = []
    try:
        with open('events_log.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    events.append(obj)
                except Exception:
                    pass
    except FileNotFoundError:
        return

    buckets = defaultdict(lambda: defaultdict(int))
    for e in events:
        ts = parse_mmss(e.get('timestamp'))
        if ts is None:
            continue
        minute = ts // 60
        emotion = e.get('emotion')
        buckets[minute][emotion] += 1

    with open('timeline_heatmap.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['minute','confused','sleepy','disengaged','sad','neutral','happy'])
        for m in sorted(buckets.keys()):
            row = [m]
            for label in ['confused','sleepy','disengaged','sad','neutral','happy']:
                row.append(buckets[m].get(label, 0))
            writer.writerow(row)

if __name__ == '__main__':
    main()
