import urllib.request
import json
import time

def post(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    return json.loads(urllib.request.urlopen(req).read())

BASE = 'http://localhost:8001'

# Select config_1 (7MB world with real pulses)
r = post(BASE + '/api/tsrd/select_world', {'world_id': 'config_1'})
print('Select:', r)

# Start
r2 = post(BASE + '/api/tsrd/start', {})
print('Start:', r2)

time.sleep(2.0)

s = json.loads(urllib.request.urlopen(BASE + '/api/tsrd/status').read())
print(f"Status: {s['status']} | slot: {s['slot']}/{s['total_slots']} | hits: {s['total_hits']} | misses: {s['total_misses']} | HR: {round(s['hit_rate']*100,1)}% | reward: {s['cumulative_reward']}")

# Stop
post(BASE + '/api/tsrd/stop', {})
print('Done - test PASSED')
