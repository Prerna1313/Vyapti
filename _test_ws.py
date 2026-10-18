import asyncio
import websockets
import json

async def test_ws():
    uri = 'ws://localhost:8001/ws/tsrd'
    async with websockets.connect(uri) as ws:
        ack = await ws.recv()
        ack_data = json.loads(ack)
        print('Ack received:', ack_data.get('type'), 'Worlds:', len(ack_data.get('worlds', [])))
        
        await ws.send(json.dumps({'command': 'select_world', 'world_id': 'config_1'}))
        sel_msg = await ws.recv()
        print('Select msg:', json.loads(sel_msg).get('type'))
        
        await ws.send(json.dumps({'command': 'start'}))
        for _ in range(5):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Telemetry: slot={data.get('slot')} action={data.get('action')} hit={data.get('hit')} HR={round(data.get('hit_rate', 0)*100, 1)}%")
        await ws.send(json.dumps({'command': 'stop'}))
        print('WebSocket test completed successfully!')

if __name__ == '__main__':
    asyncio.run(test_ws())
