"""事件流重放、关闭和浏览器停止重连的边界。"""
import json
import shutil
import subprocess
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from ningsi_studio.http import sse


class EventLifecycleTests(unittest.TestCase):
    def test_closed_bus_does_not_retain_new_subscribers(self):
        bus = sse.EventBus('audit')
        bus.publish('finished', {'status': 'done'})
        bus.close()
        for _ in range(4):
            subscriber, count = bus.subscribe()
            self.assertEqual(count, 1)
            self.assertEqual(subscriber.events.get_nowait()['type'], 'finished')
            self.assertEqual(subscriber.events.get_nowait()['type'], 'closed')
        self.assertEqual(bus.subscriber_count(), 0)

    def test_live_publish_cannot_overtake_replay(self):
        bus = sse.EventBus('audit')
        bus.publish('phase', {'key': 'qc'})
        replay_entered, release_replay, published = (threading.Event() for _ in range(3))
        original_put = sse._Subscriber.put
        result = []

        def delayed_put(subscriber, event):
            if threading.current_thread().name == 'audit-replay' and event['id'] == 1:
                replay_entered.set()
                release_replay.wait(2)
            original_put(subscriber, event)

        def publish():
            bus.publish('phase', {'key': 'baseline_open'})
            published.set()

        with patch.object(sse._Subscriber, 'put', delayed_put):
            replay = threading.Thread(target=lambda: result.append(bus.subscribe(0)), name='audit-replay')
            replay.start()
            self.assertTrue(replay_entered.wait(1))
            live = threading.Thread(target=publish)
            live.start()
            try:
                self.assertFalse(published.wait(0.05), '实时事件不可抢在重放事件之前入队')
            finally:
                release_replay.set()
                replay.join(2)
                live.join(2)
        self.assertFalse(replay.is_alive())
        self.assertFalse(live.is_alive())
        subscriber, _ = result[0]
        self.assertEqual([subscriber.events.get_nowait()['id'] for _ in range(2)], [1, 2])
        bus.close()

    @unittest.skipUnless(shutil.which('node'), '需要 Node 运行前端模块')
    def test_browser_closed_event_stops_reconnect_while_connection_is_open(self):
        source = Path(__file__).resolve().parents[1] / 'web/js/sse.js'
        program = """
            import assert from 'node:assert/strict';
            globalThis.document = {querySelector: () => null};
            globalThis.window = {location: {search: ''}};
            let instance;
            globalThis.EventSource = class {
                readyState = 1;
                listeners = new Map();
                constructor() { instance = this; }
                addEventListener(name, callback) {
                    this.listeners.set(name, [...(this.listeners.get(name) || []), callback]);
                }
                removeEventListener() {}
                close() { this.readyState = 2; }
            };
            const {connectSessionEvents} = await import(URI);
            const stream = connectSessionEvents('audit');
            for (const callback of instance.listeners.get('closed')) callback({data: '{}'});
            assert.equal(instance.readyState, 2);
            stream.close();
        """.replace('URI', json.dumps(source.as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', program],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
