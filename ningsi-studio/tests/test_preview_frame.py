"""Preview frames keep the same source and rate across asynchronous discovery."""
import threading
import unittest
from unittest.mock import patch

import numpy as np
from ningsi_studio.core.live_source import PreviewStream, SourceInfo


class BufferedEngine:
    srate = 500
    channels = 1

    def window(self, state, seconds):
        return np.ones((1, 500))


class PreviewFrameTests(unittest.TestCase):
    def test_frame_duration_and_quality_use_the_discovered_rate(self):
        preview = PreviewStream()
        source = SourceInfo('lsl:a', 'lsl', 250, 1, 'a', '', BufferedEngine())
        with patch('ningsi_studio.core.live_source.build_source', return_value=source):
            preview.start(source.key)
        frame = preview.window()
        self.assertEqual(frame['srate'], 500)
        self.assertEqual(frame['seconds'], 1)
        self.assertEqual(frame['quality']['reasons'], ['insufficient_samples'])
        self.assertEqual(frame['quality']['metrics']['required'], 1000)

    def test_source_switched_during_read_does_not_publish_old_wave_or_quality(self):
        preview = PreviewStream()
        entered, release = threading.Event(), threading.Event()

        class Slow(BufferedEngine):
            def window(self, state, seconds):
                entered.set()
                if not release.wait(3):
                    raise TimeoutError('test window did not release')
                return super().window(state, seconds)

        initial = SourceInfo('lsl:a', 'lsl', 250, 1, 'a', '', Slow())
        replacement = SourceInfo('lsl:b', 'lsl', 500, 1, 'b', '', BufferedEngine())
        with patch('ningsi_studio.core.live_source.build_source', side_effect=[initial, replacement]):
            preview.start(initial.key)
            result = []
            reader = threading.Thread(target=lambda: result.append(preview.window()))
            reader.start()
            try:
                self.assertTrue(entered.wait(2))
                preview.start(replacement.key)
            finally:
                release.set()
                reader.join(3)
        self.assertEqual(result[0]['source'], replacement.key)
        self.assertEqual(result[0]['samples'], [])
        self.assertNotIn('quality', result[0])
