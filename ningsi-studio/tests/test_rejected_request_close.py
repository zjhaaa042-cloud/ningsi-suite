"""Refused origins get a response before bounded, non-dispatching TCP cleanup."""

from email.message import Message
import http.client
import io
import json
import socket
import threading
import unittest
from unittest.mock import Mock, patch

from ningsi_studio.http.router import Forbidden, Response, Router
from ningsi_studio.http import server as http_server
from ningsi_studio.http.server import AppServer, StudioRequestHandler
from ningsi_studio.settings import Settings


class RejectedBodyCleanupTests(unittest.TestCase):
    def handler(self, lengths, data=b"", transfer=None):
        handler = object.__new__(StudioRequestHandler)
        handler.headers = Message()
        for length in lengths:
            handler.headers["Content-Length"] = length
        if transfer is not None:
            handler.headers["Transfer-Encoding"] = transfer
        handler.path = "/api/models/human/import"
        handler.connection = Mock()
        handler.connection.gettimeout.return_value = 3
        handler.rfile = io.BytesIO(data)
        handler.close_connection = True
        return handler

    def test_unknown_ambiguous_and_excessive_framing_is_never_drained(self):
        for lengths, transfer in [
            ([], None), ([""], None), (["-1"], None), (["+1"], None),
            (["1", "1"], None), (["abc"], None), (["１"], None),
            ([str(http_server.MAX_BODY_BYTES + 1)], None), (["9" * 5000], None),
            (["1"], "chunked"), ([], "chunked"), (["0"], None),
        ]:
            with self.subTest(lengths=str(lengths)[:50], transfer=transfer):
                handler = self.handler(lengths, transfer=transfer)
                handler.rfile = Mock()
                handler.rfile.read1.side_effect = AssertionError("Unknown framing must not be read")
                handler._discard_rejected_body()
                handler.rfile.read1.assert_not_called()
                handler.connection.settimeout.assert_not_called()
                self.assertTrue(handler.close_connection)

    def test_known_body_is_discarded_in_chunks_without_parsing_or_eating_pipeline(self):
        handler = self.handler(["150000"], b"x" * 150000 + b"POST /api/write HTTP/1.1\r\n")
        original = handler.rfile.read1
        handler.rfile = Mock(wraps=handler.rfile)
        handler.rfile.read1.side_effect = original
        handler._discard_rejected_body()
        self.assertEqual(handler.rfile.read(), b"POST /api/write HTTP/1.1\r\n")
        self.assertTrue(all(0 < call.args[0] <= 65536 for call in handler.rfile.read1.call_args_list))
        handler.connection.settimeout.assert_called_with(3)

    def test_cleanup_has_a_single_short_deadline_and_restores_socket_timeout(self):
        for body_timeout, times, reads in [(10, [5, 5.05, 5.2, 5.3], 2), (.1, [5, 5.01, 5.11], 1)]:
            with self.subTest(body_timeout=body_timeout):
                handler = self.handler(["10"])
                handler.rfile = Mock()
                handler.rfile.read1.return_value = b"x"
                with patch.object(http_server, "BODY_TIMEOUT_SEC", body_timeout), \
                     patch.object(http_server.time, "monotonic", side_effect=times):
                    handler._discard_rejected_body()
                self.assertEqual(handler.rfile.read1.call_count, reads)
                bounded = [call.args[0] for call in handler.connection.settimeout.call_args_list[:-1]]
                self.assertTrue(all(0 < value <= min(body_timeout, http_server.REJECT_DRAIN_TIMEOUT_SEC)
                                    for value in bounded))
                handler.connection.settimeout.assert_called_with(3)
        self.assertLessEqual(http_server.REJECT_DRAIN_TIMEOUT_SEC, .25)

    def test_refusal_is_sent_flushed_and_half_closed_before_discarding(self):
        handler = self.handler(["1"], b"x")
        handler.close_connection = False
        operations = []
        handler._send = Mock(side_effect=lambda response: operations.append("response"))
        handler.wfile = Mock()
        handler.wfile.flush.side_effect = lambda: operations.append("flush")
        handler.connection.shutdown.side_effect = lambda direction: operations.append(("shutdown", direction))
        handler._discard_rejected_body = Mock(side_effect=lambda: operations.append("discard"))
        handler._reject_origin(Forbidden("refused"))
        self.assertEqual(operations, ["response", "flush", ("shutdown", socket.SHUT_WR), "discard"])
        response = handler._send.call_args.args[0]
        self.assertEqual(response.status, 403)
        self.assertEqual(response.headers["Connection"], "close")
        self.assertEqual(json.loads(response.body)["error"]["code"], "forbidden")
        self.assertTrue(handler.close_connection)


class RejectedRequestSocketTests(unittest.TestCase):
    def setUp(self):
        self.writes = []
        router = Router()
        router.post("/api/write")(lambda request: self.writes.append(request.body) or Response.json({}))
        self.server = AppServer(router, Settings(port=0))
        self.server.start_background()
        self.addCleanup(self.server.stop)

    def headers(self, method, origin, size):
        return (f"{method} /api/write HTTP/1.1\r\nHost: 127.0.0.1:{self.server.port}\r\n"
                f"Origin: {origin}\r\nContent-Length: {size}\r\n\r\n").encode()

    def response(self, connection):
        response = http.client.HTTPResponse(connection)
        response.begin()
        self.assertEqual(response.status, 403)
        self.assertEqual(response.getheader("Connection"), "close")
        self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
        self.assertEqual(json.loads(response.read())["error"]["code"], "forbidden")

    def test_headers_and_small_body_sent_separately_still_receive_all_refusals(self):
        original = StudioRequestHandler._check_request_origin
        body = b'{"auto_id":true}'
        origins = ("http://untrusted.invalid", "null", "http://localhost:1",
                   f"http://127.0.0.1:{self.server.port}/path")
        for origin in origins:
            for method in ("GET", "POST", "OPTIONS"):
                with self.subTest(origin=origin, method=method):
                    ready, body_sent = threading.Event(), threading.Event()

                    def separated(handler):
                        try:
                            return original(handler)
                        except Forbidden:
                            ready.set()
                            if not body_sent.wait(2):
                                raise AssertionError("Body must be released after headers are checked")
                            if not handler.connection.recv(1, socket.MSG_PEEK):
                                raise AssertionError("Body must be pending without being consumed")
                            raise

                    with patch.object(StudioRequestHandler, "_check_request_origin", separated), \
                         socket.create_connection(("127.0.0.1", self.server.port), timeout=2) as connection:
                        connection.sendall(self.headers(method, origin, len(body)))
                        self.assertTrue(ready.wait(2))
                        connection.sendall(body)
                        body_sent.set()
                        self.response(connection)
        self.assertEqual(self.writes, [])

    def test_a_pipelined_legitimate_write_is_not_executed_after_a_refusal(self):
        closed = threading.Event()
        original = StudioRequestHandler.finish

        def finished(handler):
            try:
                return original(handler)
            finally:
                closed.set()

        following = (f"POST /api/write HTTP/1.1\r\nHost: 127.0.0.1:{self.server.port}\r\n"
                     "Content-Length: 1\r\n\r\ny").encode()
        with patch.object(StudioRequestHandler, "finish", finished), \
             socket.create_connection(("127.0.0.1", self.server.port), timeout=2) as connection:
            connection.sendall(self.headers("POST", "http://untrusted.invalid", 1) + b"x" + following)
            self.response(connection)
            self.assertTrue(closed.wait(2))
        self.assertEqual(self.writes, [])

    def test_refusal_does_not_wait_for_a_missing_body_and_cleanup_finishes(self):
        closed = threading.Event()
        original = StudioRequestHandler.finish

        def finished(handler):
            try:
                return original(handler)
            finally:
                closed.set()

        with patch.object(StudioRequestHandler, "finish", finished), \
             socket.create_connection(("127.0.0.1", self.server.port), timeout=2) as connection:
            connection.sendall(self.headers("POST", "http://untrusted.invalid", 14))
            self.response(connection)
            self.assertTrue(closed.wait(2), "Cleanup must finish while the sender still leaves the body absent")
        self.assertEqual(self.writes, [])


if __name__ == "__main__":
    unittest.main()
