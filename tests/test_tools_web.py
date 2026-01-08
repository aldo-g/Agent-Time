import json
import unittest
from unittest.mock import patch

from agent.tools import web as tools


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict | None = None) -> None:
        self._body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _Result:
    def __init__(self, title: str, url: str, snippet: str) -> None:
        self.title = title
        self.url = url
        self.snippet = snippet


class TestWebTools(unittest.TestCase):
    def test_run_search(self) -> None:
        results = [_Result("Title", "https://example.com", "Snippet")]
        with patch("agent.tools.web.search_web", return_value=results):
            output = tools._run_search("query", limit=1)
        self.assertIn("1. Title", output)
        self.assertIn("https://example.com", output)

    def test_run_rss_fetch(self) -> None:
        xml = b"""<?xml version="1.0"?>
        <rss><channel>
            <item><title>News</title><link>https://example.com</link><pubDate>Today</pubDate></item>
        </channel></rss>
        """
        with patch("urllib.request.urlopen", return_value=_FakeResponse(xml)):
            output = tools._run_rss_fetch(limit=1, sources="https://feed.example.com")
        self.assertIn("1. News", output)
        self.assertIn("https://example.com", output)

    def test_run_bluesky_search(self) -> None:
        payload = {
            "posts": [
                {
                    "record": {"text": "Hello world"},
                    "author": {"handle": "tester"},
                    "uri": "at://example/1",
                }
            ]
        }
        body = json.dumps(payload).encode("utf-8")
        with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
            output = tools._run_bluesky_search("hello", limit=1)
        self.assertIn("Hello world (@tester)", output)
        self.assertIn("at://example/1", output)

    def test_run_web_scrape_html(self) -> None:
        html = b"<html><body>Hello <b>World</b></body></html>"
        headers = {"Content-Type": "text/html"}
        with patch("urllib.request.urlopen", return_value=_FakeResponse(html, headers=headers)):
            output = tools._run_web_scrape("https://example.com", max_chars=2000)
        self.assertEqual(output, "Hello World")

    def test_run_notebook_eval(self) -> None:
        output = tools._run_notebook_eval("result = 2 + 2\nprint('hi')")
        self.assertIn("Result: 4", output)
        self.assertIn("Output:\nhi", output)


if __name__ == "__main__":
    unittest.main()
