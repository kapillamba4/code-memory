"""Tests for the main() entrypoint CLI argument parsing."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from code_memory import server as server_mod
from code_memory.server import build_arg_parser


class TestMainArgParsing:
    """Tests for CLI argument parsing in main()."""

    def _parse_args(self, argv: list[str]) -> argparse.Namespace:
        """Parse args using the real CLI parser from server.py."""
        return build_arg_parser().parse_args(argv)

    def test_default_transport_is_stdio(self):
        """Default transport should be stdio."""
        args = self._parse_args([])
        assert args.transport == "stdio"

    def test_default_port_is_8765(self):
        """Default SSE port should be 8765."""
        args = self._parse_args([])
        assert args.port == 8765

    def test_default_host_is_localhost(self):
        """Default SSE host should be 127.0.0.1."""
        args = self._parse_args([])
        assert args.host == "127.0.0.1"

    def test_sse_transport_flag(self):
        """--transport sse should set transport to sse."""
        args = self._parse_args(["--transport", "sse"])
        assert args.transport == "sse"

    def test_custom_port(self):
        """--port should accept a custom port number."""
        args = self._parse_args(["--transport", "sse", "--port", "9000"])
        assert args.port == 9000

    def test_custom_host(self):
        """--host should accept a custom host."""
        args = self._parse_args(["--transport", "sse", "--host", "0.0.0.0"])
        assert args.host == "0.0.0.0"

    def test_invalid_transport_raises_error(self):
        """Invalid transport should raise SystemExit."""
        with pytest.raises(SystemExit):
            self._parse_args(["--transport", "invalid"])


class TestMainRunsBehavior:
    """Tests that main() calls mcp.run() with the correct arguments."""

    def test_stdio_calls_run_with_stdio(self):
        """main() with no args should call mcp.run(transport='stdio')."""
        with patch("sys.argv", ["code-memory"]):
            with patch.object(server_mod, "mcp") as mock_mcp:
                mock_mcp.settings = MagicMock()
                server_mod.main()
                mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_sse_calls_run_with_sse(self):
        """main() with --transport sse should call mcp.run(transport='sse')."""
        with patch("sys.argv", ["code-memory", "--transport", "sse"]):
            with patch.object(server_mod, "mcp") as mock_mcp:
                mock_mcp.settings = MagicMock()
                server_mod.main()
                mock_mcp.run.assert_called_once_with(transport="sse")

    def test_sse_sets_port_on_settings(self):
        """main() with --transport sse --port 9000 should set mcp.settings.port."""
        with patch("sys.argv", ["code-memory", "--transport", "sse", "--port", "9000"]):
            with patch.object(server_mod, "mcp") as mock_mcp:
                mock_mcp.settings = MagicMock()
                server_mod.main()
                assert mock_mcp.settings.port == 9000

    def test_sse_sets_host_on_settings(self):
        """main() with --transport sse --host 0.0.0.0 should set mcp.settings.host."""
        with patch("sys.argv", ["code-memory", "--transport", "sse", "--host", "0.0.0.0"]):
            with patch.object(server_mod, "mcp") as mock_mcp:
                mock_mcp.settings = MagicMock()
                server_mod.main()
                assert mock_mcp.settings.host == "0.0.0.0"
