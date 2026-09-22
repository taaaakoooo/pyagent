"""Tests for the ``mcp`` configuration section."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config.config import (
    ConfigError,
    MCPConfig,
    MCPServerEntry,
    _parse_simple_yaml,
    load_config,
)


def _write(tmp: str, body: str) -> str:
    path = Path(tmp) / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


class MCPConfigDefaultsTests(unittest.TestCase):
    def test_defaults_when_section_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                _write(tmp, "agent:\n  max_turns: 5\n"), resolve_key=False
            )

        self.assertEqual(config.mcp, MCPConfig())
        self.assertTrue(config.mcp.enabled)
        self.assertTrue(config.mcp.autostart)
        self.assertEqual(config.mcp.servers, [])

    def test_enabled_servers_filters_disabled(self) -> None:
        config = MCPConfig(
            servers=[
                MCPServerEntry(name="a", command="a"),
                MCPServerEntry(name="b", command="b", enabled=False),
            ]
        )

        self.assertEqual([entry.name for entry in config.enabled_servers()], ["a"])

    def test_master_switch_disables_every_server(self) -> None:
        config = MCPConfig(
            enabled=False,
            servers=[MCPServerEntry(name="a", command="a")],
        )

        self.assertEqual(config.enabled_servers(), [])


class MCPConfigParsingTests(unittest.TestCase):
    def test_stdio_server_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                _write(
                    tmp,
                    "agent:\n"
                    "  max_turns: 5\n"
                    "mcp:\n"
                    "  enabled: true\n"
                    "  servers:\n"
                    "    - name: \"fs\"\n"
                    "      transport: \"stdio\"\n"
                    "      command: \"npx\"\n"
                    "      args:\n"
                    "        - \"-y\"\n"
                    "        - \"server-filesystem\"\n"
                    "      timeout_seconds: 30\n"
                    "      enabled: true\n",
                ),
                resolve_key=False,
            )

        self.assertEqual(len(config.mcp.servers), 1)
        server = config.mcp.servers[0]
        self.assertEqual(server.name, "fs")
        self.assertEqual(server.transport, "stdio")
        self.assertEqual(server.command, "npx")
        self.assertEqual(server.args, ["-y", "server-filesystem"])
        self.assertEqual(server.timeout_seconds, 30)
        self.assertTrue(server.enabled)

    def test_http_server_keeps_url_and_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                _write(
                    tmp,
                    "agent:\n"
                    "  max_turns: 5\n"
                    "mcp:\n"
                    "  servers:\n"
                    "    - name: \"remote\"\n"
                    "      transport: \"http\"\n"
                    "      url: \"http://127.0.0.1:8080/mcp\"\n",
                ),
                resolve_key=False,
            )

        server = config.mcp.servers[0]
        self.assertEqual(server.url, "http://127.0.0.1:8080/mcp")
        self.assertIsNone(server.command)

    def test_empty_servers_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                _write(
                    tmp,
                    "agent:\n  max_turns: 5\nmcp:\n  servers: []\n",
                ),
                resolve_key=False,
            )

        self.assertEqual(config.mcp.servers, [])


class MCPConfigValidationTests(unittest.TestCase):
    def _load(self, mcp_body: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            load_config(
                _write(tmp, "agent:\n  max_turns: 5\nmcp:\n" + mcp_body),
                resolve_key=False,
            )

    def test_duplicate_name_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(
                "  servers:\n"
                "    - name: \"dup\"\n"
                "      command: \"a\"\n"
                "    - name: \"dup\"\n"
                "      command: \"b\"\n"
            )

    def test_empty_name_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("  servers:\n    - command: \"a\"\n")

    def test_stdio_without_command_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("  servers:\n    - name: \"fs\"\n")

    def test_http_without_url_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(
                "  servers:\n"
                "    - name: \"remote\"\n"
                "      transport: \"http\"\n"
            )

    def test_unknown_transport_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(
                "  servers:\n"
                "    - name: \"weird\"\n"
                "      transport: \"carrier-pigeon\"\n"
                "      command: \"a\"\n"
            )

    def test_non_positive_timeout_raises(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(
                "  servers:\n"
                "    - name: \"fs\"\n"
                "      command: \"a\"\n"
                "      timeout_seconds: 0\n"
            )

    def test_servers_must_be_a_list(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("  servers: \"nope\"\n")

    def test_enabled_must_be_bool(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("  enabled: \"yes\"\n")


class SimpleYamlFallbackTests(unittest.TestCase):
    """The PyYAML-free parser must understand nested lists of mappings."""

    def test_list_of_mappings(self) -> None:
        parsed = _parse_simple_yaml(
            "mcp:\n"
            "  servers:\n"
            "    - name: fs\n"
            "      command: npx\n"
            "      args:\n"
            "        - \"-y\"\n"
            "        - pkg\n"
            "    - name: remote\n"
            "      transport: http\n"
        )

        servers = parsed["mcp"]["servers"]
        self.assertEqual(len(servers), 2)
        self.assertEqual(
            servers[0],
            {"name": "fs", "command": "npx", "args": ["-y", "pkg"]},
        )
        self.assertEqual(servers[1], {"name": "remote", "transport": "http"})

    def test_empty_containers(self) -> None:
        parsed = _parse_simple_yaml("a: []\nb: {}\n")

        self.assertEqual(parsed["a"], [])
        self.assertEqual(parsed["b"], {})

    def test_scalar_urls_are_not_mistaken_for_mappings(self) -> None:
        parsed = _parse_simple_yaml("urls:\n  - https://example.com/mcp\n")

        self.assertEqual(parsed["urls"], ["https://example.com/mcp"])


class SkillAlwaysVisibleSourceTests(unittest.TestCase):
    def test_default_includes_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                _write(tmp, "agent:\n  max_turns: 5\n"), resolve_key=False
            )

        self.assertEqual(config.skills.always_visible_sources, ["mcp"])

    def test_override_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(
                _write(
                    tmp,
                    "agent:\n"
                    "  max_turns: 5\n"
                    "skills:\n"
                    "  always_visible_sources:\n"
                    "    - \"local\"\n",
                ),
                resolve_key=False,
            )

        self.assertEqual(config.skills.always_visible_sources, ["local"])

    def test_invalid_source_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_config(
                    _write(
                        tmp,
                        "agent:\n"
                        "  max_turns: 5\n"
                        "skills:\n"
                        "  always_visible_sources:\n"
                        "    - \"telepathy\"\n",
                    ),
                    resolve_key=False,
                )


if __name__ == "__main__":
    unittest.main()
