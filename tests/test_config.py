"""Tests for configuration loading, including the skills section."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config.config import (
    AppConfig,
    ConfigError,
    SessionConfig,
    SkillConfig,
    load_config,
)


class SkillConfigTests(unittest.TestCase):
    def test_defaults_when_section_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("agent:\n  max_turns: 5\n", encoding="utf-8")
            config = load_config(str(path), resolve_key=False)

        self.assertEqual(config.skills, SkillConfig())
        self.assertTrue(config.skills.enabled)
        self.assertEqual(config.skills.dir, "skills")
        self.assertEqual(config.skills.files, [])
        self.assertTrue(config.skills.inject_system_prompt)
        # Never default to "none": a zero-tool turn makes models invent
        # tool-call syntax as plain text.
        self.assertEqual(config.skills.unmatched_tools, "readonly")
        self.assertEqual(config.skills.unmatched_tools, SkillConfig().unmatched_tools)

    def test_section_overrides_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "agent:\n"
                "  max_turns: 5\n"
                "skills:\n"
                "  enabled: false\n"
                "  dir: \"custom\"\n"
                "  inject_system_prompt: false\n"
                "  unmatched_tools: \"readonly\"\n",
                encoding="utf-8",
            )
            config = load_config(str(path), resolve_key=False)

        self.assertFalse(config.skills.enabled)
        self.assertEqual(config.skills.dir, "custom")
        self.assertFalse(config.skills.inject_system_prompt)
        self.assertEqual(config.skills.unmatched_tools, "readonly")

    def test_files_list_is_parsed_without_pyyaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "agent:\n"
                "  max_turns: 5\n"
                "skills:\n"
                "  files:\n"
                "    - skills/alpha.md\n"
                "    - skills/beta.md\n",
                encoding="utf-8",
            )
            config = load_config(str(path), resolve_key=False)

        self.assertEqual(
            config.skills.files,
            ["skills/alpha.md", "skills/beta.md"],
        )

    def test_invalid_unmatched_tools_raises(self) -> None:
        config = AppConfig()
        config.skills = SkillConfig(unmatched_tools="bogus")
        with self.assertRaises(ConfigError):
            config.validate()

    def test_files_as_string_raises(self) -> None:
        config = AppConfig()
        config.skills = SkillConfig(files="skills/a.md")
        with self.assertRaises(ConfigError):
            config.validate()

    def test_enabled_must_be_bool(self) -> None:
        config = AppConfig()
        config.skills = SkillConfig(enabled="yes")
        with self.assertRaises(ConfigError):
            config.validate()


class SessionConfigTests(unittest.TestCase):
    def test_defaults_when_section_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("agent:\n  max_turns: 5\n", encoding="utf-8")
            config = load_config(str(path), resolve_key=False)

        self.assertEqual(config.session.storage_dir, ".sessions")
        self.assertEqual(config.session.default_id, "default")
        self.assertTrue(config.session.isolate_tool_output)

    def test_section_overrides_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "agent:\n"
                "  max_turns: 5\n"
                "session:\n"
                "  storage_dir: \"state\"\n"
                "  default_id: \"work\"\n"
                "  isolate_tool_output: false\n",
                encoding="utf-8",
            )
            config = load_config(str(path), resolve_key=False)

        self.assertEqual(config.session.storage_dir, "state")
        self.assertEqual(config.session.default_id, "work")
        self.assertFalse(config.session.isolate_tool_output)

    def test_invalid_default_id_raises(self) -> None:
        config = AppConfig()
        config.session = SessionConfig(default_id="../escape")
        with self.assertRaises(ConfigError):
            config.validate()

    def test_isolate_tool_output_must_be_bool(self) -> None:
        config = AppConfig()
        config.session = SessionConfig(isolate_tool_output="yes")
        with self.assertRaises(ConfigError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
