"""Tests for skill loading, matching, and prompt rendering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from internal.skills import (
    Skill,
    SkillError,
    SkillManager,
    import_skills_to_system_prompt,
    load_skill_markdown,
)


class SkillTests(unittest.TestCase):
    def test_matches_english_case_phrase_and_word_boundary(self) -> None:
        manager = SkillManager(
            [
                Skill(
                    name="python",
                    keywords=("Python", "read file"),
                    tools=("read_file",),
                    instructions="Inspect Python files.",
                )
            ]
        )

        self.assertEqual(manager.match("Fix this PYTHON module").skill_names, ("python",))
        self.assertEqual(manager.match("Please read   file now").skill_names, ("python",))
        self.assertEqual(manager.match("pythonic style").skill_names, ())

    def test_matches_chinese_substring_and_unions_tools_in_skill_order(self) -> None:
        manager = SkillManager(
            [
                Skill(
                    name="files",
                    keywords=("文件",),
                    tools=("read_file", "write_file"),
                    instructions="Handle files.",
                ),
                Skill(
                    name="search",
                    keywords=("搜索",),
                    tools=("read_file", "read_regex"),
                    instructions="Search files.",
                ),
            ]
        )

        match = manager.match("搜索并修改文件")
        self.assertEqual(match.skill_names, ("files", "search"))
        self.assertEqual(
            match.tool_names,
            ("read_file", "write_file", "read_regex"),
        )

    def test_loads_markdown_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "files.md"
            path.write_text(
                "---\n"
                "name: files\n"
                "description: File workflow\n"
                "keywords:\n"
                "  - 文件\n"
                "tools:\n"
                "  - read_file\n"
                "---\n"
                "Always inspect before editing.\n",
                encoding="utf-8",
            )
            skill = load_skill_markdown(path)

        self.assertEqual(skill.name, "files")
        self.assertEqual(skill.keywords, ("文件",))
        self.assertEqual(skill.tools, ("read_file",))
        self.assertEqual(skill.instructions, "Always inspect before editing.")

    def test_rejects_invalid_markdown_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.md"
            path.write_text("# no front matter", encoding="utf-8")
            with self.assertRaises(SkillError):
                load_skill_markdown(path)

    def test_prompt_rendering_is_stable_and_idempotent(self) -> None:
        skills = [
            Skill(
                name="first",
                keywords=("one",),
                tools=("read_file",),
                instructions="First instructions.",
            ),
            Skill(
                name="second",
                keywords=("two",),
                tools=("write_file",),
                instructions="Second instructions.",
            ),
        ]
        first = import_skills_to_system_prompt("Base prompt.", skills)
        second = import_skills_to_system_prompt(first, skills)

        self.assertEqual(first, second)
        self.assertLess(first.index("Skill: first"), first.index("Skill: second"))
        self.assertEqual(first.count("<!-- pyagent-skill:first -->"), 1)

    def test_duplicate_name_with_different_content_is_rejected(self) -> None:
        manager = SkillManager()
        manager.register(
            Skill("same", ("one",), ("read_file",), "First.")
        )
        with self.assertRaises(SkillError):
            manager.register(
                Skill("same", ("two",), ("write_file",), "Second.")
            )


if __name__ == "__main__":
    unittest.main()
