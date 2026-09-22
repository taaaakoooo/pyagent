"""Tests for skill loading, matching, and prompt rendering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from internal.skills import (
    Skill,
    SkillError,
    SkillManager,
    discover_skill_files,
    import_skills_to_system_prompt,
    load_skill_markdown,
    strip_skill_sections,
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

    def test_discover_skill_files_sorts_and_handles_missing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "b.md").write_text("x", encoding="utf-8")
            (directory / "a.md").write_text("x", encoding="utf-8")
            (directory / "notes.txt").write_text("x", encoding="utf-8")

            found = discover_skill_files(directory)

        self.assertEqual([path.name for path in found], ["a.md", "b.md"])
        self.assertEqual(discover_skill_files(Path(tmp) / "missing"), [])

    def test_load_directory_skips_broken_files(self) -> None:
        manager = SkillManager()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "good.md").write_text(
                "---\n"
                "name: good\n"
                "keywords:\n"
                "  - good\n"
                "tools:\n"
                "  - read_file\n"
                "---\n"
                "Good instructions.\n",
                encoding="utf-8",
            )
            (directory / "bad.md").write_text("# no front matter", encoding="utf-8")

            result = manager.load_directory(directory)

        self.assertEqual([skill.name for skill in result.skills], ["good"])
        self.assertEqual(len(result.errors), 1)
        self.assertIn("bad.md", next(iter(result.errors)))
        self.assertEqual([skill.name for skill in manager.list_skills()], ["good"])

    def test_unregister_replace_and_clear(self) -> None:
        manager = SkillManager()
        manager.register(Skill("alpha", ("one",), ("read_file",), "First."))

        self.assertTrue(manager.unregister("alpha"))
        self.assertFalse(manager.unregister("alpha"))
        self.assertEqual(manager.list_skills(), [])

        manager.replace(Skill("alpha", ("one",), ("read_file",), "First."))
        manager.replace(Skill("alpha", ("two",), ("write_file",), "Second."))
        self.assertEqual(len(manager.list_skills()), 1)
        self.assertEqual(manager.list_skills()[0].instructions, "Second.")

        manager.clear()
        self.assertEqual(manager.list_skills(), [])

    def test_strip_skill_sections_removes_injected_blocks(self) -> None:
        skills = [
            Skill("first", ("one",), ("read_file",), "First instructions."),
            Skill("second", ("two",), ("write_file",), "Second instructions."),
        ]
        rendered = import_skills_to_system_prompt("Base.", skills)

        stripped = strip_skill_sections(rendered)
        self.assertEqual(stripped, "Base.")
        self.assertNotIn("Imported Skills", stripped)
        self.assertEqual(strip_skill_sections("plain text"), "plain text")

    def test_reinjection_after_removal_leaves_no_residue(self) -> None:
        first = Skill("first", ("one",), ("read_file",), "First instructions.")
        second = Skill("second", ("two",), ("write_file",), "Second instructions.")

        rendered = import_skills_to_system_prompt("Base.", [first, second])
        rendered = import_skills_to_system_prompt(rendered, [first])

        self.assertIn("Skill: first", rendered)
        self.assertNotIn("Skill: second", rendered)
        self.assertEqual(rendered.count("Imported Skills"), 1)


if __name__ == "__main__":
    unittest.main()
