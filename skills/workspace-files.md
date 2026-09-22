---
name: workspace-files
description: Reading and editing files inside the workspace
keywords:
  - 文件
  - read file
  - edit file
tools:
  - read_file
  - list_dir
  - write_file
  - replace_line
---
When working with files in the workspace:

1. Always read the current contents before editing, so your edit matches reality.
2. Prefer `replace_line` over `write_file` when changing a few lines; it keeps the
   rest of the file intact and produces a smaller diff.
3. Use workspace-relative paths. Never try to escape the workspace root.
4. After a write, summarize exactly which file changed and why.
