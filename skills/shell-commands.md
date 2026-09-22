---
name: shell-commands
description: Running shell commands safely
keywords:
  - 命令
  - shell
  - run command
tools:
  - run_shell
  - read_disk_data
---
When running shell commands:

1. Start with the least destructive command that answers the question, such as
   listing a directory instead of moving files.
2. Never run recursive force deletes, privilege escalation, or anything that
   formats or repartitions storage without explicit confirmation.
3. If a command output is very large, read it back with `read_disk_data` instead
   of re-running the command.
4. Report the exact command you ran and its exit status.
