"""CLI entry point for the coding agent."""

from __future__ import annotations

from internal.agent import create_agent


def main() -> None:
    agent = create_agent()
    if agent.restored_message_count > 0:
        print(f"Restored {agent.restored_message_count} message(s) from session.")
    print("PyAgent ready. Type 'exit' to quit.")

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            answer = agent.chat(user_input)
            if answer and not agent.state.is_streaming:
                print()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
