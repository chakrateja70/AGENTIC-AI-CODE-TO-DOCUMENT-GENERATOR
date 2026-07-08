from __future__ import annotations

from src.graph.orchestrator import run


def main() -> None:
    print("Code-Document-Generator agent demo. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        print(f"Agent: {run(query)}")


if __name__ == "__main__":
    main()
