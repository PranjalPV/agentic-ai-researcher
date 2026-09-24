import sys
import time
import argparse
from dotenv import load_dotenv
from crew import run_research

load_dotenv()

BANNER = r"""
====================================================================
          AGENTIC AI ACADEMIC RESEARCHER 2.0
   Autonomous Literature Scout, Ingestion & Comparative Engine
====================================================================
"""


def main():
    parser = argparse.ArgumentParser(description="Autonomous Academic Multi-Agent Researcher")
    parser.add_argument("--query", "-q", type=str, help="Research topic or question to investigate")
    parser.add_argument("--no-save", action="store_true", help="Do not save the generated report to disk")
    args = parser.parse_args()

    print(BANNER)

    query = args.query
    if not query:
        try:
            query = input("Enter Research Topic / Hypothesis: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user.")
            sys.exit(0)

    if not query:
        print("Error: Research query cannot be empty.")
        sys.exit(1)

    print(f"\n[*] Commencing investigation on: '{query}'")
    start_time = time.time()

    try:
        result = run_research(query=query, save_report=not args.no_save)
        elapsed = time.time() - start_time

        print("\n" + "=" * 68)
        print("                RESEARCH INVESTIGATION COMPLETE")
        print("=" * 68)
        print(f"Elapsed Time: {elapsed:.2f} seconds\n")
        print(result)

    except Exception as e:
        print(f"\n[!] Pipeline error during research execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()