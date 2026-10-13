"""
Entry point.

    python main.py compare                                  # synthetic random walk (offline)
    python main.py compare --ticker SPY --start 2005-01-01 --end 2025-01-01 --ml
    python main.py optimize --strategy ma --ticker SPY --start 2005-01-01 --end 2025-01-01

See README.md, or `python main.py compare -h`.
"""

from backtester.cli import main

if __name__ == "__main__":
    main()
