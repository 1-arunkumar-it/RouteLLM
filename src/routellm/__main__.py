"""Allow running the package directly with ``python -m routellm``."""

from .cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
