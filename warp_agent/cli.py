#!/usr/bin/env python3
"""
Warp Agent CLI - Main entry point for the Warp-Cortex-enabled Hermes Agent.
"""

import sys
import os
import click
from pathlib import Path

# Add the src directory to the Python path so we can import modules
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def main():
    """Main CLI entry point for Warp Agent."""
    try:
        # Import and run the main Hermes CLI
        from hermes_agent.cli import main as hermes_main
        hermes_main()
    except ImportError as e:
        click.echo(f"Error: Could not import Hermes Agent CLI: {e}", err=True)
        click.echo("Please ensure all dependencies are installed.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()