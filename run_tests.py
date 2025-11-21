#!/usr/bin/env python3
"""
Test runner script for the RAG agent.
"""

import sys
import logging
import pytest

# Setup basic logging for test runner
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run all tests with pytest."""
    logger.info("=" * 60)
    logger.info("Running SOP Agent Test Suite")
    logger.info("=" * 60)

    # Run pytest with verbose output
    args = [
        "src/tests/",
        "-v",
        "--tb=short",
        "--color=yes",
        "-ra",  # Show summary of all test outcomes
    ]

    exit_code = pytest.main(args)

    logger.info("=" * 60)
    if exit_code == 0:
        logger.info("All tests passed!")
    else:
        logger.error(f"Tests failed with exit code: {exit_code}")
    logger.info("=" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
