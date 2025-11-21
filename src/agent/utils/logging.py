import logging
import sys
from src.agent.config.settings import settings


def setup_logging():
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
