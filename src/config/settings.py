from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)


class Settings:
    """
    A centralized class to hold all application settings loaded from environment variables.
    """

    def __init__(self):
        # Required settings - validate immediately
        self.GROQ_API_KEY: str = self._get_required("GROQ_API_KEY")
        self.GROQ_MODEL: str = self._get_required("GROQ_MODEL")
        self.TAVILY_API_KEY: str = self._get_required("TAVILY_API_KEY")
    
    @staticmethod
    def _get_required(key: str) -> str:
        """Get a required environment variable or raise ValueError."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} not found in environment variables. Please check your .env file.")
        return value

# Create a single instance of the settings to be imported across the application
settings = Settings()
