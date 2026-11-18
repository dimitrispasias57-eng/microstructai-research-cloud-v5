"""Deployment mode is chosen by the launcher, never by an online visitor."""

import os


def is_web_deployment(environ=None):
    """Fail closed: only an explicit desktop launch enables filesystem inputs."""
    source = os.environ if environ is None else environ
    return source.get("MICROSTRUCTAI_MODE", "web").strip().lower() != "desktop"
