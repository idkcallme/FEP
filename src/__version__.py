"""
Version management for FEP Cognitive Architecture
================================================
"""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)

# Release information
__title__ = "FEP Cognitive Architecture"
__description__ = "Free Energy Principle-based Cognitive AI System with Mathematical Rigor"
__author__ = "FEP Research Team"
__author_email__ = "research@fep-cognitive.ai"
__license__ = "MIT"
__url__ = "https://github.com/idkcallme/FEP"

# Build information
__build__ = "stable"
__status__ = "Production/Stable"

# Version history markers
MAJOR_VERSION = 1  # Breaking changes
MINOR_VERSION = 0  # New features, backwards compatible
PATCH_VERSION = 0  # Bug fixes, backwards compatible

def get_version():
    """Return the current version string."""
    return __version__

def get_version_info():
    """Return version as a tuple of integers."""
    return __version_info__

def get_full_version():
    """Return detailed version information."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "status": __status__,
        "build": __build__
    }
