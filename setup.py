#!/usr/bin/env python3
"""
Setup script for FEP Cognitive Architecture
===========================================

Professional installation and distribution configuration.
"""

import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Import version information
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from __version__ import __version__, __description__, __author__, __author_email__, __url__

# Read long description from README (if it exists)
this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')
else:
    long_description = __description__

# Core requirements (essential for basic functionality)
install_requires = [
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'scikit-learn>=1.0.0',
    'torch>=1.11.0',
    'tensorflow>=2.8.0',
    'tensorflow-probability>=0.15.0',
    'pandas>=1.3.0',
    'matplotlib>=3.4.0',
    'requests>=2.25.0',
    'jsonschema>=4.0.0',
    'joblib>=1.1.0',
]

# Optional extras for different use cases
extras_require = {
    'web': [
        'flask>=2.0.0',
        'flask-cors>=3.0.0',
        'flask-socketio>=5.0.0',
    ],
    'visualization': [
        'plotly>=5.0.0',
        'seaborn>=0.11.0',
        'ipywidgets>=7.6.0',
    ],
    'dev': [
        'pytest>=6.0.0',
        'black>=21.0.0',
        'flake8>=4.0.0',
        'mypy>=0.910',
        'isort>=5.9.0',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'sphinx-autodoc-typehints>=1.12.0',
    ],
    'security': [
        'cryptography>=3.4.0',
        'validators>=0.18.0',
    ],
    'monitoring': [
        'psutil>=5.8.0',
        'memory-profiler>=0.60.0',
    ],
    'nlp': [
        'transformers>=4.20.0',
        'huggingface-hub>=0.8.0',
        'tokenizers>=0.12.0',
    ]
}

# All extras combined
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    # Basic package information
    name='fep-cognitive-architecture',
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # Author information
    author=__author__,
    author_email=__author_email__,
    url=__url__,
    
    # Package discovery
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Include additional files
    include_package_data=True,
    package_data={
        'fep_cognitive': [
            'data/*.json',
            'data/*.csv',
            'docs/*.md',
        ],
    },
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'fep-demo=fep_cognitive_architecture:main',
            'fep-test=test_fep_mathematics:main',
            'fep-benchmark=real_benchmark_integration:main',
            'fep-security=real_fep_security_system:main',
        ],
    },
    
    # Classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords for discovery
    keywords=[
        'artificial intelligence',
        'cognitive architecture',
        'free energy principle',
        'active inference',
        'predictive coding',
        'machine learning',
        'neuroscience',
        'bayesian brain',
        'variational inference',
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/idkcallme/FEP/docs',
        'Source': 'https://github.com/idkcallme/FEP',
        'Tracker': 'https://github.com/idkcallme/FEP/issues',
        'Research': 'https://github.com/idkcallme/FEP/research',
    },
    
    # License
    license='MIT',
    
    # Zip safety
    zip_safe=False,
)