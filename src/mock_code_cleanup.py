#!/usr/bin/env python3
"""
Mock Code Cleanup and Deprecation Management
===========================================

This script identifies, documents, and manages deprecated and mock code
as required by peer review recommendations. It provides clear marking
of placeholder implementations and removes or archives outdated code.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
import logging

logger = logging.getLogger(__name__)


class MockCodeManager:
    """
    Manages identification and cleanup of mock implementations and deprecated code.
    
    Provides systematic approach to code quality management as required
    by peer review feedback.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.mock_patterns = [
            r'mock_implementation',
            r'random\.normal\(',
            r'np\.random\.',
            r'# Mock',
            r'# TODO.*mock',
            r'placeholder',
            r'DEPRECATED',
            r'fallback.*random',
            r'if.*available.*else.*random'
        ]
        
        self.deprecated_files = []
        self.mock_implementations = []
        self.cleanup_actions = []
    
    def scan_project(self) -> Dict[str, List[str]]:
        """
        Scan entire project for mock implementations and deprecated code.
        
        Returns:
            Dictionary categorizing found issues
        """
        results = {
            'deprecated_files': [],
            'mock_implementations': [],
            'random_fallbacks': [],
            'placeholder_code': [],
            'todo_items': []
        }
        
        # Scan Python files
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            file_issues = self._analyze_file(py_file)
            
            # Categorize issues
            for category, issues in file_issues.items():
                if issues:
                    results[category].extend([(str(py_file), issue) for issue in issues])
        
        # Identify deprecated files by name patterns
        deprecated_patterns = [
            '*deprecated*',
            '*mock*',
            '*old*',
            '*backup*',
            '*temp*'
        ]
        
        for pattern in deprecated_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file() and file_path.suffix == '.py':
                    results['deprecated_files'].append(str(file_path))
        
        return results
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'venv',
            'env',
            '.pytest_cache',
            'node_modules'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _analyze_file(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze individual file for mock implementations and issues."""
        issues = {
            'deprecated_files': [],
            'mock_implementations': [],
            'random_fallbacks': [],
            'placeholder_code': [],
            'todo_items': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for deprecated file markers
            if re.search(r'DEPRECATED|deprecated|OBSOLETE', content, re.IGNORECASE):
                issues['deprecated_files'].append("File marked as deprecated")
            
            # Analyze line by line
            for line_num, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Check for mock implementations
                if re.search(r'mock_implementation|Mock.*Implementation', line, re.IGNORECASE):
                    issues['mock_implementations'].append(f"Line {line_num}: {line_stripped}")
                
                # Check for random fallbacks
                if re.search(r'if.*available.*else.*random|fallback.*random', line, re.IGNORECASE):
                    issues['random_fallbacks'].append(f"Line {line_num}: {line_stripped}")
                
                # Check for placeholder code
                if re.search(r'placeholder|TODO|FIXME|XXX', line, re.IGNORECASE):
                    issues['placeholder_code'].append(f"Line {line_num}: {line_stripped}")
                
                # Check for random number generation in computational contexts
                if re.search(r'np\.random\.|random\.normal|random\.randn', line):
                    # Check if it's in a computational context (not test or demo)
                    if not re.search(r'test|demo|example', str(file_path), re.IGNORECASE):
                        issues['random_fallbacks'].append(f"Line {line_num}: Random computation - {line_stripped}")
        
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return issues
    
    def generate_cleanup_report(self) -> str:
        """Generate comprehensive cleanup report."""
        scan_results = self.scan_project()
        
        report = []
        report.append("MOCK CODE CLEANUP REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        total_issues = sum(len(issues) for issues in scan_results.values())
        report.append(f"Total issues found: {total_issues}")
        report.append("")
        
        # Detailed breakdown
        for category, issues in scan_results.items():
            if issues:
                report.append(f"{category.upper().replace('_', ' ')} ({len(issues)} issues):")
                report.append("-" * 40)
                
                # Group by file
                file_issues = {}
                for item in issues:
                    if isinstance(item, tuple) and len(item) == 2:
                        file_path, issue = item
                        if file_path not in file_issues:
                            file_issues[file_path] = []
                        file_issues[file_path].append(issue)
                    else:
                        # Handle single items (like deprecated files)
                        file_path = str(item)
                        if file_path not in file_issues:
                            file_issues[file_path] = []
                        file_issues[file_path].append("File identified as deprecated")
                
                for file_path, file_issue_list in file_issues.items():
                    report.append(f"  {file_path}:")
                    for issue in file_issue_list:
                        report.append(f"    - {issue}")
                    report.append("")
                
                report.append("")
        
        # Cleanup recommendations
        report.append("CLEANUP RECOMMENDATIONS:")
        report.append("-" * 30)
        
        if scan_results['deprecated_files']:
            report.append("1. DEPRECATED FILES:")
            report.append("   - Move to archive/ directory or delete")
            report.append("   - Update imports and references")
            report.append("")
        
        if scan_results['mock_implementations']:
            report.append("2. MOCK IMPLEMENTATIONS:")
            report.append("   - Replace with real implementations")
            report.append("   - Add clear placeholder documentation")
            report.append("   - Mark as experimental if needed")
            report.append("")
        
        if scan_results['random_fallbacks']:
            report.append("3. RANDOM FALLBACKS:")
            report.append("   - Remove random fallback mechanisms")
            report.append("   - Implement proper error handling")
            report.append("   - Add clear limitations documentation")
            report.append("")
        
        if scan_results['placeholder_code']:
            report.append("4. PLACEHOLDER CODE:")
            report.append("   - Complete TODO items")
            report.append("   - Remove FIXME markers")
            report.append("   - Document known limitations")
            report.append("")
        
        return "\n".join(report)
    
    def create_archive_directory(self) -> Path:
        """Create archive directory for deprecated code."""
        archive_dir = self.project_root / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (archive_dir / "deprecated").mkdir(exist_ok=True)
        (archive_dir / "mock_implementations").mkdir(exist_ok=True)
        (archive_dir / "backup").mkdir(exist_ok=True)
        
        return archive_dir
    
    def mark_mock_implementations(self) -> List[str]:
        """Add clear markers to mock implementations."""
        scan_results = self.scan_project()
        marked_files = []
        
        for file_path, issue in scan_results['mock_implementations']:
            try:
                # Add warning header to mock implementation files
                self._add_mock_warning_header(Path(file_path))
                marked_files.append(file_path)
            except Exception as e:
                logger.warning(f"Failed to mark {file_path}: {e}")
        
        return marked_files
    
    def _add_mock_warning_header(self, file_path: Path):
        """Add warning header to mock implementation file."""
        warning_header = '''#!/usr/bin/env python3
"""
⚠️  MOCK IMPLEMENTATION WARNING ⚠️
================================

This file contains mock implementations and placeholder code that do not
represent real functionality. Results from this code should not be used
for scientific claims or production deployment.

Status: PLACEHOLDER - Requires real implementation
Last Updated: {timestamp}
Peer Review Status: IDENTIFIED FOR REPLACEMENT

Mock components in this file:
- Random number generation instead of real computation
- Heuristic implementations without theoretical foundation
- Placeholder algorithms requiring proper implementation

DO NOT USE FOR:
- Scientific validation
- Performance benchmarking  
- Production deployment
- Academic publication claims

"""

import warnings
warnings.warn(
    "This module contains mock implementations. Results are not scientifically valid.",
    UserWarning,
    stacklevel=2
)

'''
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check if warning already exists
            if "MOCK IMPLEMENTATION WARNING" in original_content:
                return
            
            # Find the first import or class/function definition
            lines = original_content.split('\n')
            insert_index = 0
            
            # Skip shebang and existing docstring
            for i, line in enumerate(lines):
                if line.startswith('#!/') or line.startswith('"""') or line.startswith("'''"):
                    continue
                if line.strip().startswith('import ') or line.strip().startswith('from ') or \
                   line.strip().startswith('class ') or line.strip().startswith('def '):
                    insert_index = i
                    break
            
            # Insert warning
            import datetime
            warning_with_timestamp = warning_header.format(
                timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            new_lines = lines[:insert_index] + [warning_with_timestamp] + lines[insert_index:]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
                
            logger.info(f"Added mock warning to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to add warning to {file_path}: {e}")
    
    def create_replacement_stubs(self) -> Dict[str, str]:
        """Create stub files for deprecated implementations."""
        stubs = {}
        scan_results = self.scan_project()
        
        for file_path, issue in scan_results['deprecated_files']:
            stub_content = self._generate_replacement_stub(Path(file_path))
            stubs[file_path] = stub_content
        
        return stubs
    
    def _generate_replacement_stub(self, file_path: Path) -> str:
        """Generate replacement stub for deprecated file."""
        stub_template = '''#!/usr/bin/env python3
"""
DEPRECATED MODULE STUB
=====================

This module has been deprecated and moved to archive.

Original file: {original_file}
Deprecation date: {date}
Reason: {reason}

Replacement: {replacement}
Migration guide: {migration_guide}
"""

import warnings

def __getattr__(name):
    warnings.warn(
        f"Module {{name}} from {file_path.name} is deprecated. "
        f"Please use the replacement implementation.",
        DeprecationWarning,
        stacklevel=2
    )
    raise ImportError(f"{{name}} is no longer available in deprecated module {file_path.name}")

# Legacy imports for backward compatibility (with warnings)
__all__ = []
'''
        
        import datetime
        
        return stub_template.format(
            original_file=file_path.name,
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
            reason="Mock implementation replaced with real functionality",
            replacement="See src/ directory for real implementations",
            migration_guide="Update imports to use src.* modules instead"
        )


def main():
    """Run mock code cleanup analysis and management."""
    print("Mock Code Cleanup and Management")
    print("=" * 40)
    
    # Initialize manager
    manager = MockCodeManager()
    
    # Generate cleanup report
    print("Scanning project for mock implementations and deprecated code...")
    report = manager.generate_cleanup_report()
    
    # Save report
    report_file = "MOCK_CODE_CLEANUP_REPORT.md"
    with open(report_file, 'w') as f:
        f.write("# " + report.replace('\n', '\n'))
    
    print(f"Cleanup report saved to: {report_file}")
    print()
    print("Report Summary:")
    print("-" * 20)
    
    # Print summary
    scan_results = manager.scan_project()
    for category, issues in scan_results.items():
        if issues:
            print(f"{category.replace('_', ' ').title()}: {len(issues)} issues")
    
    # Perform cleanup actions
    print("\nPerforming cleanup actions...")
    
    # Create archive directory
    archive_dir = manager.create_archive_directory()
    print(f"Created archive directory: {archive_dir}")
    
    # Mark mock implementations
    marked_files = manager.mark_mock_implementations()
    if marked_files:
        print(f"Added mock warnings to {len(marked_files)} files")
    
    # Create replacement stubs
    stubs = manager.create_replacement_stubs()
    if stubs:
        print(f"Generated {len(stubs)} replacement stubs")
    
    print("\nCleanup actions completed!")
    print(f"Review the full report in: {report_file}")


if __name__ == "__main__":
    main()
