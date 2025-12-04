#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch update file headers with Synria Robotics copyright notice.

This script updates all Python files in the RobotCore project with
a standardized header containing copyright and license information.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


# Standard header template for Python files
PYTHON_HEADER = '''"""{}

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""'''


def extract_title(content: str) -> str:
    """Extract title from existing docstring or filename."""
    # Try to find existing docstring
    docstring_match = re.match(r'^"""(.+?)"""', content, re.DOTALL)
    if docstring_match:
        first_line = docstring_match.group(1).strip().split('\n')[0]
        # Remove common prefixes
        first_line = re.sub(r'^(RobotCore\s*-?\s*)?', '', first_line, flags=re.IGNORECASE)
        return first_line.strip()
    return "RobotCore Module"


def update_python_file(filepath: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Update a single Python file with standard header.
    
    Args:
        filepath: Path to the Python file
        dry_run: If True, only print changes without modifying files
        
    Returns:
        Tuple of (success, message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from existing content
        title = extract_title(content)
        
        # Generate new header
        new_header = PYTHON_HEADER.format(title)
        
        # Remove old docstring (first triple-quoted string)
        # Match shebang and encoding if present
        shebang_match = re.match(r'^(#!/.+?\n)?', content)
        encoding_match = re.match(r'^(#!/.+?\n)?(#.*?coding[:=].+?\n)?', content)
        
        prefix = encoding_match.group(0) if encoding_match else ''
        remainder = content[len(prefix):]
        
        # Remove old docstring
        remainder = re.sub(r'^""".*?"""\s*', '', remainder, count=1, flags=re.DOTALL)
        remainder = remainder.lstrip('\n')
        
        # Construct new content
        new_content = prefix + new_header + '\n\n' + remainder
        
        if dry_run:
            return True, f"Would update: {filepath}"
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True, f"Updated: {filepath}"
            
    except Exception as e:
        return False, f"Error updating {filepath}: {e}"


def find_python_files(root_dir: Path, exclude_dirs: List[str] = None) -> List[Path]:
    """Find all Python files in the directory tree.
    
    Args:
        root_dir: Root directory to search
        exclude_dirs: List of directory names to exclude
        
    Returns:
        List of Python file paths
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', 'venv', 'env', '.eggs', 'build', 'dist']
    
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Remove excluded directories from search
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update file headers with Synria Robotics copyright notice'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--path',
        type=str,
        default='.',
        help='Root path to search for Python files (default: current directory)'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        nargs='+',
        default=['__pycache__', '.git', 'venv', 'env', '.eggs', 'build', 'dist', 'scripts'],
        help='Directories to exclude from search'
    )
    
    args = parser.parse_args()
    
    root_path = Path(args.path).resolve()
    print(f"Searching for Python files in: {root_path}")
    print(f"Excluded directories: {', '.join(args.exclude)}\n")
    
    # Find all Python files
    python_files = find_python_files(root_path, args.exclude)
    print(f"Found {len(python_files)} Python files\n")
    
    if args.dry_run:
        print("=== DRY RUN MODE - No files will be modified ===\n")
    
    # Update each file
    success_count = 0
    error_count = 0
    
    for filepath in python_files:
        success, message = update_python_file(filepath, dry_run=args.dry_run)
        if success:
            success_count += 1
            print(f"✓ {message}")
        else:
            error_count += 1
            print(f"✗ {message}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files: {len(python_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    
    if args.dry_run:
        print(f"\nThis was a dry run. Use without --dry-run to apply changes.")
    else:
        print(f"\n✓ All files updated successfully!")


if __name__ == '__main__':
    main()
