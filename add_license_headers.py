"""Add Apache 2.0 license headers to all Python files."""

from pathlib import Path

LICENSE_HEADER = '''# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''

def has_license_header(content: str) -> bool:
    """Check if file already has Apache license header."""
    return 'Apache License' in content or 'Licensed under the Apache License' in content

def has_shebang(content: str) -> bool:
    """Check if file starts with shebang."""
    return content.startswith('#!')

def add_header(file_path: Path) -> bool:
    """Add license header to a Python file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Skip if already has license
        if has_license_header(content):
            print(f"SKIP (has license): {file_path}")
            return False
        
        # Handle different file structures
        if has_shebang(content):
            # Keep shebang at top
            lines = content.split('\n', 1)
            if len(lines) == 2:
                new_content = lines[0] + '\n' + LICENSE_HEADER + lines[1]
            else:
                new_content = lines[0] + '\n' + LICENSE_HEADER
        elif content.startswith('"""') or content.startswith("'''"):
            # Keep module docstring after license
            new_content = LICENSE_HEADER + content
        elif content.strip() == '':
            # Empty file
            new_content = LICENSE_HEADER
        else:
            # Regular file
            new_content = LICENSE_HEADER + content
        
        file_path.write_text(new_content, encoding='utf-8')
        print(f"ADDED: {file_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: {file_path}: {e}")
        return False

def main():
    """Add license headers to all Python files in src/."""
    src_dir = Path(__file__).parent / 'src'
    
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist")
        return
    
    python_files = list(src_dir.rglob('*.py'))
    print(f"Found {len(python_files)} Python files\n")
    
    added = 0
    skipped = 0
    errors = 0
    
    for py_file in sorted(python_files):
        result = add_header(py_file)
        if result:
            added += 1
        elif 'ERROR' in str(result):
            errors += 1
        else:
            skipped += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Added:   {added}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print(f"  Total:   {len(python_files)}")

if __name__ == '__main__':
    main()
