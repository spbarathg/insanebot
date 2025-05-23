#!/usr/bin/env python3
"""
Script to fix encoding and line ending issues for server deployment.
"""
import os
import sys

def fix_file_encoding(filepath):
    """Fix file encoding and line endings."""
    try:
        # Read the file with different encodings
        content = None
        encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"Successfully read {filepath} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"Error: Could not read {filepath} with any encoding")
            return False
        
        # Normalize line endings to Unix format
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Write back with UTF-8 encoding and Unix line endings
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        
        print(f"Fixed encoding and line endings for {filepath}")
        return True
        
    except Exception as e:
        print(f"Error fixing {filepath}: {str(e)}")
        return False

def main():
    """Main function to fix all Python files."""
    files_to_fix = [
        'src/main.py',
        'main.py'
    ]
    
    # Find all Python files in src directory
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                files_to_fix.append(os.path.join(root, file))
    
    success_count = 0
    total_count = 0
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            total_count += 1
            if fix_file_encoding(filepath):
                success_count += 1
    
    print(f"\nFixed {success_count}/{total_count} files")
    
    # Test syntax of main file
    try:
        import py_compile
        py_compile.compile('src/main.py', doraise=True)
        print("✓ src/main.py syntax check passed")
    except py_compile.PyCompileError as e:
        print(f"✗ src/main.py syntax error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Error checking syntax: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 