"""
Script to update pyproject.toml dependencies based on uv.lock file.

Prerequisites:
    1. Python 3.11+ (for tomllib)
    2. tomli-w package (`pip install tomli-w`)
    3. Updated uv.lock file (`uv lock --update` or `uv lock -U`)

Usage:
    1. Update your lockfile: `uv lock -U`
    2. Run this script: `python upgrade_pyproject.py`

Notes:
    - Creates a backup of pyproject.toml before making changes
    - Preserves dependency extras (e.g., fastapi[standard])
    - Updates both main dependencies and dependency groups
    - Removes duplicate version constraints
"""

import shutil
import tomllib
from pathlib import Path

import tomli_w


def clean_dependency(dep: str) -> tuple[str, str]:
    """Clean dependency string and return package name and extras."""
    base = dep.split(">=")[0].strip()

    # Extract extras if present
    if "[" in base:
        pkg_name = base.split("[")[0].strip()
        extras = "[" + base.split("[")[1]
        return pkg_name.lower(), extras
    return base.lower(), ""


def update_dependencies(pyproject_path: Path, lock_path: Path) -> None:
    """Update pyproject.toml dependencies based on uv.lock."""
    try:
        # Create backup
        backup_path = pyproject_path.with_suffix(".toml.backup")
        shutil.copy2(pyproject_path, backup_path)
        print(f"Created backup at: {backup_path}")

        # Read files
        print("\nReading project files...")
        pyproject = tomllib.loads(pyproject_path.read_text())
        lock_content = lock_path.read_text().split("\n")

        # Parse lock file
        print("\nParsing lock file...")
        versions = {}
        for i, line in enumerate(lock_content):
            if 'name = "' in line:
                name = line.split('"')[1].lower()
                if i + 1 < len(lock_content) and 'version = "' in lock_content[i + 1]:
                    versions[name] = lock_content[i + 1].split('"')[1]
        print(f"Found {len(versions)} packages in lock file")

        # Update main dependencies
        print("\nUpdating dependencies...")
        if deps := pyproject.get("project", {}).get("dependencies"):
            print("\nMain dependencies:")
            for i, dep in enumerate(deps):
                pkg_name, extras = clean_dependency(dep)
                if pkg_name in versions:
                    old_dep = deps[i]
                    deps[i] = f"{pkg_name}{extras}>={versions[pkg_name]}"
                    print(f"  {old_dep} -> {deps[i]}")

        # Update dependency groups
        if groups := pyproject.get("dependency-groups"):
            for group_name, deps in groups.items():
                print(f"\n{group_name} dependencies:")
                for i, dep in enumerate(deps):
                    pkg_name, extras = clean_dependency(dep)
                    if pkg_name in versions:
                        old_dep = deps[i]
                        deps[i] = f"{pkg_name}{extras}>={versions[pkg_name]}"
                        print(f"  {old_dep} -> {deps[i]}")

        # Write updated pyproject.toml
        print("\nWriting updated pyproject.toml...")
        pyproject_path.write_bytes(tomli_w.dumps(pyproject).encode())
        print("Successfully updated pyproject.toml")

    except Exception as e:
        print(f"\nError: {e}")
        exit(1)


if __name__ == "__main__":
    print("Starting dependency update process...\n")
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")

    pyproject_path = current_dir / "pyproject.toml"
    lock_path = current_dir / "uv.lock"

    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        exit(1)
    if not lock_path.exists():
        print("Error: uv.lock not found")
        exit(1)

    update_dependencies(pyproject_path, lock_path)
