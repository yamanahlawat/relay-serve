"""
Script to automatically update uv.lock and then update pyproject.toml dependencies.

Prerequisites:
    1. Python 3.11+ (for tomllib)
    2. tomli-w package (`pip install tomli-w`)
    3. uv installed and available in PATH

Usage:
    Run this script: `python upgrade_pyproject.py`

    The script will:
    1. Automatically run `uv lock -U` to update the lockfile
    2. Update pyproject.toml based on the new lock file

Notes:
    - Preserves dependency extras (e.g., fastapi[standard])
    - Updates both main dependencies and dependency groups
    - Removes duplicate version constraints
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import tomli_w

# Regex to split on any PEP 440 version specifier operator
_VERSION_SPEC_RE = re.compile(r"(~=|==|!=|<=|>=|<|>)")


def run_uv_lock_update() -> bool:
    """Run uv lock -U command and return success status."""
    print("Running 'uv lock -U' to update dependencies...")
    print("-" * 50)

    try:
        result = subprocess.run(["uv", "lock", "-U"], capture_output=True, text=True, check=True)

        if result.stdout:
            print(result.stdout)

        print("-" * 50)
        print("✓ Successfully updated uv.lock")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Error running 'uv lock -U': {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Error: 'uv' command not found. Please ensure uv is installed and in PATH.")
        return False


def _normalize_name(name: str) -> str:
    """Normalize a package name per PEP 503 (lowercase, [-_.] -> hyphens)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _version_tuple(version: str) -> tuple[int, ...]:
    """Convert a version string like '1.2.3' into a tuple of ints for comparison."""
    parts: list[int] = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def _parse_dependency(dep: str) -> tuple[str, str]:
    """Parse a dependency string into (package name with extras, raw name for lookup).

    Returns:
        A tuple of (base string before version specifier, normalized package name).
        The base string preserves the original casing and extras for display.

    Examples:
        "rich>=14.0"        -> ("rich", "rich")
        "fastapi[standard]>=0.100" -> ("fastapi[standard]", "fastapi")
        "Ruff==0.5.0"       -> ("Ruff", "ruff")
        "httpx"             -> ("httpx", "httpx")
    """
    # Split on the first version specifier operator
    parts = _VERSION_SPEC_RE.split(dep, maxsplit=1)
    base = parts[0].strip()

    # Extract the bare package name (without extras) for version lookup
    pkg_name = base.split("[")[0].strip() if "[" in base else base
    return base, _normalize_name(pkg_name)


def _parse_lock_versions(lock_path: Path) -> dict[str, str]:
    """Parse uv.lock (TOML) and return a dict of normalized package name -> version.

    When a package appears multiple times (e.g. different versions per Python
    resolution marker), keeps the lowest version so that the '>=' constraint
    is satisfiable across all supported Python versions.
    """
    lock_data = tomllib.loads(lock_path.read_text())
    versions: dict[str, str] = {}

    for package in lock_data.get("package", []):
        name = _normalize_name(package["name"])
        version = package.get("version", "")
        if not version:
            continue

        if name not in versions or _version_tuple(version) < _version_tuple(versions[name]):
            versions[name] = version

    return versions


def _update_dep_list(deps: list[str], versions: dict[str, str]) -> list[tuple[str, str]]:
    """Update a list of dependency strings in place. Returns list of (old, new) changes."""
    changes: list[tuple[str, str]] = []
    for i, dep in enumerate(deps):
        base, pkg_name = _parse_dependency(dep)
        if pkg_name in versions:
            new_dep = f"{base}>={versions[pkg_name]}"
            if dep != new_dep:
                changes.append((dep, new_dep))
                deps[i] = new_dep
    return changes


def update_dependencies(pyproject_path: Path, lock_path: Path) -> None:
    """Update pyproject.toml dependencies based on uv.lock."""
    try:
        print("\nReading project files...")
        pyproject = tomllib.loads(pyproject_path.read_text())

        print("\nParsing lock file...")
        versions = _parse_lock_versions(lock_path)
        print(f"Found {len(versions)} packages in lock file")

        print("\nUpdating dependencies...")
        updated_count = 0

        # Update main dependencies
        if deps := pyproject.get("project", {}).get("dependencies"):
            changes = _update_dep_list(deps, versions)
            if changes:
                print("\nMain dependencies:")
                for old, new in changes:
                    print(f"  {old} -> {new}")
                updated_count += len(changes)

        # Update dependency groups
        if groups := pyproject.get("dependency-groups"):
            for group_name, deps in groups.items():
                changes = _update_dep_list(deps, versions)
                if changes:
                    print(f"\n{group_name} dependencies:")
                    for old, new in changes:
                        print(f"  {old} -> {new}")
                    updated_count += len(changes)

        # Write updated pyproject.toml
        if updated_count > 0:
            print(f"\nWriting updated pyproject.toml ({updated_count} dependencies updated)...")
            with pyproject_path.open("wb") as f:
                tomli_w.dump(pyproject, f)
            print("✓ Successfully updated pyproject.toml")
        else:
            print("\n✓ No updates needed - all dependencies are already up to date")

    except Exception as e:
        print(f"\n✗ Error updating dependencies: {e}")
        sys.exit(1)


def main():
    """Main function to coordinate the update process."""
    print("🚀 Starting automated dependency update process...\n")
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")

    pyproject_path = current_dir / "pyproject.toml"
    lock_path = current_dir / "uv.lock"

    # Check if required files exist
    if not pyproject_path.exists():
        print("✗ Error: pyproject.toml not found")
        sys.exit(1)
    if not lock_path.exists():
        print("✗ Error: uv.lock not found")
        print("  Run 'uv lock' first to create the lock file")
        sys.exit(1)

    # Step 1: Run uv lock -U
    print("\n" + "=" * 60)
    print("STEP 1: Updating lock file")
    print("=" * 60 + "\n")

    if not run_uv_lock_update():
        print("\n✗ Failed to update lock file. Aborting.")
        sys.exit(1)

    # Step 2: Update pyproject.toml
    print("\n" + "=" * 60)
    print("STEP 2: Updating pyproject.toml")
    print("=" * 60 + "\n")

    update_dependencies(pyproject_path, lock_path)

    print("\n" + "=" * 60)
    print("✓ All done! Your dependencies have been updated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
