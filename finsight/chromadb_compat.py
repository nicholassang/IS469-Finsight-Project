"""
chromadb_compat.py
Compatibility layer for ChromaDB on clusters with old SQLite.

This module patches the sqlite3 module to use pysqlite3-binary
before ChromaDB imports, allowing it to work on systems with SQLite < 3.35.

Usage:
    # At the top of your script, before any ChromaDB imports:
    import chromadb_compat

    # Then import ChromaDB normally:
    import chromadb
"""

import sys

# Check if we need the compatibility layer
try:
    import sqlite3
    sqlite_version = sqlite3.sqlite_version_info

    if sqlite_version < (3, 35, 0):
        print(f"[chromadb_compat] Detected old SQLite {sqlite3.sqlite_version}, patching with pysqlite3...")

        # Replace sqlite3 with pysqlite3
        try:
            import pysqlite3 as sqlite3_compat

            # Monkey-patch the sqlite3 module in sys.modules
            sys.modules['sqlite3'] = sys.modules['pysqlite3']

            # Verify the patch worked
            import sqlite3 as sqlite3_new
            new_version = sqlite3_new.sqlite_version
            print(f"[chromadb_compat] ✓ Successfully patched to SQLite {new_version}")

        except ImportError:
            print("[chromadb_compat] ✗ pysqlite3-binary not installed!")
            print("[chromadb_compat] Install with: python3 -m pip install --user pysqlite3-binary")
            raise
    else:
        print(f"[chromadb_compat] System SQLite {sqlite3.sqlite_version} is sufficient, no patch needed")

except Exception as e:
    print(f"[chromadb_compat] Warning: Could not check SQLite version: {e}")
