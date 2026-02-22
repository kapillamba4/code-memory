"""
PyInstaller hook for sqlite-vec.

Ensures the native SQLite extension is properly bundled.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

# Collect native libraries
binaries = collect_dynamic_libs('sqlite_vec')

# Include metadata
datas = copy_metadata('sqlite-vec')
