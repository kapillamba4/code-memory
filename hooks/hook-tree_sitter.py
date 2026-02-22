"""
PyInstaller hook for tree-sitter language bindings.

This hook ensures all native libraries for tree-sitter language parsers
are properly collected and bundled in the executable.
"""

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# Collect all tree-sitter submodules
hiddenimports = collect_submodules('tree_sitter')

# Collect metadata for proper package resolution
datas = copy_metadata('tree-sitter')
