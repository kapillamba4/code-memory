"""
PyInstaller hooks for all tree-sitter language bindings.

These hooks ensure native libraries for each language parser are bundled.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

TREE_SITTER_LANGUAGES = [
    'tree_sitter_c',
    'tree_sitter_cpp',
    'tree_sitter_go',
    'tree_sitter_java',
    'tree_sitter_javascript',
    'tree_sitter_kotlin',
    'tree_sitter_python',
    'tree_sitter_ruby',
    'tree_sitter_rust',
    'tree_sitter_typescript',
]

# Collect all native libraries and metadata
binaries = []
datas = []
hiddenimports = TREE_SITTER_LANGUAGES

for lang in TREE_SITTER_LANGUAGES:
    try:
        binaries.extend(collect_dynamic_libs(lang))
        datas.extend(copy_metadata(lang.replace('_', '-')))
    except Exception:
        pass
