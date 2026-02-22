# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for code-memory MCP server.

Build commands:
    # Development build
    pyinstaller code-memory.spec

    # Clean build
    pyinstaller --clean code-memory.spec
"""

import sys
from pathlib import Path

block_cipher = None

# Get the project root directory
PROJECT_ROOT = Path(SPECPATH)

# Collect all tree-sitter language bindings
tree_sitter_languages = [
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

# Hidden imports that PyInstaller might miss
hidden_imports = [
    # Core dependencies
    'mcp',
    'mcp.server',
    'mcp.server.fastmcp',
    'sqlite3',
    'sqlite_vec',
    'sentence_transformers',
    'torch',
    'transformers',
    'huggingface_hub',
    'safetensors',
    # Tree-sitter core
    'tree_sitter',
    # Local modules
    'server',
    'db',
    'parser',
    'doc_parser',
    'queries',
    'git_search',
    'errors',
    'validation',
    'logging_config',
    # GitPython dependencies
    'git',
    'gitdb',
    # Pathspec
    'pathspec',
    # Markdown parsing
    'markdown_it',
    'mdit_py_plugins',
    # Hashing
    'xxhash',
]

# Add all tree-sitter languages to hidden imports
hidden_imports.extend(tree_sitter_languages)

# Collect data files
datas = []

# Collect tree-sitter native libraries
for lang in tree_sitter_languages:
    try:
        module = __import__(lang)
        if hasattr(module, '__file__') and module.__file__:
            module_dir = Path(module.__file__).parent
            # Include the entire package directory to get native libs
            datas.append((str(module_dir), lang))
    except ImportError:
        pass

# Include sentence-transformers and transformers data if available
try:
    import sentence_transformers
    if hasattr(sentence_transformers, '__file__'):
        st_dir = Path(sentence_transformers.__file__).parent
        datas.append((str(st_dir), 'sentence_transformers'))
except ImportError:
    pass

a = Analysis(
    ['server.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[str(PROJECT_ROOT / 'hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter',
        'unittest',
        'pydoc',
        'doctest',
        'test',
        'tests',
        'pytest',
        'ruff',
        'mypy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='code-memory',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Enable UPX compression for smaller binaries
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # MCP servers need console for stdio
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
