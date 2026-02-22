"""
PyInstaller hook for sentence-transformers.

Ensures all necessary model loading components are included.
"""

from PyInstaller.utils.hooks import collect_submodules, copy_metadata, collect_data_files

# Collect all submodules
hiddenimports = collect_submodules('sentence_transformers')

# Include metadata and data files
datas = copy_metadata('sentence-transformers')
datas += collect_data_files('sentence_transformers', include_py_files=False)
