#!/usr/bin/env python3
"""Download and bundle the embedding model for PyInstaller builds."""

import os

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('jinaai/jina-code-embeddings-0.5b', trust_remote_code=True)
# Save to bundled_model directory for PyInstaller
model.save('bundled_model')
print('Model saved to bundled_model/')
print(f'Files: {os.listdir("bundled_model")}')
