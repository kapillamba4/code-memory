#!/usr/bin/env python3
"""Download and bundle the embedding model for PyInstaller builds."""

from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
# Save to bundled_model directory for PyInstaller
model.save('bundled_model')
print(f'Model saved to bundled_model/')
print(f'Files: {os.listdir("bundled_model")}')
