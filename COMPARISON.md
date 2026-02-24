# code-memory vs. Cloud-Based Code Intelligence

A comparison of local-first vs. cloud-dependent code intelligence tools.

## The Fundamental Difference

| | code-memory | Cloud Tools (Sourcegraph, Cody, Cursor) |
|---|---|---|
| **Data Location** | Your machine only | Their servers |
| **Network Required** | No (after initial setup) | Yes, always |
| **API Key Needed** | No | Yes |
| **Code Leaves Your Machine** | Never | Yes |
| **Works Offline** | Yes | No |
| **Air-gapped Compatible** | Yes | No |
| **Telemetry** | Zero | Varies |

## Feature Comparison

### Core Capabilities

| Feature | code-memory | Sourcegraph | Cody | Cursor |
|---------|-------------|-------------|------|--------|
| Semantic code search | ✅ | ✅ | ✅ | ✅ |
| Symbol definitions | ✅ | ✅ | ✅ | ✅ |
| Cross-references | ✅ | ✅ | ✅ | ✅ |
| Git history search | ✅ | ✅ | ✅ | ✅ |
| Documentation search | ✅ | ⚠️ Limited | ✅ | ✅ |
| Multi-language AST | ✅ | ✅ | ✅ | ✅ |

### Privacy & Security

| Aspect | code-memory | Cloud Tools |
|--------|-------------|-------------|
| Code sent to external servers | ❌ Never | ✅ Required |
| API keys to manage | ❌ None | ✅ Required |
| Telemetry/tracking | ❌ Zero | ⚠️ Varies |
| SOC 2 compliance needed | ❌ No | ✅ Often required |
| Data residency concerns | ❌ None | ✅ Considerations apply |
| Works in restricted networks | ✅ Yes | ❌ No |

### Deployment

| Aspect | code-memory | Cloud Tools |
|--------|-------------|-------------|
| Installation | `uvx code-memory` | Account + API key |
| Setup time | ~1 minute | Varies |
| Infrastructure | None | Their cloud or self-hosted |
| Air-gapped support | ✅ Yes | ❌ No |
| Self-hosted option | N/A (already local) | ✅ Often available |

## When to Choose code-memory

### Ideal For:

- **Proprietary codebases** — Your code never leaves your machine
- **Security-conscious organizations** — Zero external data transmission
- **Air-gapped environments** — Works in completely isolated networks
- **Offline development** — Full functionality without internet
- **Privacy-focused developers** — Zero telemetry, zero tracking
- **Quick setup** — No accounts, no API keys, no configuration

### Consider Cloud Tools If:

- You need team-wide code search across repositories
- You want cloud-based AI code generation
- Your workflow benefits from cloud sync
- You're comfortable with code being processed externally

## Technical Deep Dive

### How code-memory Stays Local

1. **Embeddings**: Uses `sentence-transformers` running locally on your CPU/GPU
2. **Vector Search**: SQLite with `sqlite-vec` extension — no external database
3. **Code Parsing**: Tree-sitter runs entirely in-process
4. **Git Operations**: Local git repository access only
5. **Model Storage**: Downloaded once to `~/.cache/huggingface/`

### Network Activity

```
code-memory network footprint:
├── Initial setup only (optional):
│   └── Model download (~600MB to local cache)
└── Runtime: ZERO network calls
```

Compare to cloud tools which require persistent network connections for every operation.

### Air-gapped Deployment

code-memory can run in completely isolated environments:

1. **Pre-download the embedding model** on a connected machine
2. **Transfer** the model cache directory (`~/.cache/huggingface/`)
3. **Install** code-memory via offline pip or standalone binary
4. **Run** — no network required

See [AIRGAPPED.md](AIRGAPPED.md) for detailed instructions.

## Cost Comparison

| | code-memory | Cloud Tools |
|---|---|---|
| Monetary cost | Free (MIT license) | Often subscription-based |
| Compute cost | Your hardware | Their infrastructure |
| Hidden costs | None | API usage, overages |
| Privacy cost | Zero | Your code on their servers |

## Zero Telemetry Guarantee

code-memory contains **no telemetry, no analytics, no tracking code**.

This isn't a configuration option — it's architectural. The codebase has:

- No HTTP clients for analytics
- No usage tracking
- No error reporting to external services
- No "phone home" functionality

You can verify this yourself by examining the source code.

## Summary

| Priority | Recommended Tool |
|----------|-----------------|
| Privacy & security | **code-memory** |
| Offline/air-gapped work | **code-memory** |
| Zero setup friction | **code-memory** |
| Team collaboration | Cloud tools (Sourcegraph) |
| Cloud AI features | Cloud tools (Cody, Cursor) |

**code-memory is the only option that guarantees your code stays on your machine.**
