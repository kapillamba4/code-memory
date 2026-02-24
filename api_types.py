"""
TypedDict definitions for all MCP tool API response shapes.

These types provide:
- IDE autocompletion support
- Static type checking via mypy
- Documentation of API contracts
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Shared / Nested Types
# ---------------------------------------------------------------------------


class ParentSymbol(TypedDict):
    """Information about a symbol's containing parent."""

    name: str
    kind: str


class ContainingSymbol(TypedDict):
    """Information about the symbol containing a reference."""

    name: str
    kind: str


class CodeSnippet(TypedDict):
    """A code snippet from a matching symbol."""

    name: str
    kind: str
    line_range: str
    code: str


class FileChangeStats(TypedDict):
    """Statistics for a file changed in a commit."""

    path: str
    insertions: int
    deletions: int


class ContextChunk(TypedDict):
    """An adjacent chunk for context in documentation search."""

    type: Literal["previous", "current", "next"]
    content: str


# ---------------------------------------------------------------------------
# Error Response
# ---------------------------------------------------------------------------


class ErrorResponse(TypedDict):
    """Standard error response returned by all tools on failure."""

    error: Literal[True]
    error_type: str
    message: str
    details: str | dict | None


# ---------------------------------------------------------------------------
# check_index_status Tool
# ---------------------------------------------------------------------------


class CheckIndexStatusResponse(TypedDict):
    """Response from the check_index_status tool."""

    indexed: bool
    symbols_indexed: int
    doc_chunks_indexed: int
    code_files_indexed: int
    doc_files_indexed: int
    suggestion: str
    error: NotRequired[str]


# ---------------------------------------------------------------------------
# get_index_stats Tool
# ---------------------------------------------------------------------------


class IndexCounts(TypedDict):
    """Count metrics for indexed items."""

    symbols: int
    files: int
    doc_chunks: int
    doc_files: int
    references: int
    symbol_embeddings: int
    doc_embeddings: int


class IndexDistributions(TypedDict):
    """Distribution of symbol kinds and file extensions."""

    symbol_kinds: dict[str, int]
    file_extensions: dict[str, int]


class IndexFreshness(TypedDict):
    """Timestamps for last indexing operations."""

    last_code_indexed: str | None
    last_doc_indexed: str | None


class EmbeddingInfo(TypedDict):
    """Information about the embedding model."""

    model: str
    dimension: int


class DatabaseInfo(TypedDict):
    """Database file information."""

    size_mb: float
    journal_mode: str
    wal_exists: bool
    wal_size_mb: float


class GetIndexStatsResponse(TypedDict):
    """Response from the get_index_stats tool."""

    status: Literal["ok"]
    indexed: bool
    counts: IndexCounts
    distributions: IndexDistributions
    freshness: IndexFreshness
    embedding: EmbeddingInfo
    database: DatabaseInfo


# ---------------------------------------------------------------------------
# search_code Tool - Topic Discovery Results
# ---------------------------------------------------------------------------


class TopicDiscoveryResult(TypedDict):
    """A single result from topic_discovery search."""

    file_path: str
    relevance_score: float
    matched_symbols: list[str]
    symbol_kinds: str
    summary: str
    top_snippets: list[CodeSnippet]


class SearchCodeTopicDiscoveryResponse(TypedDict):
    """Response from search_code with search_type='topic_discovery'."""

    status: Literal["ok"]
    search_type: Literal["topic_discovery"]
    query: str
    results: list[TopicDiscoveryResult]
    hint: NotRequired[str]


# ---------------------------------------------------------------------------
# search_code Tool - Definition Results
# ---------------------------------------------------------------------------


class DefinitionResult(TypedDict):
    """A single result from definition search."""

    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    source_text: str
    score: float
    match_reason: str
    confidence: float
    match_highlights: list[str]
    docstring: str | None
    parent: ParentSymbol | None
    signature: str | None


class SearchCodeDefinitionResponse(TypedDict):
    """Response from search_code with search_type='definition'."""

    status: Literal["ok"]
    search_type: Literal["definition"]
    query: str
    results: list[DefinitionResult]
    hint: NotRequired[str]


# ---------------------------------------------------------------------------
# search_code Tool - References Results
# ---------------------------------------------------------------------------


class ReferenceResult(TypedDict):
    """A single result from references search."""

    symbol_name: str
    file_path: str
    line_number: int
    source_line: str | None
    containing_symbol: ContainingSymbol | None


class SearchCodeReferencesResponse(TypedDict):
    """Response from search_code with search_type='references'."""

    status: Literal["ok"]
    search_type: Literal["references"]
    query: str
    results: list[ReferenceResult]
    hint: NotRequired[str]


# ---------------------------------------------------------------------------
# search_code Tool - File Structure Results
# ---------------------------------------------------------------------------


class FileStructureResult(TypedDict):
    """A single symbol entry in file structure."""

    name: str
    kind: str
    line_start: int
    line_end: int
    parent: str | None


class SearchCodeFileStructureResponse(TypedDict):
    """Response from search_code with search_type='file_structure'."""

    status: Literal["ok"]
    search_type: Literal["file_structure"]
    query: str
    results: list[FileStructureResult]
    hint: NotRequired[str]


# ---------------------------------------------------------------------------
# search_code Tool - Union Response Type
# ---------------------------------------------------------------------------


SearchCodeResponse = (
    SearchCodeTopicDiscoveryResponse
    | SearchCodeDefinitionResponse
    | SearchCodeReferencesResponse
    | SearchCodeFileStructureResponse
    | ErrorResponse
)


# ---------------------------------------------------------------------------
# index_codebase Tool
# ---------------------------------------------------------------------------


class IndexingPerformance(TypedDict):
    """Performance metrics for indexing."""

    total_time_seconds: float
    files_per_second: float
    total_files_processed: int


class CodeIndexingStats(TypedDict):
    """Statistics for code indexing."""

    files_newly_indexed: int
    files_unchanged: int
    total_indexed_files: int
    total_symbols: int
    total_references: int


class DocIndexingStats(TypedDict):
    """Statistics for documentation indexing."""

    files_newly_indexed: int
    files_unchanged: int
    total_indexed_files: int
    total_chunks: int
    docstrings_extracted: int


class FileIndexDetail(TypedDict):
    """Details about a single indexed file."""

    file: str
    symbols_indexed: int
    references_indexed: int


class DocIndexDetail(TypedDict):
    """Details about a single indexed documentation file."""

    file: str
    chunks_indexed: int


class IndexingDetails(TypedDict):
    """Detailed breakdown of indexed files."""

    code: list[FileIndexDetail]
    docs: list[DocIndexDetail]


class IndexCodebaseResponse(TypedDict):
    """Response from the index_codebase tool."""

    status: Literal["ok"]
    directory: str
    performance: IndexingPerformance
    code: CodeIndexingStats
    documentation: DocIndexingStats
    details: IndexingDetails


# ---------------------------------------------------------------------------
# search_docs Tool
# ---------------------------------------------------------------------------


class DocSearchResult(TypedDict):
    """A single result from documentation search."""

    content: str
    source_file: str
    section_title: str | None
    line_start: int
    line_end: int
    doc_type: str
    score: float
    context: NotRequired[list[ContextChunk]]


class SearchDocsResponse(TypedDict):
    """Response from the search_docs tool."""

    status: Literal["ok"]
    query: str
    results: list[DocSearchResult]
    count: int
    hint: NotRequired[str]


# ---------------------------------------------------------------------------
# search_history Tool - Commit Results
# ---------------------------------------------------------------------------


class CommitInfo(TypedDict):
    """Basic commit information."""

    hash: str
    full_hash: str
    message: str
    author: str
    author_email: str
    date: str


class SearchHistoryCommitsResponse(TypedDict):
    """Response from search_history with search_type='commits'."""

    status: Literal["ok"]
    search_type: Literal["commits"]
    query: str
    results: list[CommitInfo]


# ---------------------------------------------------------------------------
# search_history Tool - File History Results
# ---------------------------------------------------------------------------


class SearchHistoryFileHistoryResponse(TypedDict):
    """Response from search_history with search_type='file_history'."""

    status: Literal["ok"]
    search_type: Literal["file_history"]
    target_file: str
    results: list[CommitInfo]


# ---------------------------------------------------------------------------
# search_history Tool - Blame Results
# ---------------------------------------------------------------------------


class BlameEntry(TypedDict):
    """A single blame entry (grouped consecutive lines)."""

    line_start: int
    line_end: int
    commit_hash: str
    author: str
    date: str
    line_content: str
    commit_message: str


class SearchHistoryBlameResponse(TypedDict):
    """Response from search_history with search_type='blame'."""

    status: Literal["ok"]
    search_type: Literal["blame"]
    target_file: str
    results: list[BlameEntry]


# ---------------------------------------------------------------------------
# search_history Tool - Commit Detail Results
# ---------------------------------------------------------------------------


class CommitDetail(TypedDict):
    """Detailed commit information with diff."""

    hash: str
    full_hash: str
    message: str
    author: str
    author_email: str
    date: str
    parent_hashes: list[str]
    files_changed: list[FileChangeStats]
    diff: str | None


class SearchHistoryCommitDetailResponse(TypedDict):
    """Response from search_history with search_type='commit_detail'."""

    status: Literal["ok"]
    search_type: Literal["commit_detail"]
    result: CommitDetail


# ---------------------------------------------------------------------------
# search_history Tool - Union Response Type
# ---------------------------------------------------------------------------


SearchHistoryResponse = (
    SearchHistoryCommitsResponse
    | SearchHistoryFileHistoryResponse
    | SearchHistoryBlameResponse
    | SearchHistoryCommitDetailResponse
    | ErrorResponse
)


# ---------------------------------------------------------------------------
# Tool Response Union Types
# ---------------------------------------------------------------------------

# Union of all possible tool responses (for documentation purposes)
ToolResponse = (
    CheckIndexStatusResponse
    | GetIndexStatsResponse
    | SearchCodeResponse
    | IndexCodebaseResponse
    | SearchDocsResponse
    | SearchHistoryResponse
    | ErrorResponse
)
