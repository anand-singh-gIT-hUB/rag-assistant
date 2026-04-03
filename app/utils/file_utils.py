"""
app/utils/file_utils.py
"""
import hashlib
import shutil
from pathlib import Path

from app.core.constants import SUPPORTED_EXTENSIONS
from app.core.exceptions import UnsupportedFileTypeError, StorageError


def validate_extension(filename: str) -> str:
    """Return the normalised extension or raise UnsupportedFileTypeError."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix}",
            detail=f"Allowed types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    return suffix


def save_upload(source_bytes: bytes, dest_path: Path) -> None:
    """Write bytes to *dest_path*, creating parent dirs as needed."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(source_bytes)
    except OSError as e:
        raise StorageError(f"Failed to save file: {e}") from e


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_delete(path: Path) -> None:
    """Delete file if it exists, suppress errors."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
