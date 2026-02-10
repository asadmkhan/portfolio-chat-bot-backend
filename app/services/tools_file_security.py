from __future__ import annotations

import ipaddress
from io import BytesIO
import mimetypes
import re
import socket
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse, urlunparse
from zipfile import ZipFile

RESUME_CONTENT_TYPE_EXTENSION_HINTS = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/markdown": "md",
    "application/rtf": "rtf",
    "text/rtf": "rtf",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
}

PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
GIF_MAGICS = (b"GIF87a", b"GIF89a")
BMP_MAGIC = b"BM"
WEBP_RIFF_MAGIC = b"RIFF"
WEBP_WEBP_MAGIC = b"WEBP"
ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def _safe_str(value: Any, max_len: int = 255) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text[:max_len]


def normalize_public_url(raw_url: str, *, field_label: str) -> tuple[str, str]:
    value = (raw_url or "").strip()
    if not value:
        raise ValueError(f"{field_label} is required.")
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"https://{value}"
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http/https {field_label.lower()}s are supported.")
    if not parsed.netloc:
        raise ValueError(f"Invalid {field_label.lower()}.")
    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        raise ValueError(f"Invalid {field_label.lower()} host.")
    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            "",
            parsed.query,
            "",
        )
    )
    return normalized, hostname


def normalize_job_url(raw_url: str) -> tuple[str, str]:
    return normalize_public_url(raw_url, field_label="Job URL")


def normalize_resume_url(raw_url: str) -> tuple[str, str]:
    return normalize_public_url(raw_url, field_label="Resume URL")


def host_is_private_or_local(hostname: str) -> bool:
    host = (hostname or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        pass
    try:
        for _family, _socktype, _proto, _canon, sockaddr in socket.getaddrinfo(host, None):
            address = sockaddr[0] if sockaddr else ""
            if not address:
                continue
            try:
                resolved = ipaddress.ip_address(address)
            except ValueError:
                continue
            if resolved.is_private or resolved.is_loopback or resolved.is_link_local or resolved.is_reserved:
                return True
    except Exception:
        return False
    return False


def _normalize_google_drive_url(parsed: Any) -> str | None:
    host = (parsed.netloc or "").lower()
    if "drive.google.com" not in host:
        return None
    path = parsed.path or ""
    query = parse_qs(parsed.query or "")
    file_id = ""
    match = re.search(r"/file/d/([^/]+)", path)
    if match:
        file_id = match.group(1)
    if not file_id:
        file_id = _safe_str((query.get("id") or [""])[0], max_len=200)
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _normalize_dropbox_url(parsed: Any) -> str | None:
    host = (parsed.netloc or "").lower()
    if "dropbox.com" not in host and "dropboxusercontent.com" not in host:
        return None
    path = parsed.path or ""
    if not path:
        return None
    query = parse_qs(parsed.query or "")
    query["dl"] = ["1"]
    rebuilt_query = "&".join(f"{key}={value}" for key, values in query.items() for value in values)
    netloc = parsed.netloc.lower().replace("www.", "")
    if "dropbox.com" in netloc:
        netloc = netloc.replace("dropbox.com", "dl.dropboxusercontent.com")
    return urlunparse(("https", netloc, path, "", rebuilt_query, ""))


def _normalize_onedrive_url(parsed: Any) -> str | None:
    host = (parsed.netloc or "").lower()
    if "1drv.ms" not in host and "onedrive.live.com" not in host:
        return None
    query = parse_qs(parsed.query or "")
    query["download"] = ["1"]
    rebuilt_query = "&".join(f"{key}={value}" for key, values in query.items() for value in values)
    return urlunparse(("https", parsed.netloc.lower(), parsed.path or "/", "", rebuilt_query, ""))


def normalize_resume_download_url(url: str) -> str:
    parsed = urlparse(url)
    for resolver in (_normalize_google_drive_url, _normalize_dropbox_url, _normalize_onedrive_url):
        resolved = resolver(parsed)
        if resolved:
            return resolved
    return url


def _is_zip_payload(content: bytes) -> bool:
    return any(content.startswith(prefix) for prefix in ZIP_MAGICS)


def _zip_has_paths(content: bytes, prefixes: tuple[str, ...]) -> bool:
    try:
        with ZipFile(BytesIO(content)) as archive:
            names = archive.namelist()
        return any(any(name.startswith(prefix) for prefix in prefixes) for name in names)
    except Exception:
        return False


def _is_probably_text_payload(content: bytes) -> bool:
    if not content:
        return False
    sample = content[:4096]
    if b"\x00" in sample:
        return False
    printable = 0
    for byte in sample:
        if byte in (9, 10, 13) or 32 <= byte <= 126:
            printable += 1
    return (printable / len(sample)) >= 0.75


def _looks_like_mp4_family(content: bytes) -> bool:
    if len(content) < 12:
        return False
    if content[4:8] != b"ftyp":
        return False
    major = content[8:12]
    return major in {
        b"isom",
        b"iso2",
        b"avc1",
        b"mp41",
        b"mp42",
        b"M4V ",
        b"qt  ",
        b"MSNV",
    }


def validate_upload_signature(*, filename: str, content: bytes) -> None:
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    if ext == "doc":
        raise ValueError("Legacy .doc is not supported. Convert to .docx.")

    if ext == "pdf":
        if not content.startswith(PDF_MAGIC):
            raise ValueError("File signature does not match .pdf content.")
        return

    if ext == "docx":
        if not _is_zip_payload(content) or not _zip_has_paths(content, ("word/",)):
            raise ValueError("File signature does not match .docx content.")
        return

    if ext == "pptx":
        if not _is_zip_payload(content) or not _zip_has_paths(content, ("ppt/",)):
            raise ValueError("File signature does not match .pptx content.")
        return

    if ext == "ppt":
        raise ValueError("Legacy .ppt files are not supported. Please upload .pptx or paste text.")

    if ext in {"txt", "md", "rtf"}:
        if not _is_probably_text_payload(content):
            raise ValueError(f"File signature does not match .{ext} text content.")
        return

    if ext == "png":
        if not content.startswith(PNG_MAGIC):
            raise ValueError("File signature does not match .png content.")
        return

    if ext in {"jpg", "jpeg"}:
        if not content.startswith(JPEG_MAGIC):
            raise ValueError("File signature does not match .jpg/.jpeg content.")
        return

    if ext == "gif":
        if not any(content.startswith(magic) for magic in GIF_MAGICS):
            raise ValueError("File signature does not match .gif content.")
        return

    if ext == "bmp":
        if not content.startswith(BMP_MAGIC):
            raise ValueError("File signature does not match .bmp content.")
        return

    if ext == "webp":
        if len(content) < 12 or not content.startswith(WEBP_RIFF_MAGIC) or content[8:12] != WEBP_WEBP_MAGIC:
            raise ValueError("File signature does not match .webp content.")
        return

    if ext in {"mp4", "mov", "m4v"}:
        if not _looks_like_mp4_family(content):
            raise ValueError(f"File signature does not match .{ext} content.")
        return

    if ext in {"avi", "mkv", "webm"} and len(content) < 4:
        raise ValueError(f"File signature does not match .{ext} content.")


def extract_filename_from_content_disposition(value: str, safe_str: Callable[[Any, int], str] | None = None) -> str:
    sanitizer = safe_str or _safe_str
    if not value:
        return ""
    filename_star = re.search(r"filename\\*=UTF-8''([^;]+)", value, flags=re.IGNORECASE)
    if filename_star:
        try:
            return sanitizer(re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), filename_star.group(1)), 255)
        except Exception:
            pass
    filename = re.search(r'filename=\"?([^\";]+)\"?', value, flags=re.IGNORECASE)
    if filename:
        return sanitizer(filename.group(1).strip(), 255)
    return ""


def extension_from_content_type(content_type: str, safe_str: Callable[[Any, int], str] | None = None) -> str:
    sanitizer = safe_str or _safe_str
    sanitized = sanitizer(content_type.split(";")[0], 120).lower()
    if not sanitized:
        return ""
    explicit = RESUME_CONTENT_TYPE_EXTENSION_HINTS.get(sanitized)
    if explicit:
        return explicit
    guessed = mimetypes.guess_extension(sanitized) or ""
    return guessed.lstrip(".").lower()


def filename_from_url_and_headers(final_url: str, headers: Any, safe_str: Callable[[Any, int], str] | None = None) -> str:
    sanitizer = safe_str or _safe_str
    disposition = ""
    try:
        disposition = sanitizer(headers.get("content-disposition"), 500)
    except Exception:
        disposition = ""
    filename = extract_filename_from_content_disposition(disposition, sanitizer)
    if filename:
        return filename
    path_name = (urlparse(final_url).path or "").split("/")[-1].strip()
    return sanitizer(path_name, 255)


def extension_from_filename(filename: str, safe_str: Callable[[Any, int], str] | None = None) -> str:
    sanitizer = safe_str or _safe_str
    if "." not in filename:
        return ""
    return sanitizer(filename.rsplit(".", 1)[-1], 20).lower()


def safe_resume_filename(
    final_url: str,
    headers: Any,
    content_type: str,
    safe_str: Callable[[Any, int], str] | None = None,
) -> str:
    sanitizer = safe_str or _safe_str
    filename = filename_from_url_and_headers(final_url, headers, sanitizer)
    ext = extension_from_filename(filename, sanitizer)
    if not ext:
        guessed_ext = extension_from_content_type(content_type, sanitizer)
        if guessed_ext:
            base = sanitizer(filename or "resume", 220).rstrip(".")
            filename = f"{base}.{guessed_ext}"
    if not filename:
        filename = "resume.txt"
    return filename
