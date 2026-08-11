"""PDF upload and indexing routes with security checks and database persistence."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from middleware.rate_limiter import limiter
from models import DocumentModel, User
from routes.schemas import UploadResponse
from services.rag_service import RAGService
from utils.files import compute_file_hash

logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit per file


@router.post("/upload", response_model=UploadResponse)
@limiter.limit("20 per minute")
async def upload_pdfs(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more PDF files"),
    session_id: str | None = None,
    reindex: bool = False,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    Upload PDFs, extract text, chunk, embed, store in ChromaDB, and register in PostgreSQL.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    service = RAGService.get_instance(session_id)

    if reindex:
        service.reset()
        service = RAGService.get_instance(session_id)

    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        filename = upload.filename or "file.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported: {filename}",
            )
        data = await upload.read()
        if not data:
            raise HTTPException(
                status_code=400,
                detail=f"Empty file: {filename}",
            )
        if len(data) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds maximum allowed size of 50MB: {filename}",
            )
        if not data.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail=f"File is not a valid PDF document header: {filename}",
            )
        payloads.append((filename, data))

    try:
        summary = service.process_upload_bytes(payloads)
        try:
            service.initialize_llm()
        except Exception as exc:
            logger.warning("Upload OK but LLM init failed: %s", exc)

        # Register uploaded documents into PostgreSQL DB
        for filename, data in payloads:
            file_hash = compute_file_hash(data) if isinstance(data, (bytes, bytearray)) else ""
            doc_entry = DocumentModel(
                user_id=current_user.id if current_user else None,
                session_id=service.session_id,
                filename=filename,
                file_hash=file_hash,
                chunk_count=summary.get("chunks_added", 0),
            )
            db.add(doc_entry)
        db.commit()

    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UploadResponse(
        chunks_added=summary.get("chunks_added", 0),
        files_processed=summary.get("files_processed", 0),
        duplicates_skipped=summary.get("duplicates_skipped", 0),
        total_chunks=summary.get("total_chunks", 0),
        message=summary.get("message"),
    )
