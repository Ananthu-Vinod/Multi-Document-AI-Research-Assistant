"""PDF upload and indexing routes."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from routes.schemas import UploadResponse
from services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    files: list[UploadFile] = File(..., description="One or more PDF files"),
    session_id: str | None = None,
    reindex: bool = False,
) -> UploadResponse:
    """
    Upload PDFs, extract text, chunk, embed, and store in ChromaDB.

    Set reindex=true to clear the index before processing (full re-index).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    service = RAGService.get_instance(session_id)

    if reindex:
        service.reset()
        service = RAGService.get_instance(session_id)

    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported: {upload.filename}",
            )
        data = await upload.read()
        if not data:
            raise HTTPException(
                status_code=400,
                detail=f"Empty file: {upload.filename}",
            )
        payloads.append((upload.filename, data))

    try:
        summary = service.process_upload_bytes(payloads)
        try:
            service.initialize_llm()
        except Exception as exc:
            logger.warning("Upload OK but LLM init failed: %s", exc)
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
