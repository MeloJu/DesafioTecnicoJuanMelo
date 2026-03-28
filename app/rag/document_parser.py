"""
DocumentParser — converte PDF e DOCX em List[Chunk] para indexação no RAG.

Estratégia de chunking: divisão por parágrafo (dupla quebra de linha).
Blocos com menos de MIN_CHUNK_LENGTH caracteres são descartados — evitam
ruído semântico (títulos de seção, números de página, etc.).

Formatos suportados: .pdf (pdfplumber), .docx (python-docx).
"""
import uuid
from pathlib import Path
from typing import List

import docx
import pdfplumber

from app.logging.logger import get_logger
from app.schemas.output import Chunk

log = get_logger()

MIN_CHUNK_LENGTH = 20


class DocumentParser:
    _READERS: dict = {
        ".pdf":  "_read_pdf",
        ".docx": "_read_docx",
    }

    def parse(self, file_path: str, empresa: str, setor: str) -> List[Chunk]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        reader_name = self._READERS.get(suffix)

        if not reader_name:
            supported = list(self._READERS)
            raise ValueError(f"Unsupported file format: '{suffix}'. Use {supported}.")

        blocks = getattr(self, reader_name)(file_path)

        chunks = [
            Chunk(
                text=block.strip(),
                source=path.name,
                empresa=empresa,
                setor=setor,
                chunk_id=str(uuid.uuid4()),
            )
            for block in blocks
            if block and len(block.strip()) >= MIN_CHUNK_LENGTH
        ]

        log.info(
            "document_parsed",
            source=path.name,
            empresa=empresa,
            setor=setor,
            total_chunks=len(chunks),
        )
        return chunks

    def _read_pdf(self, file_path: str) -> List[str]:
        blocks: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    blocks.extend(text.split("\n\n"))
        return blocks

    def _read_docx(self, file_path: str) -> List[str]:
        doc = docx.Document(file_path)
        return [p.text for p in doc.paragraphs]
