"""
DocumentParser — converte PDF e DOCX em List[Chunk] para indexação no RAG.

Estratégia de chunking: recursive split — tenta dividir por \n\n, depois \n,
depois ". ", depois " ", garantindo chunks de até _CHUNK_SIZE caracteres.
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

# Comprimento mínimo em caracteres — evita ruído de títulos curtos, números de página, etc.
MIN_CHUNK_LENGTH = 20

# Parâmetros do recursive split — equivalente ao RecursiveCharacterTextSplitter do LangChain,
# implementado sem a dependência pesada.
_CHUNK_SIZE    = 500  # tamanho máximo de cada chunk em caracteres
_CHUNK_OVERLAP = 50   # sobreposição entre chunks consecutivos (evita cortar regras no meio)
_SEPARATORS    = ["\n\n", "\n", ". ", " "]  # hierarquia de divisão: parágrafo → linha → frase → palavra


def _recursive_split(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """
    Divide texto em chunks de até chunk_size caracteres.

    Tenta cada separador em _SEPARATORS até encontrar um que produza mais de uma parte.
    Agrupa partes pequenas consecutivas até atingir chunk_size, com sobreposição de overlap.
    Fallback: divisão por tamanho fixo se nenhum separador funcionar.
    """
    for sep in _SEPARATORS:
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        if len(parts) > 1:
            break
    else:
        # Nenhum separador produziu divisão — divide por tamanho fixo
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Se a própria parte for maior que chunk_size, divide recursivamente
            if len(part) > chunk_size:
                chunks.extend(_recursive_split(part, chunk_size, overlap))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)

    return chunks


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
                    blocks.extend(_recursive_split(text))
        return blocks

    def _read_docx(self, file_path: str) -> List[str]:
        doc = docx.Document(file_path)
        full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return _recursive_split(full_text)
