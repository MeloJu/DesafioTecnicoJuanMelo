"""
Unit tests for the RAG document parser.

What we test (pdfplumber and python-docx always mocked):
- PDF: extrai texto por página, divide em chunks por parágrafo
- DOCX: extrai parágrafos do documento
- Metadados: source, empresa, setor corretos em todos os chunks
- chunk_id: único por chunk, string não vazia
- Filtro: blocos curtos/vazios são descartados
- Formato não suportado → ValueError
- PDF sem texto (página vazia) → lista vazia, sem erro
"""
import uuid
import pytest
from unittest.mock import MagicMock, Mock, patch, PropertyMock

from app.rag.document_parser import DocumentParser
from app.schemas.output import Chunk


EMPRESA = "Construtiva"
SETOR = "obras"


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

class TestDocumentParserPdf:
    def _make_pdf_mock(self, pages_text: list[str]):
        """Build a pdfplumber context manager mock from a list of page texts."""
        mock_pages = []
        for text in pages_text:
            page = Mock()
            page.extract_text.return_value = text
            mock_pages.append(page)

        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        return mock_pdf

    @patch("app.rag.document_parser.pdfplumber")
    def test_pdf_returns_list_of_chunks(self, mock_pdfplumber):
        mock_pdfplumber.open.return_value = self._make_pdf_mock(
            ["Capacete obrigatório em obras.\n\nColete refletivo obrigatório."]
        )
        parser = DocumentParser()
        chunks = parser.parse("manual.pdf", EMPRESA, SETOR)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    @patch("app.rag.document_parser.pdfplumber")
    def test_pdf_metadata_set_correctly(self, mock_pdfplumber):
        mock_pdfplumber.open.return_value = self._make_pdf_mock(
            ["Capacete obrigatório em obras.\n\nColete refletivo obrigatório."]
        )
        parser = DocumentParser()
        chunks = parser.parse("manual.pdf", EMPRESA, SETOR)

        for chunk in chunks:
            assert chunk.empresa == EMPRESA
            assert chunk.setor == SETOR
            assert chunk.source == "manual.pdf"

    @patch("app.rag.document_parser.pdfplumber")
    def test_pdf_multiple_pages_all_parsed(self, mock_pdfplumber):
        mock_pdfplumber.open.return_value = self._make_pdf_mock([
            "Capacete obrigatório em todas as áreas de obras.",
            "Colete refletivo obrigatório para operadores.",
        ])
        parser = DocumentParser()
        chunks = parser.parse("manual.pdf", EMPRESA, SETOR)
        assert len(chunks) == 2

    @patch("app.rag.document_parser.pdfplumber")
    def test_pdf_empty_page_ignored(self, mock_pdfplumber):
        mock_pdfplumber.open.return_value = self._make_pdf_mock([None, "Capacete obrigatório em todas as áreas de obras."])
        parser = DocumentParser()
        chunks = parser.parse("manual.pdf", EMPRESA, SETOR)
        assert len(chunks) == 1

    @patch("app.rag.document_parser.pdfplumber")
    def test_pdf_all_empty_pages_returns_empty(self, mock_pdfplumber):
        mock_pdfplumber.open.return_value = self._make_pdf_mock([None, None])
        parser = DocumentParser()
        chunks = parser.parse("manual.pdf", EMPRESA, SETOR)
        assert chunks == []


# ---------------------------------------------------------------------------
# DOCX parsing
# ---------------------------------------------------------------------------

class TestDocumentParserDocx:
    def _make_docx_mock(self, paragraphs: list[str]):
        mock_doc = Mock()
        mock_doc.paragraphs = [Mock(text=p) for p in paragraphs]
        return mock_doc

    @patch("app.rag.document_parser.docx")
    def test_docx_returns_list_of_chunks(self, mock_docx):
        mock_docx.Document.return_value = self._make_docx_mock([
            "Capacete obrigatório em obras.",
            "Colete refletivo obrigatório.",
        ])
        parser = DocumentParser()
        chunks = parser.parse("manual.docx", EMPRESA, SETOR)
        assert len(chunks) == 2

    @patch("app.rag.document_parser.docx")
    def test_docx_metadata_set_correctly(self, mock_docx):
        mock_docx.Document.return_value = self._make_docx_mock([
            "Capacete obrigatório em obras.",
        ])
        parser = DocumentParser()
        chunks = parser.parse("guia.docx", EMPRESA, SETOR)

        assert chunks[0].empresa == EMPRESA
        assert chunks[0].setor == SETOR
        assert chunks[0].source == "guia.docx"

    @patch("app.rag.document_parser.docx")
    def test_docx_empty_paragraphs_filtered(self, mock_docx):
        mock_docx.Document.return_value = self._make_docx_mock([
            "",
            "   ",
            "Regra válida com texto suficiente aqui.",
        ])
        parser = DocumentParser()
        chunks = parser.parse("manual.docx", EMPRESA, SETOR)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# chunk_id
# ---------------------------------------------------------------------------

class TestChunkId:
    @patch("app.rag.document_parser.pdfplumber")
    def test_chunk_ids_are_unique(self, mock_pdfplumber):
        pages_text = "Regra A suficientemente longa.\n\nRegra B suficientemente longa."
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock(extract_text=Mock(return_value=pages_text))]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        parser = DocumentParser()
        chunks = parser.parse("manual.pdf", EMPRESA, SETOR)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    @patch("app.rag.document_parser.pdfplumber")
    def test_chunk_id_is_valid_uuid(self, mock_pdfplumber):
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock(extract_text=Mock(return_value="Regra válida com texto suficiente."))]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        parser = DocumentParser()
        chunks = parser.parse("manual.pdf", EMPRESA, SETOR)

        for chunk in chunks:
            uuid.UUID(chunk.chunk_id)  # raises if invalid


# ---------------------------------------------------------------------------
# Short blocks filtered
# ---------------------------------------------------------------------------

class TestShortBlockFilter:
    @patch("app.rag.document_parser.pdfplumber")
    def test_very_short_blocks_discarded(self, mock_pdfplumber):
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock(extract_text=Mock(return_value="Ok\n\nEsta regra tem texto suficiente para ser um chunk."))]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)
        mock_pdfplumber.open.return_value = mock_pdf

        parser = DocumentParser()
        chunks = parser.parse("manual.pdf", EMPRESA, SETOR)

        texts = [c.text for c in chunks]
        assert not any(len(t) < 20 for t in texts)


# ---------------------------------------------------------------------------
# Unsupported format
# ---------------------------------------------------------------------------

class TestUnsupportedFormat:
    def test_unsupported_extension_raises_value_error(self):
        parser = DocumentParser()
        with pytest.raises(ValueError, match="Unsupported"):
            parser.parse("manual.txt", EMPRESA, SETOR)

    def test_xlsx_raises_value_error(self):
        parser = DocumentParser()
        with pytest.raises(ValueError, match="Unsupported"):
            parser.parse("manual.xlsx", EMPRESA, SETOR)
