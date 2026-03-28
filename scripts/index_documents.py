"""
Script de indexação: lê todos os company.yaml em data/raw/ e indexa no ChromaDB.

Para adicionar uma nova empresa:
  1. Crie a pasta: data/raw/<NomeDaEmpresa>/
  2. Coloque o PDF/DOCX dentro
  3. Crie data/raw/<NomeDaEmpresa>/company.yaml com:
       empresa: Nome da Empresa
       setor: setor_da_empresa
       doc: nome_do_arquivo.pdf
       images_folder: images

Uso:
    .venv/Scripts/python scripts/index_documents.py

O ChromaDB é criado em ./chroma_db (persistente).
Só precisa rodar quando documentos são adicionados ou alterados.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from app.rag.document_parser import DocumentParser
from app.rag.embedding_service import EmbeddingService
from app.rag.ollama_embedder import OllamaEmbedder
from scripts.utils import discover_companies

DATA_ROOT = Path("data/raw")
CHROMA_PATH = "./chroma_db"


def main():
    companies = discover_companies(DATA_ROOT)

    if not companies:
        print(f"Nenhum company.yaml encontrado em {DATA_ROOT}/")
        print("Crie data/raw/<empresa>/company.yaml para cada empresa.")
        return

    print(f"Encontradas {len(companies)} empresa(s).\n")

    parser = DocumentParser()
    embedder = OllamaEmbedder(model="nomic-embed-text")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_service = EmbeddingService(
        chroma_client=chroma_client,
        embedding_client=embedder,
    )

    total_chunks = 0

    for company in companies:
        empresa = company["empresa"]
        setor = company["setor"]
        doc_path = company["folder"] / company["doc"]

        print(f"Indexando: {empresa} ({setor})")
        print(f"  Arquivo: {doc_path.name}")

        if not doc_path.exists():
            print(f"  [AVISO] Arquivo não encontrado: {doc_path}")
            print()
            continue

        try:
            chunks = parser.parse(str(doc_path), empresa=empresa, setor=setor)
            print(f"  Chunks extraídos: {len(chunks)}")

            if not chunks:
                print("  [AVISO] Nenhum chunk extraído — documento pode estar vazio ou muito curto")
                print()
                continue

            collection = re.sub(r"[^a-z0-9]+", "_", empresa.lower()).strip("_")
            embedding_service.index(chunks, collection=collection)
            print(f"  Collection: '{collection}' ✓")
            total_chunks += len(chunks)

        except Exception as exc:
            print(f"  [ERRO] {exc}")

        print()

    print(f"Indexação concluída. Total: {total_chunks} chunks → {CHROMA_PATH}/")


if __name__ == "__main__":
    main()
