# Compliance AI — Sistema de Verificação de EPIs

> Você tem uma câmera, uma foto de um trabalhador e um documento com as regras da empresa.
> Esse sistema lê tudo isso e te diz: **"Esse trabalhador está em conformidade ou não?"** — com explicação.

---

## O que esse sistema faz?

Imagine que você é um fiscal de obras e precisa verificar se cada trabalhador está usando os equipamentos de proteção individual (EPI) obrigatórios: capacete, colete, botas, luvas.

Fazer isso manualmente é lento e sujeito a erro. Esse sistema faz isso automaticamente:

1. **Recebe uma foto** de um trabalhador
2. **Detecta a pessoa** na imagem (onde ela está, coordenadas exatas)
3. **Analisa o que ela está usando** (capacete? colete? botas?)
4. **Consulta as regras da empresa** (ex: "capacete obrigatório em obras", do PDF da empresa)
5. **Decide se está em conformidade** com base nas regras
6. **Retorna um JSON estruturado** com a decisão e uma justificativa obrigatória

### Exemplo de saída:

```json
{
  "results": [
    {
      "pessoa_id": 1,
      "bbox": { "x1": 45.0, "y1": 12.0, "x2": 180.0, "y2": 410.0 },
      "status": "Não conforme",
      "justificativa": "Capacete ausente — viola NR-18 item 18.23.1 exigido para área de obras."
    }
  ]
}
```

---

## Como o sistema funciona por dentro?

O sistema é dividido em **4 módulos independentes** que se comunicam em sequência, como uma linha de montagem:

```
Imagem → [Vision] → [RAG] → [Reasoning] → JSON de saída
```

### Módulo 1: Vision (Visão Computacional)

Usa dois modelos de IA especializados:

- **YOLO**: detecta onde as pessoas estão na imagem (retorna "caixas" ao redor de cada pessoa)
- **CLIP**: olha para cada pessoa detectada e classifica os atributos um por um:
  - "Esta pessoa está usando capacete?" → sim / não / incerto
  - "Esta pessoa está usando colete?" → sim / não / incerto
  - E assim por diante para botas e luvas

O CLIP usa uma técnica chamada *zero-shot*: ele compara a imagem com frases em inglês como *"a person wearing a hard hat"* vs *"a person not wearing a hard hat"* e escolhe qual se encaixa melhor. Não precisa ser treinado com fotos de capacetes — ele já sabe.

### Módulo 2: RAG (Recuperação de Regras)

RAG significa *Retrieval-Augmented Generation* — basicamente: **busca inteligente em documentos**.

A empresa tem PDFs ou documentos Word com as regras de segurança. O sistema:
1. Lê esses documentos e os divide em trechos menores (chunks)
2. Converte cada trecho em um "vetor" (representação matemática do significado)
3. Armazena esses vetores em um banco de dados especializado (ChromaDB)
4. Quando precisa verificar conformidade, busca os trechos mais relevantes para a situação

Por exemplo: se o CLIP detectou que o trabalhador não tem capacete, o sistema busca no documento da empresa os trechos que falam sobre capacete — e encontra a regra exata que está sendo violada.

### Módulo 3: Reasoning (Raciocínio)

Usa o **Llama** (um LLM — Large Language Model, como o ChatGPT mas rodando localmente) para:
1. Receber os atributos detectados pela visão
2. Receber as regras recuperadas pelo RAG
3. Analisar se há conformidade
4. Gerar uma justificativa em linguagem natural explicando o porquê

O Llama roda **100% localmente** via Ollama — sem internet, sem custo por chamada.

### Módulo 4: Pipeline (Orquestrador)

É o "maestro" que coordena tudo. Recebe a imagem e o nome da empresa/setor, e devolve o resultado final. Também gera um `correlation_id` único por requisição para rastrear cada análise nos logs.

---

## Arquitetura em diagrama

```
                    ┌─────────────────────────────────┐
                    │         Pipeline.run()           │
                    │  empresa="Construtiva"           │
                    │  setor="obras"                   │
                    │  image="foto.jpg"                │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │           VisionService                  │
              │  ┌──────────────┐  ┌──────────────────┐ │
              │  │ PersonDetector│  │AttributeExtractor│ │
              │  │    (YOLO)    │  │     (CLIP)       │ │
              │  └──────────────┘  └──────────────────┘ │
              └────────────────────┬────────────────────┘
                                   │ List[PersonDetection]
              ┌────────────────────▼────────────────────┐
              │             RagService                   │
              │  ┌──────────────────────────────────┐   │
              │  │ EmbeddingService + ChromaDB       │   │
              │  │ nomic-embed-text (via Ollama)     │   │
              │  └──────────────────────────────────┘   │
              └────────────────────┬────────────────────┘
                                   │ List[Rule]
              ┌────────────────────▼────────────────────┐
              │          ReasoningService                │
              │  ┌──────────────────────────────────┐   │
              │  │   OllamaLLM (llama3.2)           │   │
              │  └──────────────────────────────────┘   │
              └────────────────────┬────────────────────┘
                                   │ PipelineResponse
                    ┌──────────────▼──────────────────┐
                    │   JSON estruturado com status    │
                    │   e justificativa por pessoa     │
                    └─────────────────────────────────┘
```

---

## Estrutura de pastas

```
project/
├── app/                      ← código de produção
│   ├── schemas/              ← contratos de dados (Pydantic v2)
│   │   └── output.py         ← BoundingBox, PersonDetection, Rule, PersonResult...
│   ├── logging/              ← log estruturado JSON com correlation_id
│   │   └── logger.py
│   ├── pipeline/             ← orquestrador + factory (composição de dependências)
│   │   ├── orchestrator.py
│   │   └── factory.py
│   ├── vision/               ← detecção de pessoas + classificação de EPIs
│   │   ├── detector.py       ← YOLO
│   │   ├── extractor.py      ← CLIP com thresholds configuráveis
│   │   ├── service.py        ← fachada: detector + extractor
│   │   └── clip_client.py    ← wrapper HuggingFace
│   ├── rag/                  ← parse de documentos + retrieval semântico
│   │   ├── document_parser.py ← PDF + DOCX → List[Chunk]
│   │   ├── embedding_service.py ← ChromaDB index + query
│   │   ├── service.py        ← fachada: embedding → List[Rule]
│   │   └── ollama_embedder.py ← wrapper nomic-embed-text
│   └── reasoning/            ← lógica de conformidade via LLM
│       ├── service.py        ← analisa pessoa + regras → PersonResult
│       └── ollama_llm.py     ← wrapper Llama via Ollama
│
├── tests/
│   ├── unit/                 ← testes unitários (100% cobertura obrigatória)
│   ├── integration/          ← testes de integração (2+ módulos reais)
│   ├── e2e/                  ← testes end-to-end (pipeline completo real)
│   └── fixtures/
│       ├── images/           ← fotos reais de trabalhadores (para E2E)
│       └── documents/        ← documentos de regras gerados nos testes E2E
│
├── data/                     ← PDFs/DOCXs das empresas (ignorado pelo git)
├── chroma_db/                ← banco vetorial persistente (ignorado pelo git)
├── pyproject.toml            ← config pytest + cobertura
└── requirements.txt          ← dependências
```

---

## Como usar

### Pré-requisitos

1. **Python 3.11+**
2. **Ollama** instalado e rodando com os modelos:
   ```bash
   ollama serve
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Instalar as dependências:
   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # Windows
   pip install -r requirements.txt
   ```

### Indexar os documentos da empresa

Antes de usar o pipeline, você precisa indexar os documentos PDF/DOCX de cada empresa no ChromaDB:

```python
from app.rag.document_parser import DocumentParser
from app.rag.embedding_service import EmbeddingService
from app.rag.ollama_embedder import OllamaEmbedder
import chromadb

# Configurar
parser = DocumentParser()
embedder = OllamaEmbedder()
chroma = chromadb.PersistentClient(path="./chroma_db")
embedding_service = EmbeddingService(chroma_client=chroma, embedding_client=embedder)

# Indexar PDF ou DOCX
chunks = parser.parse("data/normas_construtiva.pdf", empresa="Construtiva", setor="obras")
embedding_service.index(chunks, collection="construtiva")
```

### Rodar o pipeline

```python
from app.pipeline.factory import create_pipeline

pipeline = create_pipeline()
response = pipeline.run("foto_trabalhador.jpg", empresa="Construtiva", setor="obras")

for result in response.results:
    print(f"Pessoa {result.pessoa_id}: {result.status}")
    print(f"  Justificativa: {result.justificativa}")
```

---

## Como rodar os testes

```bash
# Testes unitários (rápido, sem modelos reais)
pytest tests/unit/ --cov=app

# Testes de integração (sem modelos reais, apenas lógica de módulos)
pytest tests/integration/

# Testes E2E (requer Ollama rodando + fotos reais em tests/fixtures/images/)
pytest tests/e2e/ -m e2e -s
```

---

## Stack tecnológica

| Componente | Tecnologia | Por quê |
|---|---|---|
| Detecção de pessoas | YOLO (Ultralytics) | Estado da arte em detecção em tempo real |
| Classificação de EPIs | CLIP (HuggingFace) | Zero-shot — não precisa de treinamento específico |
| Embeddings (RAG) | nomic-embed-text via Ollama | Leve, eficiente, roda localmente |
| Banco vetorial | ChromaDB | Simples, persistente em disco, sem servidor separado |
| LLM (raciocínio) | Llama 3.2 via Ollama | Gratuito, local, sem dependência de API externa |
| Schemas | Pydantic v2 | Validação rigorosa de contratos de dados |
| Logging | structlog (JSON) | Rastreabilidade completa via correlation_id |
| Testes | pytest + pytest-cov | 100% cobertura unitária exigida pelo CI |

**Princípio central:** Nenhuma API externa, nenhum custo por chamada, nenhum dado sai da máquina.

---

## Decisões de design

### Por que separar Vision em dois sub-módulos?

YOLO e CLIP fazem coisas diferentes: YOLO encontra *onde* a pessoa está, CLIP classifica *o que* ela está usando. Separar em `PersonDetector` e `AttributeExtractor` permite testar cada um independentemente e trocar a implementação sem afetar o outro.

### Por que usar CLIP zero-shot em vez de treinar um classificador?

Treinar um classificador específico para capacetes exigiria centenas de fotos rotuladas e tempo de treinamento. CLIP já entende linguagem e visão de forma geral — basta descrever o que você quer detectar em texto. A desvantagem é menor precisão em casos ambíguos, tratada pela "zona de incerteza" (quando o score está entre os thresholds, retorna `None` em vez de forçar True/False).

### Por que RAG em vez de passar todas as regras para o LLM?

Documentos de empresas podem ser longos (dezenas de páginas). Passar tudo para o LLM a cada análise é caro (tokens) e lento. O RAG recupera apenas os trechos relevantes para a situação atual — tipicamente 5 trechos de algumas linhas cada.

### Por que Ollama em vez de OpenAI/Anthropic?

100% local, sem custo por chamada, sem envio de dados para terceiros. Em produção industrial, isso é frequentemente exigido por requisitos de conformidade (irônico para um sistema de compliance).

### Por que TDD?

Sistemas de IA são difíceis de testar porque os modelos são não-determinísticos. TDD força a separação entre "lógica de orquestração" (determinística, 100% testável) e "chamada ao modelo" (não-determinística, mockada nos testes unitários). O resultado é um sistema onde bugs de lógica são capturados pelos testes, e bugs de integração com modelos são capturados pelos testes E2E.

---

## Status do projeto

| Camada | Status | Testes |
|---|---|---|
| Schemas (Pydantic) | ✅ Completo | 37 testes unitários |
| Logging (structlog) | ✅ Completo | 22 testes unitários |
| Pipeline (orquestrador) | ✅ Completo | 16 testes unitários |
| Reasoning (Llama) | ✅ Completo | 16 testes unitários |
| RAG (ChromaDB + nomic) | ✅ Completo | 34 testes unitários |
| Vision (YOLO + CLIP) | ✅ Completo | 24 testes unitários |
| Integração | ✅ Completo | 10 testes de integração |
| E2E | ✅ Estrutura completa | 2 testes E2E (requerem Ollama + fotos reais) |

**Total: 170 testes, 100% de cobertura unitária.**
