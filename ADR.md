# 📋 ADR — Architecture Decision Records
### Compliance AI — Decisões Arquiteturais Registradas

> ADRs documentam **o que foi decidido, por que, e quais alternativas foram rejeitadas**.
> São imutáveis: uma decisão superada gera um novo ADR que depreca o anterior,
> nunca edita o original.

---

## ADR-001 — LLM local via Ollama em vez de API externa

**Status:** Aceito  
**Data:** 2024  
**Contexto:** O sistema precisa de um LLM para raciocínio de conformidade.

**Decisão:** Usar Llama via Ollama (local).

**Motivos:**
- Sem custo de API, sem dependência de internet, sem latência de rede
- Dados sensíveis de imagens de trabalhadores não saem da máquina (LGPD/privacidade)
- O mesmo daemon Ollama serve Llama, LLaVA e nomic-embed-text — uma infra, três modelos

**Alternativas rejeitadas:**
- OpenAI GPT-4: custo por token, dados saem da empresa, dependência de uptime externo
- Claude API: mesmos problemas de OpenAI para este contexto

**Consequências:** O sistema requer que Ollama esteja rodando localmente. Testes E2E dependem desse ambiente. Testes unitários e de integração são completamente independentes via mocks.

---

## ADR-002 — RAG semântico com nomic-embed-text + ChromaDB

**Status:** Aceito  
**Data:** 2024  
**Contexto:** O sistema precisa recuperar regras relevantes de documentos de múltiplas empresas.

**Decisão:** RAG semântico com `nomic-embed-text` (Ollama) para embeddings e ChromaDB para armazenamento vetorial persistente.

**Motivos:**
- `nomic-embed-text` roda no mesmo Ollama — sem servidor adicional
- ChromaDB persiste em disco: indexação acontece uma vez, retrieval é rápido
- Busca semântica recupera regras por significado, não apenas por keyword — essencial para queries geradas automaticamente pelo pipeline
- Uma collection por empresa no ChromaDB evita contaminação cruzada de regras

**Alternativas rejeitadas:**
- RAG por keyword/setor (Estratégia A): simples, mas falha em queries ambíguas e não escala com documentos ricos
- FAISS: mais rápido, mas sem persistência nativa e sem filtragem por metadados
- LangChain: adiciona abstração desnecessária para um RAG custom; preferimos implementação própria para controle e testabilidade

**Consequências:** A indexação dos documentos é uma etapa de setup separada do pipeline de inferência. Testes de componente do RAG usam documentos reais de fixture (os manuais das empresas).

---

## ADR-003 — Extração de atributos visuais via LLaVA (multimodal) em vez de classificadores dedicados

**Status:** Aceito  
**Data:** 2024  
**Contexto:** Após YOLO detectar pessoas e extrair bounding boxes, o sistema precisa identificar atributos de EPI (capacete, colete, etc.).

**Decisão:** Enviar a região recortada da bbox ao LLaVA via Ollama com prompt estruturado que retorna JSON.

**Motivos:**
- LLaVA já está no Ollama — sem modelo adicional para treinar ou baixar
- Zero-shot: funciona sem dataset de treinamento de EPIs
- Output JSON estruturado é parseável diretamente para `PersonAttributes`
- Arquitetura simétrica: Llama para reasoning, LLaVA para vision, nomic para embedding — todos via Ollama

**Alternativas rejeitadas:**
- Classificadores YOLO dedicados por EPI: mais preciso em produção, mas requer dataset rotulado, treinamento, e adiciona complexidade de infra
- CLIP zero-shot: viável, mas adiciona outra dependência fora do Ollama

**Consequências:** A qualidade da extração de atributos depende da qualidade do LLaVA e do prompt. O campo `confidence` no output do LLaVA mapeia para `None` nos atributos quando "low", resultando em `"Indeterminado"` — nunca em falso negativo silencioso.

---

## ADR-004 — Separação do Vision em IPersonDetector + IAttributeExtractor

**Status:** Aceito  
**Data:** 2024  
**Contexto:** O módulo de visão tem duas responsabilidades distintas: detectar pessoas e extrair atributos.

**Decisão:** Separar em duas interfaces (`IPersonDetector` com YOLO, `IAttributeExtractor` com LLaVA), orquestradas pela fachada `IVisionService`.

**Motivos:** SRP — cada interface tem exatamente um motivo para mudar. Trocar YOLO por outro detector não afeta o extrator de atributos. Trocar LLaVA por classificadores dedicados não afeta o detector.

**Consequências:** Testes unitários podem mockar cada interface separadamente. Testes de componente podem testar o detector real sem LLaVA, e vice-versa.

---

## ADR-005 — Separação do RAG em IDocumentParser + IEmbeddingService + IDocumentRetriever

**Status:** Aceito  
**Data:** 2024  
**Contexto:** O RAG semântico tem três etapas com motivações de mudança distintas.

**Decisão:** Três interfaces separadas, compostas pelo `IRagService` (fachada do pipeline).

| Interface | Motivo de mudança isolado |
|---|---|
| `IDocumentParser` | Mudar estratégia de chunking, suporte a novo formato de arquivo |
| `IEmbeddingService` | Trocar modelo de embedding, trocar vector store |
| `IDocumentRetriever` | Mudar lógica de filtragem, formato de Rule |

**Consequências:** Cada interface tem seus próprios testes unitários com mocks. O `IEmbeddingService` tem testes de componente com ChromaDB real em diretório temporário (`tmp_path` do pytest).

---

## ADR-006 — Injeção de dependência manual (sem container/framework)

**Status:** Aceito  
**Data:** 2024  
**Contexto:** O sistema precisa de DIP (Dependency Inversion) para testabilidade.

**Decisão:** Injeção de dependência manual via `__init__` — sem framework (sem `dependency-injector`, sem FastAPI DI, sem `injector`).

**Motivos:**
- O sistema é um pipeline de processamento, não uma aplicação web com muitos endpoints
- DI manual é explícita, legível, e suficiente para este escopo
- Frameworks de DI adicionam curva de aprendizado e abstração sem benefício real aqui
- A composição do pipeline acontece em um único ponto (`pipeline/factory.py`)

**Consequências:** Existe um `factory.py` que monta o pipeline completo com implementações reais. Em testes, o pipeline é montado com mocks diretamente no `__init__`.

---

## ADR-007 — TDD adaptado: modelos de IA nunca chamados em testes unitários

**Status:** Aceito  
**Data:** 2024  
**Contexto:** Sistemas de IA têm componentes não-determinísticos que não podem ser testados de forma tradicional.

**Decisão:** Llama, LLaVA e nomic-embed-text são **sempre mockados em testes unitários**. Apenas testes E2E (marcados com `@pytest.mark.e2e`) chamam modelos reais.

**Regra derivada:** O que é testável unitariamente em módulos de IA:
- Parsing da resposta do modelo (string → schema)
- Lógica de decisão dado um output fixo
- Tratamento de erros (timeout, JSON inválido, resposta vazia)
- Propagação correta do `correlation_id`
- Formato e campos do log estruturado

**O que NÃO é testável unitariamente:** qualidade do output do modelo, latência, comportamento com prompts variados — esses pertencem a testes E2E ou evals separados.

---

## ADR-003 — ~~Extração de atributos visuais via LLaVA~~ ⚠️ DEPRECADO por ADR-003b

**Status:** Deprecado  
**Substituído por:** ADR-003b

---

## ADR-003b — Extração de atributos visuais via CLIP zero-shot (classificadores dedicados)

**Status:** Aceito  
**Depreca:** ADR-003  
**Data:** 2024  
**Contexto:** Após YOLO detectar pessoas e extrair bounding boxes, o sistema precisa identificar atributos de EPI (capacete, colete, etc.).

**Decisão:** Usar `openai/clip-vit-base-patch32` via HuggingFace Transformers, rodando em CPU, para classificação zero-shot de atributos por similaridade de texto-imagem.

**Como funciona:** Para cada atributo, o CLIP compara a bbox recortada contra um par de textos opostos ("a person wearing a hard hat" vs "a person not wearing a hard hat") e retorna um score de similaridade entre 0 e 1.

**Motivos:**
- Sem dependência de parsing de JSON — elimina risco de alucinação e falha de formato
- Score de similaridade contínuo permite zona de incerteza explícita (ADR-009)
- Mais preciso e robusto em produção do que prompt engineering com LLM
- Mais escalável: adicionar novo atributo = adicionar novo par de textos, sem retreinar
- Roda em CPU sem servidor adicional — HuggingFace Transformers como dependência direta
- Thresholds configuráveis por atributo via `CLIPThresholds(BaseSettings)` — calibráveis por empresa/setor sem modificar código

**Alternativas rejeitadas:**
- LLaVA via Ollama (ADR-003): dependência de JSON parsing, risco de alucinação, mais lento em CPU
- Classificadores YOLO dedicados por EPI: requer dataset rotulado de EPIs e treinamento — inviável sem dados

**Consequências:**
- `IAttributeExtractor` recebe bbox recortada e retorna `PersonAttributes` com `Optional[bool]`
- Scores abaixo do threshold de incerteza resultam em `None` — ver ADR-009
- Em testes unitários, `IAttributeExtractor` é sempre mockado; o modelo CLIP nunca é carregado
- Testes de componente podem carregar o modelo real com imagens de fixture

---

## ADR-008 — Score CLIP ambíguo mapeia para `None`, não `false` *(atualizado de LLaVA para CLIP)*

**Status:** Aceito (atualizado — antes se referia a `confidence: "low"` do LLaVA)  
**Data:** 2024  
**Contexto:** O CLIP retorna scores contínuos de similaridade. Um score de 0.51 não deve ser tratado da mesma forma que 0.10 — ambos virariam `False` num threshold binário, mas carregam significados completamente diferentes.

**Decisão:** Atributos cujo score cair na zona de incerteza são armazenados como `None` em `PersonAttributes` (não `false`). O reasoning trata `None` como `"Indeterminado"`.

**Motivos:** Falso negativo com alta confiança (score 0.1) e falso negativo por imagem ambígua (score 0.49) são situações opostas. Tratá-las igual esconde informação relevante para operação e auditoria.

**Consequências:** `PersonAttributes` usa `Optional[bool]` para cada atributo, não `bool`. O reasoning tem lógica explícita para `None`. A `justificativa` inclui o score numérico quando o status for `"Indeterminado"`.

---

## ADR-009 — Zona de incerteza CLIP com thresholds configuráveis por atributo

**Status:** Aceito  
**Data:** 2024  
**Contexto:** Com CLIP produzindo scores contínuos, precisamos de uma regra de decisão que separe presença confirmada, ausência confirmada e incerteza genuína.

**Decisão:** Implementar zona de incerteza (Opção 2) com thresholds configuráveis por atributo via `CLIPThresholds(BaseSettings)`.

```
score >= threshold_positive  →  True   (presença confirmada)
score <= threshold_negative  →  False  (ausência confirmada)
entre os dois                →  None   (zona de incerteza → Indeterminado)
```

**Defaults iniciais:**

| Atributo | threshold_positive | threshold_negative | Justificativa |
|---|---|---|---|
| helmet | 0.65 | 0.35 | Risco alto — zona estreita, mais decisões binárias |
| vest | 0.60 | 0.40 | Visualmente mais fácil de detectar |
| safety_boots | 0.70 | 0.30 | Mais difícil de ver na imagem — zona mais larga |
| gloves | 0.60 | 0.40 | Similar ao vest |

**Por que não threshold único global:** atributos têm características visuais diferentes. Botas de segurança aparecem parcialmente na imagem com frequência — merece zona de incerteza mais larga. Capacete é crítico para segurança — merece threshold mais alto para confirmar presença.

**Extensibilidade (OCP):** os thresholds são lidos de variáveis de ambiente (`CLIP_HELMET_POSITIVE`, etc.). Quando houver dados reais de calibração por empresa, basta setar as variáveis — sem modificar código.

**Alternativas rejeitadas:**
- Threshold binário fixo (Opção 1): esconde incerteza, produz falsos negativos silenciosos
- Thresholds por empresa no banco de dados (Opção 3 completa): prematura sem dados reais de calibração — adicionada como caminho de evolução via `BaseSettings`

---

*Novos ADRs são adicionados ao final. ADRs nunca são editados — apenas deprecados por novos.*
