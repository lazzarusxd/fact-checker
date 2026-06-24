# Verificador Factual Multiagente

Sistema multiagente de verificação factual (fact-checking) com LLM, RAG local e
busca web gratuita. Cada alegação é decomposta, investigada e submetida a um
**ciclo de debate adversarial** antes do veredito final.

## Arquitetura dos agentes

```
Alegação
   │
   ▼
[1] Decompositor   → quebra em sub-alegações atômicas verificáveis
   │
   ▼  (para cada sub-alegação, em laço)
[2] Recuperador    → busca evidências em 3 fontes:
                       • base curada (RAG sobre o corpus fixo)
                       • cache aprendido (evidências de buscas anteriores, dentro do TTL)
                       • web ao vivo (DuckDuckGo)
                     e "aprende" as evidências novas da web para reuso futuro
[3] Verificador    → classifica SUPPORTS / REFUTES / NOT_ENOUGH_INFO, citando a fonte
[4] Crítico        → desafia o veredito; se houver objeção, dispara nova rodada
   │  (repete até convergir ou atingir MAX_DEBATE_ROUNDS)
   ▼
[5] Adjudicador    → veredito final + confiança calibrada
   │
   ▼
Veredito geral (agregado das sub-alegações) + traço completo persistido no SQLite
```

## Tecnologias

- **Python** + **FastAPI** (servidor e API)
- **Groq** via `AsyncOpenAI` (LLM — plano gratuito)
- **Pydantic v2** (contratos dos agentes / saídas estruturadas validadas)
- **SQLite local** via `aiosqlite` (persistência em arquivo — sem servidor, sem Docker)
- **sentence-transformers** + **numpy** (RAG local, offline)
- **ddgs** (busca web DuckDuckGo — gratuita, sem chave)

## Pré-requisitos

1. **Python 3.10+**
2. Uma **chave gratuita da Groq**: https://console.groq.com/keys

(Não precisa instalar banco de dados: o SQLite vem embutido no Python e o arquivo é criado automaticamente.)

## Instalação

```bash
# 1) ambiente virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 2) dependências
pip install -r requirements.txt

# 3) configuração
cp .env.example .env          # (Windows: copy .env.example .env)
# edite o .env: cole sua GROQ_API_KEY (o resto já vem com padrões que funcionam)
```

> Na primeira execução, o `sentence-transformers` baixa o modelo de embeddings
> (~90 MB). É feito uma única vez.

## Execução

```bash
python main.py
```

O servidor sobe em `http://127.0.0.1:8000` e o navegador abre automaticamente.
Digite uma alegação (ou clique num exemplo) e veja o veredito + o traço de cada
rodada de debate.

> O arquivo `fact_checker.db` e as tabelas (`claims`, `sub_claims`,
> `debate_rounds`, `learned_evidence`) são criados automaticamente no primeiro start.

## Trocar o domínio

O RAG indexa tudo que estiver em `data/corpus/` (arquivos `.md` ou `.txt`).
Para mudar de assunto, basta substituir os arquivos da pasta e reiniciar.

## Aprendizado incremental (cache de evidências)

A cada verificação, as evidências novas recuperadas da web são persistidas na
tabela `learned_evidence` (com embedding e timestamp) e passam a ser consultadas
em buscas futuras, junto com o corpus e a web ao vivo. Decisões de projeto:

- **Guarda-se evidência, nunca veredito.** O cache acelera e enriquece a
  recuperação, mas o Verificador sempre raciocina do zero. Isso evita que um erro
  do sistema vire "fato apurado" e se realimente (amplificação de erro).
- **TTL (`LEARNED_TTL_DAYS`).** Evidência aprendida tem validade; passou do prazo,
  é ignorada e tende a ser re-aprendida numa busca ao vivo. Mitiga o problema de
  informação que muda com o tempo.
- **A web é sempre consultada ao vivo**, em paralelo ao cache, então evidência
  fresca pode complementar ou contradizer a antiga — e o Crítico adversarial atua.
- **Proveniência visível na interface:** cada evidência aparece marcada como
  `BASE CURADA`, `CACHE APRENDIDO (há Xd)` ou `WEB AO VIVO`.

> Limitação conhecida: o cache pode conter evidência desatualizada antes de vencer
> o TTL. Por isso ele guarda material de pesquisa (com data e fonte), não a
> conclusão — a decisão final é sempre recalculada sobre a evidência vigente.