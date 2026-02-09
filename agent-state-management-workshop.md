---
marp: true
theme: default
paginate: true
width: 1920px
height: 1080px
backgroundColor: #ffffff
color: #1e1e2e
style: |
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding: 25px 40px 40px 40px;
    display: flex;
    flex-direction: column;
    justify-content: start;
    align-content: flex-start;
  }
  section::before {
    display: none !important;
  }
  h1 {
    color: #6b21a8;
    border-bottom: 3px solid #6b21a8;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 15px;
  }
  h1:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }
  h2 { color: #7c3aed; }
  code { background: #2d2b55; color: #ffffff; }
  pre { background: #1e1e2e !important; font-size: 0.75em; }
  pre code { background: #1e1e2e; color: #ffffff; }
  table { font-size: 0.8em; }
  th { background: #6b21a8; color: #fff; }
  blockquote { border-left: 4px solid #6b21a8; padding-left: 16px; font-style: italic; color: #555; }
  p, li { font-size: 0.9em; margin: 0.3em 0; }
---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong, a { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Agent AI State Management
## A 30-Minute Workshop

**Damilare Akinlaja**
<img src="https://cdn.simpleicons.org/github/white" width="20" height="20" style="vertical-align: middle;" /> [github.com/darmie](https://github.com/darmie) | <img src="https://cdn.simpleicons.org/x/white" width="18" height="18" style="vertical-align: middle;" /> [x.com/fourEyedWiz](https://x.com/fourEyedWiz)

---

# Workshop Agenda

| Block | Topic | Time |
|-------|-------|------|
| 1 | The State Problem in Agent AI | 5 min |
| 2 | Context Management Architectures | 8 min |
| 3 | Memory Optimization Strategies | 7 min |
| 4 | Best Practices | 5 min |
| 5 | Demo Walkthrough & Next Steps | 5 min |

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong, a { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Block 1
## The State Problem in Agent AI

---

# Why State Management Matters

LLMs are **stateless** — every call starts from zero.

Agents need to **remember**, **prioritize**, and **forget**.

```
Stateless LLM Call:
  User → Prompt → LLM → Response    (no memory)

Stateful Agent:
  User → State Manager → Prompt + Context → LLM → Response
              ↕
         Memory Store
```

Without state management, agents repeat mistakes, lose context mid-task, and can't learn from interactions.

---

# What Is Agent State?

| State Type | What It Holds | Example |
|-----------|---------------|---------|
| **Conversational** | Current dialog turns | Chat history |
| **Task** | Progress, goals, tool results | Multi-step plan execution |
| **Episodic** | Past interaction summaries | "Last week user asked about X" |
| **Semantic** | Domain knowledge, facts | User preferences, docs |
| **Procedural** | Learned workflows | "Deploy = build → test → push" |

The challenge: fitting the **right** state into a **finite** context window.

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong, a { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Block 2
## Context Management Architectures

---

# Architecture 1: Sliding Window

The simplest approach — keep the last N turns.

```
Window size = 5

Turn 1: ██░░░░░░░░  (in window)
Turn 2: ██░░░░░░░░  (in window)
Turn 3: ██░░░░░░░░  (in window)
Turn 4: ██░░░░░░░░  (in window)
Turn 5: ██░░░░░░░░  (in window)
Turn 6: ██░░░░░░░░  → Turn 1 dropped
```

**Pros**: Simple, predictable token usage
**Cons**: Loses early context, no prioritization

---

# Architecture 2: Summarization + RAG

Compress old context, retrieve when relevant.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Live Window │     │  Summaries   │     │  Vector DB   │
│  (last 3-5   │     │  (compressed │     │  (searchable │
│   turns)     │     │   history)   │     │   archive)   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       └────────────────────┴────────────────────┘
                            │
                    Context Assembly
```

**Pros**: Longer effective memory, token-efficient
**Cons**: Lossy compression, retrieval latency

---

# Architecture 3: Tiered Memory (Context Graph)

Organize state by **access frequency and relevance**.

```
┌─────────────────────────────────────────────────────────┐
│  HOT    │ In-memory, <10ms  │ Active context             │ ← Frequently used
├─────────┼───────────────────┼────────────────────────────┤
│  WARM   │ Cache, <50ms      │ Recent/related             │ ← Occasionally used
├─────────┼───────────────────┼────────────────────────────┤
│  COLD   │ DB/Vector, <200ms │ Full archive               │ ← Rarely accessed
│         │ Knowledge Graph   │ Ontology + entity relations │
└─────────────────────────────────────────────────────────┘

        ▲ promote (temp ≥ 0.7)
        ▼ demote  (temp < 0.5)
```

Cold tier backs onto a **Knowledge Graph / Ontology** (e.g. Neo4j, OpenLink Virtuoso) — encoding entity relationships and domain structure that enriches retrieval beyond vector similarity.

**Pros**: Self-organizing, semantically rich with graph-based reasoning
**Cons**: More infrastructure, tuning required

---

# Comparing Architectures

| | Sliding Window | Summarize + RAG | Tiered Memory |
|--|----------------|-----------------|---------------|
| **Complexity** | Low | Medium | High |
| **Token Efficiency** | Poor | Good | Best |
| **Long-term Recall** | None | Good | Excellent |
| **Adaptation** | None | None | Self-organizing |
| **Best For** | Chatbots | Assistants | Autonomous agents |

> Choose based on your agent's autonomy level. More autonomous = more state management needed.

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong, a { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Block 3
## Memory Optimization Strategies

---

# Strategy 1: Temperature Scoring

Score each memory to decide what stays accessible:

```python
temperature = (
    0.3 * recency_score(last_accessed) +
    0.25 * relevance_score(query_embedding, memory_embedding) +
    0.2 * frequency_score(access_count) +
    0.15 * entity_overlap(current_entities, memory_entities) +
    0.1 * agent_match(current_agent, memory_agent)
)
```

High temperature → keep hot. Low temperature → archive.

This turns memory management into a **continuous optimization** problem rather than a binary keep/discard decision.

---

# Strategy 2: Context Window Budgeting

Allocate your token budget like a resource:

```
Total Context Window: 128K tokens
┌─────────────────────────────────────────┐
│ System Prompt          │  2K  (fixed)   │
│ Tool Definitions       │  3K  (fixed)   │
│ Retrieved Memories     │ 15K  (dynamic) │
│ Conversation History   │  8K  (sliding) │
│ Current Task State     │  4K  (dynamic) │
│ ─────────────────────────────────────── │
│ Available for Response │ 96K            │
└─────────────────────────────────────────┘
```

**Rule of thumb**: Keep injected context under 25% of the window. Leave room for reasoning.

---

# Strategy 3: Intelligent Forgetting

Not all memories deserve persistence.

**Time-based decay**:
```python
relevance = base_score * exp(-decay_rate * hours_since_access)
```

**Merge duplicates**: Consolidate similar memories instead of storing variants.

**Threshold pruning**: Periodically evict memories below a minimum score.

```
Before: 10,000 memories → 45K tokens to search
After:   2,400 memories → 12K tokens, same quality
```

> A good memory system forgets strategically — just like humans.

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong, a { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Block 4
## Best Practices

---

# Do's and Don'ts

| Do | Don't |
|----|-------|
| Separate **storage** from **retrieval** logic | Dump entire history into the prompt |
| Set **token budgets** per context section | Let context grow unbounded |
| Use **embeddings** for semantic retrieval | Rely only on keyword matching |
| **Summarize** older interactions | Keep every raw message forever |
| **Version** your memory schemas | Change formats without migration |
| **Monitor** context utilization metrics | Assume your context strategy is working |

---

# Production Checklist

**Observability**
- Track token utilization per request
- Log cache hit rates (hot/warm/cold)
- Alert on context window overflow

**Resilience**
- Graceful degradation: if memory DB is down, agent still works (just forgetful)
- TTL on all cached state — stale data is worse than no data

**Security**
- Scope memories per user/session — no cross-contamination
- Sanitize retrieved content before injection into prompts
- Audit trail for what context influenced which response

---

# Vendor Solutions Landscape

| Layer | Vendor | What It Provides |
|-------|--------|-----------------|
| **Memory Management** | Mem0, Zep, LangMem | Automatic memory extraction, scoring, retrieval |
| **Vector Store** | Qdrant, Pinecone, Weaviate, Chroma | Embedding storage + similarity search |
| **Knowledge Graph** | OpenLink Virtuoso, Neo4j, Amazon Neptune | Ontology, entity relations, SPARQL/Cypher queries |
| **Document Store** | MongoDB Atlas, Elasticsearch | Hybrid search, full-text + vector |
| **Orchestration** | LangChain, LlamaIndex, CrewAI | Agent frameworks with memory plugins |
| **Cache** | Redis, Momento, DragonflyDB | Hot/warm tier with TTL and pub/sub |

> No single vendor covers the full stack. Production systems **combine** layers.

---

# Production Architecture: Combining Vendors

```
┌───────────────────────────────────────────────────────────────────┐
│                     AGENT ORCHESTRATOR                             │
│              (LangChain / LlamaIndex / Custom)                    │
│                                                                    │
│  ┌─────────────────┐  promote ≥0.7  ┌─────────────────┐          │
│  │    HOT TIER     │◄──────────────│    WARM TIER    │          │
│  │  Redis/Momento  │──────────────▶│   Mem0 / Zep    │          │
│  │   <10ms local   │  demote <0.5  │  scored memory  │          │
│  └─────────────────┘               └────────┬────────┘          │
│                                       demote │ promote           │
│  ┌────────────────────────────────────────────▼────────────────┐ │
│  │                       COLD TIER                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │   Qdrant /   │  │   OpenLink   │  │   MongoDB /     │  │ │
│  │  │   Pinecone   │  │   Virtuoso   │  │  Elasticsearch  │  │ │
│  │  │  (vectors)   │  │   (graph +   │  │  (documents)    │  │ │
│  │  │              │  │   ontology)  │  │                 │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      LLM Provider     │
                    │ (OpenAI / Anthropic / │
                    │   Local / Azure)      │
                    └───────────────────────┘
```

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong, a { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Block 5
## Demo Time

---

# Key Takeaways

1. **State is what makes an agent an agent** — without it, you have a chatbot

2. **Match architecture to autonomy** — sliding window for simple bots, tiered memory for autonomous agents

3. **Budget your context window** — treat tokens as a scarce resource

4. **Forget strategically** — pruning and decay are features, not bugs

5. **Observe everything** — you can't optimize what you don't measure

---

# Reading List

| Topic | Resource |
|-------|----------|
| **Context Engineering** | [ACE: Agentic Context Engineering](https://arxiv.org/abs/2510.04618) — Stanford, 2025 |
| **Memory for Agents** | [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) — UC Berkeley |
| **RAG Foundations** | [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401) — Meta AI |
| **Knowledge Graphs + LLMs** | [Unifying LLMs and Knowledge Graphs](https://arxiv.org/abs/2306.08302) — Survey |
| **Agent Architectures** | [A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.11432) — Renmin University |
| **Token Optimization** | [Lost in the Middle](https://arxiv.org/abs/2307.03172) — Stanford — positional bias in long contexts |
| **Production Patterns** | [Building LLM-Powered Applications](https://www.oreilly.com/library/view/building-llm-powered/9781835462317/) — O'Reilly |

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong, a { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Thank You

**Damilare Akinlaja**
<img src="https://cdn.simpleicons.org/github/white" width="20" height="20" style="vertical-align: middle;" /> [github.com/darmie](https://github.com/darmie) | <img src="https://cdn.simpleicons.org/x/white" width="18" height="18" style="vertical-align: middle;" /> [x.com/fourEyedWiz](https://x.com/fourEyedWiz)

> "The best agent isn't the one that knows the most — it's the one that remembers the right things at the right time."
