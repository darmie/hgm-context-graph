---
marp: true
theme: default
paginate: true
size: 4:3
backgroundColor: #ffffff
color: #2d1b4e
style: |
  section {
    background-color: #ffffff;
    color: #2d1b4e;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 24px;
    padding: 25px 40px;
  }
  h1, h2, h3 {
    color: #6b21a8;
    margin-top: 0.3em;
    margin-bottom: 0.3em;
  }
  h1 {
    font-size: 1.6em;
    border-bottom: 3px solid #6b21a8;
    padding-bottom: 0.15em;
  }
  h2 {
    font-size: 1.3em;
  }
  pre {
    font-size: 0.75em;
    line-height: 1.2;
    margin: 0.3em 0;
  }
  code {
    background-color: #f3e8ff;
    color: #581c87;
    padding: 1px 4px;
    border-radius: 3px;
  }
  pre {
    background-color: #faf5ff;
    border-left: 3px solid #6b21a8;
    padding: 0.5em 0.8em;
  }
  pre code {
    background-color: transparent;
  }
  table {
    font-size: 0.8em;
    margin: 0.3em 0;
  }
  th {
    background-color: #6b21a8;
    color: white;
    padding: 4px 8px;
  }
  td {
    border-color: #e9d5ff;
    padding: 3px 8px;
  }
  blockquote {
    border-left: 3px solid #a855f7;
    background-color: #faf5ff;
    padding: 0.3em 0.8em;
    font-style: italic;
    margin: 0.3em 0;
  }
  p {
    margin: 0.4em 0;
  }
  ul, ol {
    margin: 0.3em 0;
    padding-left: 1.2em;
  }
  li {
    margin: 0.15em 0;
  }
  .purple-bg {
    background-color: #6b21a8;
    color: white;
  }
  .purple-bg h1, .purple-bg h2, .purple-bg h3, .purple-bg p, .purple-bg strong, .purple-bg p strong {
    color: white !important;
    border-bottom-color: white;
  }
  .purple-bg * {
    color: white !important;
  }
  a {
    color: #7c3aed;
  }
  strong {
    color: #6b21a8;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8em;
  }
---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Hierarchical Context Graph Systems

## Building Cognitive Memory for AI Agents

**Context Engineering | Agent State | Agentic RAG | Memory Systems**

---

# Workshop Agenda

1. **Context Engineering Landscape** - Where HGM fits
2. **Memory Types** - Cognitive foundations
3. **Agent State Management** - Focus & episodes
4. **Temperature Scoring** - Multi-factor relevance
5. **Three-Tier Memory** - Hot/Warm/Cold architecture
6. **Agentic RAG** - Active recall with promotion
7. **Pattern Graph** - Learned response strategies
8. **LLM Orchestration** - Intelligent decision-making
9. **Hands-on Exercises**

---

# What is Context Engineering?

> "Context engineering is the new prompt engineering"
> — Andrej Karpathy

**The discipline of managing what information flows into an LLM's context window.**

| Challenge | Impact |
|-----------|--------|
| Limited context (4K-200K tokens) | Can't include everything |
| Token cost | More tokens = more $ |
| Relevance matters | Irrelevant context = poor answers |
| Static retrieval fails | Same docs regardless of context |

---

# The Evolution of Context Management

```
Prompt Engineering → RAG v1 → RAG v2 → Agentic RAG → Context Engineering
     (static)        (retrieve)  (chunk)   (active)      (holistic)
                                              ↑
                                         HGM operates here
```

**Key insight**: Traditional RAG is **passive**. HGM is **active**—it reorganizes memories based on what's relevant NOW.

---

# Research Foundations

| Paper | Key Contribution |
|-------|------------------|
| [**Generative Agents**](https://arxiv.org/abs/2304.03442) | Memory streams, recency × importance × relevance |
| [**MemGPT**](https://arxiv.org/abs/2310.08560) | Virtual context management, paging |
| [**Reflexion**](https://arxiv.org/abs/2303.11366) | Learning from failures |
| [**RAPTOR**](https://arxiv.org/abs/2401.18059) | Recursive summarization trees |
| [**ACE**](https://arxiv.org/abs/2510.04618) | Evolving context playbooks, self-improving agents |
| **ACT-R / SOAR** | Cognitive architectures |

HGM synthesizes these ideas into a unified system.

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Section 1
## Memory Types

---

# Four Memory Types (Cognitive Science)

| Type | What It Stores | Example |
|------|----------------|---------|
| **SEMANTIC** | Facts, concepts, knowledge | "Python is dynamically typed" |
| **EPISODIC** | Events, experiences | "We discussed auth yesterday" |
| **PROCEDURAL** | Instructions, how-to | "To deploy: docker compose up" |
| **EMOTIONAL** | Preferences, sentiment | "User prefers concise answers" |

Based on **Tulving's memory taxonomy** from cognitive psychology.

---

# Memory Types in Code

```python
class MemoryType(str, Enum):
    SEMANTIC = "semantic"      # Facts, concepts
    EPISODIC = "episodic"      # Events, experiences
    PROCEDURAL = "procedural"  # Instructions, how-to
    EMOTIONAL = "emotional"    # Preferences, sentiment

@dataclass
class Memory:
    id: str
    content: str
    embedding: np.ndarray
    memory_type: MemoryType
    temperature: float  # 0.0 to 1.0
```

---

# Why Memory Types Matter

**Without classification:**
- Facts and preferences treated identically
- Instructions buried in noise
- Can't prioritize by query intent

**With classification:**
- "How to deploy?" → Prioritize PROCEDURAL
- "What happened yesterday?" → Prioritize EPISODIC
- "User preferences?" → Prioritize EMOTIONAL

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Section 2
## Agent State Management

---

# Agent Context Components

```python
@dataclass
class AgentContext:
    agent_id: str
    agent_role: str  # "researcher", "coder", etc.

    # Focus tracking
    focus_embedding: np.ndarray  # What agent is "thinking about"
    focus_entities: set[str]      # Key concepts
    focus_hierarchy_path: str     # "tech/python/async"

    # Session tracking
    turn_count: int
    current_episode: Episode
```

**Focus** determines which memories are relevant RIGHT NOW.

---

# Episode Management

**Episodes** = Coherent conversation segments grouped by topic

```
Session Start
    │
    ├── Episode 1: "Debugging auth" ────────────────┐
    │   └── Memories about auth, JWT, login         │
    │                                               │ Topic drift
    ├── Episode 2: "Deployment planning" ──────────┤ detected!
    │   └── Memories about Docker, CI/CD           │
    │                                               │
    └── Episode 3: "Code review" ──────────────────┘
```

Detecting topic changes helps archive context and reset focus.

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Section 3
## Temperature Scoring

---

# What is Temperature?

A **unified metric (0.0 to 1.0)** representing how "hot" a memory is RIGHT NOW.

| Zone | Range | Meaning |
|------|-------|---------|
| 🔥 BLAZING | >0.85 | Critical for current task |
| 🌡️ HOT | 0.70-0.85 | Working memory |
| ☀️ WARM | 0.50-0.70 | Session cache |
| 🌤️ COOLING | 0.30-0.50 | Fading relevance |
| ❄️ COLD | <0.30 | Long-term archive |

---

# Temperature Formula

```
temperature = 0.30 × recency
            + 0.15 × frequency
            + 0.35 × relevance
            + 0.15 × entity_overlap
            + 0.05 × agent_match
```

| Factor | Weight | Calculation |
|--------|--------|-------------|
| Recency | 30% | `0.5^(hours_ago / 24)` |
| Frequency | 15% | `access_count / max_count` |
| Relevance | 35% | Cosine similarity to focus |
| Entity overlap | 15% | Keyword intersection |
| Agent match | 5% | Same agent bonus |

---

# Why Multi-Factor Scoring?

**Single-factor approaches fail:**

| Approach | Problem |
|----------|---------|
| Most recent only | Misses relevant older knowledge |
| Most accessed only | Ignores current context |
| Similarity only | Misses entity connections |

**Temperature combines all signals:**
- Recent + relevant + accessed = 🔥 BLAZING
- Old but highly relevant = Still promoted
- Recent but irrelevant = Cools quickly

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Section 4
## Three-Tier Memory Architecture

---

# The Three Tiers

```
┌─────────────────────────┐
│       HOT TIER          │  ← <1ms (Rust in-memory)
│     Working Memory      │     Token-budget LRU eviction
└───────────┬─────────────┘
    ↑ promote │ demote ↓
┌───────────┴─────────────┐
│       WARM TIER         │  ← <50ms (Redis)
│     Session Cache       │     Recent interactions
└───────────┬─────────────┘
    ↑ promote │ demote ↓
┌───────────┴─────────────┐
│       COLD TIER         │  ← <200ms (PostgreSQL)
│    Long-term Storage    │     Full knowledge base
└─────────────────────────┘
```

---

# Hot Tier Deep Dive

**Purpose**: Sub-millisecond access to most critical memories

**Key features**:
- **Token-budget eviction** (not count-based)
- **LRU policy** for cache management
- **Vectorized similarity** (SIMD acceleration)
- **Entity boost** for keyword matches

```python
class HotTier:
    max_tokens: int = 8000  # Context budget

    def scan(self, query_embedding, focus_entities, limit=10):
        # Batch cosine similarity (vectorized)
        similarities = (embeddings / norms) @ query_vector
```

---

# Tier Movement Rules

| Current | Temperature | Action |
|---------|-------------|--------|
| Cold | ≥ 0.70 | **Promote to Hot** |
| Cold | ≥ 0.50 | Promote to Warm |
| Warm | ≥ 0.70 | **Promote to Hot** |
| Warm | < 0.50 | Demote to Cold |
| Hot | < 0.50 | **Demote to Warm** |

**Automatic organization** based on access patterns!

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Section 5
## Agentic RAG

---

# Traditional RAG vs Agentic RAG

| Traditional RAG | Agentic RAG (HGM) |
|-----------------|-------------------|
| Search once, return results | Search + reorganize |
| All memories equally accessible | Hot = fast, Cold = slower |
| No learning from patterns | Promotes frequently used |
| Cold start every query | Warmed-up context |
| Static indexes | Self-organizing tiers |

---

# Active Recall Flow

```python
def recall(query, focus_entities):
    # 1. Search all tiers in parallel
    hot_results = hot_tier.scan(query_emb)
    warm_results = warm_tier.search(query_emb)
    cold_results = cold_tier.search(query_emb)

    # 2. Score with temperature formula
    for memory in all_results:
        memory.temperature = compute_temperature(memory, focus)

    # 3. Promote/demote based on new temperatures
    for memory in all_results:
        if memory.temperature >= 0.70:
            promote_to_hot(memory)
        elif memory.temperature < 0.50:
            demote_from_hot(memory)

    # 4. Return sorted by temperature
    return sorted(all_results, key=lambda m: m.temperature)
```

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Section 6
## Pattern Graph System

---

# Pattern Graph Structure

```
         ┌────────────────┐
         │   ENTITIES     │  Keywords from queries
         │ deploy, docker │
         └───────┬────────┘
                 │ edges (strength-weighted)
                 ▼
         ┌────────────────┐
         │   PATTERNS     │  Trigger → Strategy pairs
         │ "How to deploy │
         │  → Use docker  │
         │    compose..." │
         └────────────────┘
```

**Find patterns via keyword matching, not just embeddings!**

---

# Pattern Matching Algorithm

```python
def find_patterns(keywords):
    results = {}

    for keyword in keywords:
        if keyword in entity_to_patterns:
            for (pattern_id, strength) in entity_to_patterns[keyword]:
                score = strength × pattern.effectiveness
                results[pattern_id] = max(existing, score)

    return sorted(results, key=score, reverse=True)
```

**Complexity**: O(k × p) where k=keywords, p=patterns per keyword

---

# Pattern Relevance Scoring

```
relevance = 0.40 × semantic_similarity
          + 0.15 × keyword_overlap
          + 0.20 × topic_overlap
          + 0.10 × structure_match
          + 0.15 × keyword_boost
```

| Signal | What It Captures |
|--------|------------------|
| Semantic | Embedding cosine similarity |
| Keyword | Jaccard of extracted keywords |
| Topic | Long words (≥5 chars) overlap |
| Structure | Question pattern match ("how to" vs "what is") |

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Section 7
## Mode Selection

---

# Four Response Modes

| Mode | When | Cost | Latency |
|------|------|------|---------|
| **FAST** | Have memories | Low | <500ms |
| **AGENT** | Need exploration | Medium | 1-10s |
| **PATTERN_DIRECT** | High-confidence match | Very Low | <100ms |
| **WORKFLOW** | Complex multi-step | High | 10s+ |

---

# Mode Selection Algorithm

```python
def select_mode(query, memories, patterns):
    # Priority 1: Complex task keywords
    if "analyze" in query or "workflow" in query:
        return WORKFLOW

    # Priority 2: High-confidence pattern
    if patterns and best_pattern.relevance >= 0.7:
        return PATTERN_DIRECT  # Skip LLM!

    # Priority 3: Have memories
    if len(memories) >= 1:
        return FAST

    # Default: Need exploration
    return AGENT
```

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Section 8
## LLM-Driven Orchestration

---

# Why LLM-Driven Decisions?

**Rule-based limitations:**
- Fixed thresholds (temp > 0.7 → promote)
- Keyword matching only
- Binary decisions
- Can't handle novel situations

**LLM-driven advantages:**
- Contextual reasoning ("this is relevant because...")
- Semantic understanding
- Nuanced confidence
- Generalizes to new contexts

---

# Orchestrator Flow

```
Query → Initial Retrieval → LLM Analysis → Adjusted Decisions
              │                  │
              │   ┌──────────────┴──────────────┐
              │   │  • Evaluate memory relevance │
              │   │  • Suggest temp adjustments  │
              │   │  • Extract keywords          │
              │   │  • Recommend mode            │
              │   │  • Provide reasoning         │
              │   └──────────────────────────────┘
              │                  │
              └──────────────────┴─────→ Response
```

---

# Temperature Adjustment Formula

```python
# Combines rule-based (60%) with LLM judgment (40%)
adjusted_temp = (
    0.6 * rule_based_temperature +
    0.4 * llm_relevance_score +
    llm_suggested_adjustment
)
```

**LLM provides:**
- Relevance score (0.0-1.0)
- Reasoning ("Directly addresses deployment question")
- Adjustment suggestion (-0.3 to +0.3)

---

# LLM Provider Abstraction

```python
class BaseLLM(ABC):
    def complete(self, prompt: str, system: str = "") -> str: ...

# Available implementations
llm = get_llm("auto")  # Tries OpenAI → Anthropic → Mock

# Or specify explicitly
llm = get_llm("openai")    # GPT-4o-mini
llm = get_llm("anthropic") # Claude 3.5 Haiku
llm = get_llm("mock")      # No API needed (for workshop)
```

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# The Complete Pipeline

---

# End-to-End Flow

```
1. KEYWORD EXTRACTION
   "How do I deploy Python?" → ["deploy", "python"]

2. PATTERN GRAPH LOOKUP
   Keywords → Pattern matches with relevance

3. MEMORY RECALL (Agentic RAG)
   Query embedding → Tier search → Promotions

4. PATTERN SCORING
   Multi-factor weighted scoring

5. LLM ORCHESTRATION
   Analyze + adjust + select mode

6. MODE SELECTION
   FAST / AGENT / PATTERN_DIRECT / WORKFLOW

7. RESPONSE GENERATION
```

---

# HGM vs Traditional RAG

| Aspect | Traditional RAG | HGM |
|--------|-----------------|-----|
| Retrieval | Static vector search | Dynamic three-tier |
| Organization | Fixed indexes | Self-organizing |
| Learning | None | Pattern graph |
| Routing | Always LLM | Mode selection |
| Context | Global | Per-agent focus |
| Decisions | Rule-based | LLM-enhanced |

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Agent Chat Continuity

---

# How Hot Memory + Patterns Enable Continuity

**The Problem**: Each LLM call is stateless. How do agents maintain coherent conversations?

**The Solution**: Store messages with **labels** and reconstruct context from **hot memories + patterns**.

### Memory Labels

| Label | What It Stores | Purpose |
|-------|----------------|---------|
| `USER_QUERY` | User's exact message | Track what user asked |
| `AGENT_THOUGHT` | Agent's reasoning/response | Track decisions made |
| `PATTERN` | Learned strategy reference | Apply known solutions |

---

# Context Assembly with Labels

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT CONTEXT WINDOW                      │
├─────────────────────────────────────────────────────────────┤
│  [SYSTEM PROMPT]                                             │
│  [HOT MEMORIES]                                              │
│     • [USER_QUERY] "Deploy my Python app"                   │
│     • [AGENT_THOUGHT] "User needs containerization"         │
│     • [USER_QUERY] "Use Kubernetes"                         │
│     • [AGENT_THOUGHT] "Switching to K8s approach"           │
│  [MATCHED PATTERNS]                                          │
│     • [PATTERN] kubernetes_deployment → k8s manifest        │
│  [CURRENT EPISODE] Topic: deployment, Entities: k8s, python │
│  [USER MESSAGE] Current query                                │
└─────────────────────────────────────────────────────────────┘
```

---

# Turn-Based Continuity Flow

```
TURN 1: User → "Help me deploy my app"
   ┌─ STORE ─────────────────────────────────────────────┐
   │  [USER_QUERY] "Help me deploy my app"               │
   │  [AGENT_THOUGHT] "User needs deployment guidance"   │
   │  [PATTERN] deployment → docker_strategy             │
   └─────────────────────────────────────────────────────┘
   Agent → "I'll help you deploy using Docker..."

TURN 2: User → "Use Kubernetes instead"
   ┌─ RECALL from HOT ───────────────────────────────────┐
   │  [USER_QUERY] "Help me deploy my app"  ← Turn 1     │
   │  [AGENT_THOUGHT] "User needs deployment guidance"   │
   └─────────────────────────────────────────────────────┘
   ┌─ STORE ─────────────────────────────────────────────┐
   │  [USER_QUERY] "Use Kubernetes instead"              │
   │  [AGENT_THOUGHT] "Switching to K8s approach"        │
   │  [PATTERN] kubernetes → k8s_manifest_strategy       │
   └─────────────────────────────────────────────────────┘
   Agent → "For Kubernetes, create a deployment.yaml..."
```

---

# Successive Messages Without Breaking Context

```
TURN 3: User → "What about secrets?"  (no explicit K8s mention)
   ┌─ RECALL from HOT ───────────────────────────────────┐
   │  [USER_QUERY] "Use Kubernetes instead"  ← Turn 2    │
   │  [AGENT_THOUGHT] "Switching to K8s approach"        │
   │  [PATTERN] kubernetes → active context              │
   └─────────────────────────────────────────────────────┘
   Agent KNOWS: "secrets" refers to Kubernetes secrets!
   Agent → "For K8s secrets, use kubectl create secret..."

TURN 4: User → "And ConfigMaps?"  (builds on previous)
   ┌─ RECALL from HOT ───────────────────────────────────┐
   │  All previous context + secrets discussion          │
   └─────────────────────────────────────────────────────┘
   Agent → "ConfigMaps work similarly to secrets..."
```

**Key**: Labels preserve context type, enabling intelligent recall.

---

# The Continuity Formula with Labels

**Context Assembly** for each turn:

```python
class MemoryLabel(str, Enum):
    USER_QUERY = "user_query"      # User's message
    AGENT_THOUGHT = "agent_thought" # Agent's reasoning
    PATTERN = "pattern"             # Matched strategy

def assemble_context(query: str) -> list[str]:
    context = []

    # 1. Hot memories with labels (sorted by temperature)
    hot_memories = hot_tier.scan(query_embedding, limit=10)
    for mem in hot_memories:
        context.append(f"[{mem.label}] {mem.content}")

    # 2. Matched patterns
    patterns = pattern_graph.find_patterns(keywords)
    for p in patterns:
        if p.relevance > 0.5:
            context.append(f"[PATTERN] {p.trigger}: {p.strategy}")

    return context
```

---

# Why Labels Enable Continuity

| Label | Role | Example |
|-------|------|---------|
| `USER_QUERY` | Track what user asked | "Deploy my app" → "Use K8s" → "secrets?" |
| `AGENT_THOUGHT` | Track agent decisions | "User needs containerization" |
| `PATTERN` | Apply learned strategies | "kubernetes → use manifests" |

**How successive messages work:**
1. Each turn stores `[USER_QUERY]` + `[AGENT_THOUGHT]`
2. Next turn recalls previous labels from hot tier
3. Agent sees conversation history with context types
4. Ambiguous messages ("secrets?") resolved via recent `[PATTERN]`

**Result**: User can send short follow-ups without repeating context.

---

# Continuity Across Sessions

**Within session**: Hot tier maintains working memory

**Across sessions**: Patterns + Cold tier provide persistence

```
Session 1: Learn user prefers kubectl over helm
           → Store in COLD, create pattern

Session 2: User asks about deployment
           → Pattern match: "User prefers kubectl"
           → Agent remembers preference!
```

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Hands-On Exercises

---

# Exercise 1: Temperature Tuning

**Goal**: Understand how weights affect scoring

```python
# Try different weight configurations
recency_heavy = TemperatureConfig(weight_recency=0.50, ...)
relevance_heavy = TemperatureConfig(weight_relevance=0.55, ...)

# Compare results on same memory
```

**Question**: When would you prefer recency over relevance?

---

# Exercise 2: Pattern Creation

**Goal**: Add domain-specific patterns

```python
graph.add_pattern(
    "pat_kubernetes",
    "How to deploy to Kubernetes?",
    "Use kubectl apply or helm install",
    effectiveness=0.85,
)

# Link keywords
for kw in ["kubernetes", "k8s", "kubectl", "helm"]:
    graph.link_entity_to_pattern(kw, "pat_kubernetes")
```

**Test**: Query with "k8s deployment" - does your pattern match?

---

# Exercise 3: Mode Selector Enhancement

**Goal**: Add a CLARIFICATION mode

```python
class EnhancedModeSelector(ModeSelector):
    def _is_ambiguous(self, query: str) -> bool:
        # Very short queries
        if len(query.split()) <= 2:
            return True
        # No topic keywords
        if len(extract_keywords(query)) == 0:
            return True
        return False
```

**When should CLARIFICATION trigger?**

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Key Takeaways

---

# Core Concepts Recap

| Concept | Key Algorithm |
|---------|---------------|
| Memory Types | Cognitive classification |
| Agent State | Focus tracking + episodes |
| Temperature | 5-factor weighted scoring |
| Hot Tier | Token-budget LRU |
| Agentic RAG | Parallel search + promotion |
| Pattern Graph | Entity → Pattern traversal |
| Pattern Scoring | Multi-signal combination |
| Mode Selection | Priority-based routing |
| LLM Orchestration | Reasoning-enhanced decisions |

---

# The Key Formulas

**Temperature:**
```
temp = 0.30×recency + 0.15×frequency + 0.35×relevance
     + 0.15×entity_overlap + 0.05×agent_match
```

**Pattern Relevance:**
```
rel = 0.40×semantic + 0.15×keyword + 0.20×topic
    + 0.10×structure + 0.15×boost
```

**LLM Temperature Adjustment:**
```
adjusted = 0.6×rule_based + 0.4×llm_relevance + adjustment
```

---

# Resources

**Workshop Materials:**
- Notebook: `workshops/hgm-core-concepts.ipynb`
- Slides: `workshops/hgm-core-concepts-slides.md`

**Research Papers:**
- [Generative Agents](https://arxiv.org/abs/2304.03442)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [RAPTOR](https://arxiv.org/abs/2401.18059)

**Documentation:**
- [LangChain](https://docs.langchain.com)
- [LlamaIndex](https://docs.llamaindex.ai)

---

<!-- _class: purple-bg -->
<!-- _backgroundColor: #6b21a8 -->
<!-- _color: #ffffff -->
<style scoped>
h1, h2, p, strong { color: #ffffff !important; }
h1 { border-bottom-color: #ffffff; }
</style>

# Thank You!

## Questions?

**HGM: Cognitive infrastructure for AI that learns.**

---

# Appendix: Quick Reference

```python
# Create memory
memory = create_memory(content, MemoryType.SEMANTIC, "path/to/topic")

# Score temperature
temp = scorer.compute(memory, focus_embedding, focus_entities)

# Recall with promotion
memories, promotions = tiers.recall(query_emb, entities)

# Find patterns
patterns = graph.find_patterns(keywords)

# LLM orchestration
decision = orchestrator.orchestrate(query, memories, patterns)
```
