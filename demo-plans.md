# Demo Plans — Agent AI State Management Workshop

Comprehensive implementation plans for each demo suggested in the workshop.

---

## Demo A: Context Management & Temperature-Based Memory

### Objective

Build an agent that progresses through three context management strategies in a single session:

1. **Sliding Window** — naive approach, observe where it breaks
2. **Summarization** — compress older turns, compare recall quality
3. **Temperature Scoring** — score, promote, and demote memories across tiers

The audience sees the problem (lost context), a partial fix (summaries), and the full solution (self-organizing tiered memory with temperature scoring).

### Prerequisites

- Python 3.10+
- `numpy` for cosine similarity
- `sentence-transformers` for embeddings (or `openai` SDK)
- `tiktoken` for token counting
- (Optional) `rich` for terminal table display, `matplotlib` for charts

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         SHARED SCENARIO                          │
│                    (25-turn multi-topic chat)                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌──────────────┐   ┌───────────────┐   ┌─────────────────────────┐
│   Phase 1    │   │   Phase 2     │   │      Phase 3            │
│   Sliding    │   │  Summarize +  │   │  Temperature-Scored     │
│   Window     │   │  RAG Recall   │   │  Tiered Memory          │
│  (last 5)    │   │              │   │                         │
│              │   │              │   │  ┌─────┐ ┌─────┐ ┌─────┐│
│              │   │              │   │  │ HOT │ │WARM │ │COLD ││
│              │   │              │   │  │≥0.7 │ │0.5- │ │<0.5 ││
│              │   │              │   │  │     │ │ 0.7 │ │     ││
│              │   │              │   │  └──▲──┘ └▲──┬─┘ └──┬──┘│
│              │   │              │   │     └─────┘  └──────┘   │
└──────┬───────┘   └──────┬───────┘   └────────────┬────────────┘
       │                  │                        │
       ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   COMPARISON DASHBOARD                            │
│  Token Usage · Factual Recall · Tier Movements · Memory Scores   │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Steps

#### Part 1: Sliding Window (The Problem)

**Step 1**: Define a 25-turn scenario script where early facts matter later.

```python
scenario = [
    "Hi, I'm Alex. I'm working on a Python microservice.",
    "We deploy to AWS using ECS.",
    "The service handles payment processing with Stripe.",
    "We're seeing timeouts on the /checkout endpoint.",
    "The database is PostgreSQL on RDS, port 5432.",
    "Our CI/CD pipeline uses GitHub Actions.",
    "The team uses Docker with multi-stage builds.",
    "We recently added a Redis cache for sessions.",
    "The monitoring stack is Prometheus + Grafana.",
    "We suspect the Stripe webhook is causing the timeout.",
    "The webhook handler does a synchronous DB write.",
    "We tried increasing the timeout to 30s but it didn't help.",
    "The error logs show connection pool exhaustion.",
    "Our pool size is set to 5 connections.",
    "Traffic spikes happen during lunch hours, 11am-1pm.",
    "We also have a batch job that runs at noon.",
    "The batch job processes refunds from the previous day.",
    "It opens 3 long-running DB connections.",
    "So during peak, we have 3 batch + N request connections.",
    "We're considering PgBouncer for connection pooling.",
    # --- Recall questions ---
    "What cloud provider and service are we using?",          # Turn 2: AWS ECS
    "What's our current DB connection pool size?",            # Turn 14: 5
    "What payment processor do we use?",                      # Turn 3: Stripe
    "Why do we think the timeouts happen at lunch?",          # Turns 15-19
    "Summarize the root cause and proposed solution.",         # Full recall
]
```

**Step 2**: Implement the sliding window agent.

```python
class SlidingWindowAgent:
    def __init__(self, window_size=5):
        self.history = []
        self.window_size = window_size
        self.token_log = []

    def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        window = self.history[-self.window_size:]
        tokens_used = count_tokens(window)
        self.token_log.append(tokens_used)

        response = llm_call(messages=window)
        self.history.append({"role": "assistant", "content": response})
        return response, tokens_used
```

**Step 3**: Run and observe failures on recall questions (turns 21-25).

#### Part 2: Summarization (Partial Fix)

**Step 4**: Implement summarization agent to compress older turns.

```python
class SummarizationAgent:
    def __init__(self, window_size=3):
        self.history = []
        self.summary = ""
        self.window_size = window_size
        self.token_log = []

    def _summarize(self, old_turns):
        prompt = f"Previous summary:\n{self.summary}\n\nNew turns:\n"
        for t in old_turns:
            prompt += f"{t['role']}: {t['content']}\n"
        prompt += "\nWrite a concise summary preserving all key facts:"
        return llm_call(messages=[{"role": "user", "content": prompt}])

    def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})

        if len(self.history) > self.window_size * 2:
            old_turns = self.history[:-self.window_size]
            self.summary = self._summarize(old_turns)
            self.history = self.history[-self.window_size:]

        messages = []
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{self.summary}"
            })
        messages.extend(self.history[-self.window_size:])

        tokens_used = count_tokens(messages)
        self.token_log.append(tokens_used)

        response = llm_call(messages=messages)
        self.history.append({"role": "assistant", "content": response})
        return response, tokens_used
```

**Step 5**: Run the same scenario — observe improved recall but note lossy compression.

#### Part 3: Temperature-Scored Tiered Memory (Full Solution)

**Step 6**: Define the Memory data model.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class Memory:
    id: str
    text: str
    embedding: list[float]
    memory_type: str              # SEMANTIC, EPISODIC, PROCEDURAL
    temperature: float = 0.5
    tier: str = "cold"            # hot, warm, cold
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    entities: list[str] = field(default_factory=list)
```

**Step 7**: Implement the temperature scoring formula.

```python
from math import exp
import numpy as np

WEIGHTS = {
    "recency": 0.30,
    "relevance": 0.25,
    "frequency": 0.20,
    "entity_overlap": 0.15,
    "agent_match": 0.10,
}

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def compute_temperature(memory, query_embedding, query_entities, agent_id="default", max_access=50):
    hours_ago = (datetime.now() - memory.last_accessed).total_seconds() / 3600
    recency = exp(-0.1 * hours_ago)
    relevance = cosine_similarity(query_embedding, memory.embedding)
    frequency = min(memory.access_count / max_access, 1.0)

    if query_entities and memory.entities:
        overlap = len(set(query_entities) & set(memory.entities)) / len(set(query_entities))
    else:
        overlap = 0.0

    agent = 1.0 if agent_id == "default" else 0.0

    return (
        WEIGHTS["recency"] * recency +
        WEIGHTS["relevance"] * relevance +
        WEIGHTS["frequency"] * frequency +
        WEIGHTS["entity_overlap"] * overlap +
        WEIGHTS["agent_match"] * agent
    )
```

**Step 8**: Implement tier promotion and demotion.

```python
def update_tiers(memories):
    promotions, demotions = [], []
    tier_order = ["cold", "warm", "hot"]

    for m in memories:
        old_tier = m.tier
        if m.temperature >= 0.70:
            m.tier = "hot"
        elif m.temperature >= 0.50:
            m.tier = "warm"
        else:
            m.tier = "cold"

        if m.tier != old_tier:
            if tier_order.index(m.tier) > tier_order.index(old_tier):
                promotions.append((m.id, old_tier, m.tier, m.temperature))
            else:
                demotions.append((m.id, old_tier, m.tier, m.temperature))

    return promotions, demotions
```

**Step 9**: Build the tiered memory agent that stores every turn as a memory, scores on recall, and promotes/demotes.

```python
class TieredMemoryAgent:
    def __init__(self, embed_fn):
        self.memories = []
        self.embed_fn = embed_fn
        self.token_log = []
        self.tier_history = []  # Track tier changes per turn

    def store(self, text, entities, memory_type="EPISODIC"):
        emb = self.embed_fn(text)
        mem = Memory(
            id=f"mem_{len(self.memories):03d}",
            text=text,
            embedding=emb,
            memory_type=memory_type,
            entities=entities,
        )
        self.memories.append(mem)

    def recall(self, query, query_entities, top_k=10):
        query_emb = self.embed_fn(query)

        for m in self.memories:
            m.temperature = compute_temperature(m, query_emb, query_entities)

        promotions, demotions = update_tiers(self.memories)
        self.tier_history.append({"promotions": promotions, "demotions": demotions})

        # Update access stats for hot/warm memories
        for m in self.memories:
            if m.tier in ("hot", "warm"):
                m.access_count += 1
                m.last_accessed = datetime.now()

        # Return top-k by temperature
        ranked = sorted(self.memories, key=lambda m: m.temperature, reverse=True)
        return ranked[:top_k], promotions, demotions

    def chat(self, user_message):
        entities = extract_keywords(user_message)
        self.store(user_message, entities)

        results, promotions, demotions = self.recall(user_message, entities)

        context = "\n".join([f"[{m.tier.upper()} t={m.temperature:.2f}] {m.text}" for m in results])
        messages = [
            {"role": "system", "content": f"Relevant memories:\n{context}"},
            {"role": "user", "content": user_message},
        ]

        tokens_used = count_tokens(messages)
        self.token_log.append(tokens_used)

        response = llm_call(messages=messages)
        return response, tokens_used, promotions, demotions
```

**Step 10**: Run all three agents through the scenario, collect and compare.

```python
agents = {
    "sliding_window": SlidingWindowAgent(window_size=5),
    "summarization": SummarizationAgent(window_size=3),
    "tiered_memory": TieredMemoryAgent(embed_fn=embed),
}

results = {name: [] for name in agents}

for i, turn in enumerate(scenario):
    for name, agent in agents.items():
        if name == "tiered_memory":
            resp, tokens, promos, demos = agent.chat(turn)
            results[name].append({
                "turn": i, "response": resp, "tokens": tokens,
                "promotions": promos, "demotions": demos,
            })
        else:
            resp, tokens = agent.chat(turn)
            results[name].append({"turn": i, "response": resp, "tokens": tokens})
```

#### Comparison Dashboard

**Step 11**: Visualize the results.

**Token usage over turns** (line chart):
```python
import matplotlib.pyplot as plt

for name, data in results.items():
    plt.plot([d["tokens"] for d in data], label=name)
plt.xlabel("Turn")
plt.ylabel("Tokens Sent")
plt.legend()
plt.title("Token Usage by Strategy")
plt.show()
```

**Tier movement heatmap** (for tiered memory):
```python
def print_tier_summary(agent):
    tiers = {"hot": [], "warm": [], "cold": []}
    for m in agent.memories:
        tiers[m.tier].append(m.id)
    for tier, mems in tiers.items():
        print(f"  {tier.upper():5s}: {len(mems)} memories")
```

**Recall accuracy table** (turns 21-25):
```
Question                              │ Sliding │ Summary │ Tiered
──────────────────────────────────────┼─────────┼─────────┼────────
What cloud provider?                  │   ✗     │   ✓     │  ✓
DB connection pool size?              │   ✗     │   ~     │  ✓
Payment processor?                    │   ✗     │   ✓     │  ✓
Why timeouts at lunch?                │   ✗     │   ~     │  ✓
Summarize root cause + solution?      │   ✗     │   ~     │  ✓
```

### Expected Outcome

- **Sliding window** fails all recall questions beyond its window
- **Summarization** gets some right but loses detail (pool size, exact times)
- **Tiered memory** retains key facts because frequently referenced memories (AWS, Stripe, pool size) get promoted to hot tier
- Token usage is highest for sliding window at scale, most efficient for tiered memory (only injects relevant context)

### Time Estimate

~20-25 minutes live walkthrough

---

## Demo B: Context Budget Tracker

### Objective

Build middleware that sits between your agent and the LLM, logging token usage broken down by section (system prompt, tool definitions, retrieved memories, conversation history, task state). Visualize where your context budget is going.

### Prerequisites

- Python 3.10+
- `tiktoken` for token counting
- `openai` SDK
- (Optional) `matplotlib` for charts or `rich` for terminal tables

### Architecture

```
┌────────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Agent Logic   │────▶│  Budget Tracker   │────▶│  LLM API     │
│                │     │  (middleware)      │     │              │
│  Assembles:    │     │  Counts tokens    │     │  Receives    │
│  - system      │     │  per section,     │     │  final       │
│  - tools       │     │  logs breakdown,  │     │  messages[]  │
│  - memories    │     │  warns on overflow│     │              │
│  - history     │     │                   │     │              │
│  - task state  │     │                   │     │              │
└────────────────┘     └───────┬───────────┘     └──────────────┘
                               │
                       ┌───────▼───────────┐
                       │  Budget Log       │
                       │  (per-request     │
                       │   breakdown)      │
                       └───────────────────┘
```

### Implementation Steps

#### Step 1: Define context sections

```python
from enum import Enum

class ContextSection(Enum):
    SYSTEM_PROMPT = "system_prompt"
    TOOL_DEFINITIONS = "tool_definitions"
    RETRIEVED_MEMORIES = "retrieved_memories"
    CONVERSATION_HISTORY = "conversation_history"
    TASK_STATE = "task_state"
```

#### Step 2: Build the budget tracker

```python
import tiktoken

class BudgetTracker:
    def __init__(self, model="gpt-4", max_tokens=128_000, budget_pct=0.25):
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
        self.budget_limit = int(max_tokens * budget_pct)  # 25% for context
        self.log = []

    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def analyze(self, sections: dict[ContextSection, str]) -> dict:
        breakdown = {}
        total = 0

        for section, content in sections.items():
            tokens = self.count(content)
            breakdown[section.value] = tokens
            total += tokens

        record = {
            "breakdown": breakdown,
            "total_context_tokens": total,
            "budget_limit": self.budget_limit,
            "utilization_pct": round(total / self.max_tokens * 100, 1),
            "over_budget": total > self.budget_limit,
            "remaining_for_response": self.max_tokens - total,
        }

        self.log.append(record)
        return record
```

#### Step 3: Create the middleware wrapper

```python
def tracked_llm_call(tracker, sections, llm_client):
    report = tracker.analyze(sections)

    if report["over_budget"]:
        print(f"WARNING: Context at {report['utilization_pct']}% "
              f"({report['total_context_tokens']}/{tracker.max_tokens} tokens)")

    messages = assemble_messages(sections)

    response = llm_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    return response, report
```

#### Step 4: Simulate an agent conversation with growing context

```python
system_prompt = "You are a helpful DevOps assistant..."
tool_defs = '{"tools": [{"name": "run_command", ...}, {"name": "read_file", ...}]}'

for turn in range(1, 16):
    sections = {
        ContextSection.SYSTEM_PROMPT: system_prompt,
        ContextSection.TOOL_DEFINITIONS: tool_defs,
        ContextSection.RETRIEVED_MEMORIES: get_memories(turn),
        ContextSection.CONVERSATION_HISTORY: get_history(turn),
        ContextSection.TASK_STATE: get_task_state(turn),
    }

    response, report = tracked_llm_call(tracker, sections, client)
    print_budget_report(turn, report)
```

#### Step 5: Visualize the budget over time

```
Turn | System | Tools | Memory | History | Task  | Total  | Util%
-----+--------+-------+--------+---------+-------+--------+------
   1 |  450   | 1200  |   320  |    180  |  200  |  2350  | 1.8%
   5 |  450   | 1200  |  2800  |   3200  |  850  |  8500  | 6.6%
  10 |  450   | 1200  |  6400  |   8900  | 1200  | 18150  | 14.2%
  15 |  450   | 1200  | 12000  | 16500   | 1800  | 31950  | 24.9%
```

#### Step 6: Add budget enforcement strategies

```python
def enforce_budget(tracker, sections):
    report = tracker.analyze(sections)

    if not report["over_budget"]:
        return sections

    if report["breakdown"]["conversation_history"] > 5000:
        sections[ContextSection.CONVERSATION_HISTORY] = summarize_history(
            sections[ContextSection.CONVERSATION_HISTORY]
        )

    if report["breakdown"]["retrieved_memories"] > 8000:
        sections[ContextSection.RETRIEVED_MEMORIES] = keep_top_k_memories(
            sections[ContextSection.RETRIEVED_MEMORIES], k=5
        )

    return sections
```

### Expected Outcome

- The audience sees exactly where tokens are spent per request
- Conversation history and retrieved memories are the biggest growth areas
- The budget enforcement demo shows practical strategies for staying within limits

### Time Estimate

~10 minutes live coding + discussion

---

## Choosing a Demo

| Factor | Demo A | Demo B |
|--------|--------|--------|
| **Audience level** | Beginner → Intermediate | Beginner-Intermediate |
| **Concept taught** | Context strategies + self-organizing memory | Token economics |
| **Requires embeddings** | Yes (Phase 3) | No |
| **Visual impact** | Recall comparison + tier movements | Budget charts |
| **Standalone value** | Full learning arc (problem → solution) | Production readiness |

**Recommendation**: Run Demo A as the main demo — it tells a complete story from problem to solution. If time remains, Demo B makes a great follow-up showing how to monitor context usage in production.
