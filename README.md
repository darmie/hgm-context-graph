# Workshop: Hierarchical Context Graph Systems

**Building Cognitive Memory for AI Agents**

## Materials

| File | Description |
|------|-------------|
| [hgm-core-concepts.ipynb](hgm-core-concepts.ipynb) | Interactive Jupyter notebook with runnable code |
| [hgm-core-concepts-slides.md](hgm-core-concepts-slides.md) | MARP presentation slides |

## Topics Covered

1. **Context Engineering** - How HGM compares to Memory Streams, MemGPT, Reflexion, RAPTOR
2. **Memory Types** - SEMANTIC, EPISODIC, PROCEDURAL, EMOTIONAL (Tulving's taxonomy)
3. **Agent State Management** - Focus tracking, episodes, context management
4. **Temperature Scoring** - 5-factor weighted relevance computation
5. **Three-Tier Memory** - Hot (<1ms), Warm (<50ms), Cold (<200ms) architecture
6. **Agentic RAG** - Active recall with automatic promotion/demotion
7. **Pattern Graph** - Learned response strategies and skill triggers
8. **LLM Orchestration** - Combining rule-based scoring with LLM reasoning
9. **Chat Continuity** - Labeled memories (USER_QUERY, AGENT_THOUGHT, PATTERN)
10. **Mode Selection** - FAST, AGENT, PATTERN_DIRECT, WORKFLOW routing

## System Setup

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Notebook runtime |
| NumPy | Latest | Vector operations |
| Node.js | 18+ | MARP slides (optional) |

### Option 1: Minimal Setup (Notebook Only)

```bash
# Using pip
pip install numpy jupyter

# Or using uv (recommended)
uv pip install numpy jupyter
```

### Option 2: Full HGM Development Setup

For running the actual HGM system (not required for workshop):

```bash
# Clone and setup
git clone https://github.com/darmie/hgm.git
cd hgm

# Start infrastructure
cd docker && docker compose up -d && cd ..

# Install Python packages
uv sync

# Build Rust extension
cd hgm-core-rs && maturin develop --release && cd ..
```

### Option 3: VS Code Setup

1. Install [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
2. Install [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
3. Install [MARP extension](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
4. Open `workshops/` folder
5. Run notebook cells directly in VS Code

---

## Running the Notebook

The notebook is **self-contained** and requires only NumPy.

```bash
# Navigate to workshops
cd workshops

# Start Jupyter
jupyter notebook hgm-core-concepts.ipynb

# Or with JupyterLab
jupyter lab hgm-core-concepts.ipynb
```

**No external infrastructure required** - no PostgreSQL, Redis, or API keys.

### Notebook Sections

| Section | Cells | Description |
|---------|-------|-------------|
| 1-2 | Setup | Imports and configuration |
| 3-4 | Memory Types | SEMANTIC, EPISODIC, PROCEDURAL, EMOTIONAL |
| 5-6 | Agent State | Focus tracking, episodes |
| 7-8 | Temperature | 5-factor scoring system |
| 9-10 | Hot Tier | Working memory with LRU eviction |
| 11-12 | Three-Tier | Hot/Warm/Cold simulation |
| 13-14 | Pattern Graph | Entity-pattern linking |
| 15-16 | Pattern Scoring | Multi-factor relevance |
| 17-18 | Mode Selection | FAST/AGENT/PATTERN/WORKFLOW |
| 19-20 | LLM Orchestrator | AI-enhanced decisions |
| 21-24 | Chat Continuity | Labeled memories demo |
| 25+ | Exercises | Hands-on practice |

---

## Viewing the Slides

### Option 1: MARP CLI

```bash
# Install MARP
npm install -g @marp-team/marp-cli

# Generate PDF (for printing/sharing)
marp hgm-core-concepts-slides.md --pdf --allow-local-files

# Generate HTML (for web hosting)
marp hgm-core-concepts-slides.md --html

# Live preview with hot reload
marp -p hgm-core-concepts-slides.md
```

### Option 2: VS Code Extension

1. Install [MARP for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
2. Open `hgm-core-concepts-slides.md`
3. Click "Preview" icon or press `Ctrl+Shift+V`
4. Export via Command Palette: "Marp: Export Slide Deck"

### Option 3: Online Viewer

Upload to [MARP Web](https://web.marp.app/) for quick viewing.

---

## Troubleshooting

### Notebook Issues

**Import errors:**
```bash
pip install numpy --upgrade
```

**Kernel not found:**
```bash
python -m ipykernel install --user --name=hgm-workshop
```

### Slides Issues

**MARP not rendering:**
```bash
# Ensure Node.js is installed
node --version  # Should be 18+

# Reinstall MARP
npm uninstall -g @marp-team/marp-cli
npm install -g @marp-team/marp-cli
```

**PDF export fails:**
```bash
# Install Chrome/Chromium for PDF generation
marp hgm-core-concepts-slides.md --pdf --allow-local-files
```

### Key Concepts

#### Memory Labels for Chat Continuity

```python
class MemoryLabel(str, Enum):
    USER_QUERY = "user_query"       # User's message
    AGENT_THOUGHT = "agent_thought" # Agent's reasoning
    PATTERN = "pattern"             # Learned strategy
```

#### Temperature Scoring Formula

```
T = 0.30 * recency + 0.15 * frequency + 0.35 * relevance
  + 0.15 * entity_overlap + 0.05 * agent_match
```

#### Three-Tier Architecture

```
HOT  (<1ms)   - Rust in-memory, active session
WARM (<50ms)  - Redis, recent interactions
COLD (<200ms) - PostgreSQL, long-term knowledge
```
