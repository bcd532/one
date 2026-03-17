# one

Persistent memory and autonomous research platform for Claude Code. 37 Python files, ~17,000 lines. MIT licensed.

> **Alpha software.** Core memory and recall works. Autonomous features (`/auto`, `/morgoth`) are experimental -- they produce real results mixed with confident-sounding noise. I'm being upfront about what's real and what's aspirational.

## Why I built this

I use Claude Code daily for everything -- a trading bot, a C99 TUI engine, patent research. The problem: every session starts from zero. Claude doesn't remember what we decided yesterday, what broke last week, or what patterns I prefer. I wanted persistent memory that actually works -- not a CLAUDE.md file I have to maintain by hand, but a system that learns, recalls, and eventually reasons across everything I've ever worked on.

It started as a memory layer. Then I added entity extraction. Then rules. Then I let the system argue with itself about what it knows. That's when it got interesting.

## What it does

**Core (stable):**
- **Persistent memory** -- conversations, decisions, and context stored in SQLite with 4096-dim HDC vector embeddings, recalled by cosine similarity when relevant
- **Rule learning** -- repeated preferences detected, organized into a contextual tree, injected per-turn based on what I'm working on
- **Entity extraction** -- files, concepts, tools, code patterns, URLs extracted and linked into a knowledge graph
- **Session tracking** -- full history with cost and messages, exportable
- **Foundry sync** -- optionally push the knowledge graph to Palantir AIP as ontology objects

**Autonomous research (experimental):**
- **Morgoth mode** (`/morgoth`) -- autonomous research loop: understand, research, dialectic challenge, synthesize, build, verify, iterate. Uses Claude as the reasoning engine. Runs indefinitely.
- **Auto mode** (`/auto`) -- give Claude a goal, it plans/executes/tests/iterates unattended
- **Dialectic engine** -- thesis/antithesis/synthesis chains that challenge findings adversarially
- **Contradiction mining** -- finds conflicting claims in the knowledge graph and resolves them
- **Cross-domain synthesis** -- discovers structural patterns across unrelated domains

**Zero Hallucination Engine (new):**
- AST-parses Python edits, extracts every SQL query, checks column names against live `PRAGMA table_info`
- Multi-language: Python, C, JS, HTML, CSS, JSON
- Codebase ontology: maps every function (483), every call (4,300+), every file dependency (122)
- Impact analysis: if I rename a function, it tells me which files break
- Post-edit hooks block bad code before it saves

## What happened when I let it run

I pointed `/morgoth` at my trading bot codebase with the goal "research compound trading strategies for Kalshi binary options." I let it run for ~16 hours across 6 iterations. Here's what it actually did:

- Produced **420 research findings** citing real academic sources (Lopez de Prado 2018, Bailey et al. 2015, Vince 1992)
- Mined **472 contradictions** between findings, resolved 14 with structured reasoning
- **Deprecated 589 weak findings** through adversarial dialectic challenge
- Discovered **21 universal structural patterns** across domains I never directed it to explore
- Generated **27 cross-domain syntheses** connecting software engineering, molecular biology, quantitative finance, and AI security

### What it found

**TRUST_LAUNDERING** -- the system described a universal pattern where adversarial content from a low-trust channel gets misrouted so the receiving system treats it as high-trust. I realized this is exactly the "ghost fill" problem in my trading bot: a fill that didn't happen gets counted as a real win and feeds into calibration. The fake data launders its provenance through the system.

**SIGNAL_HOMOGRAPHY** -- identical signals carry opposite meanings depending on interpretive frame. This is the multi-timeframe trading problem: the same price at $95,000 means "bullish breakout" on the 15-minute chart and "resistance rejection" on the daily. Same signal, opposite meaning, depending which frame you read it in.

**Deferred materialization (Foundry x RNA)** -- the system connected Python's lazy import pattern to RNA transcription. Both are the same architecture: a dormant capability catalogue (SDK modules / DNA) preserved structurally but suppressed until a local invocation context is entered (function body / transcription signal). The purpose in both cases is surviving when the activating infrastructure isn't present.

**Manifold membership** -- a bug where `recall("")` returned empty results (zero vector, undefined cosine similarity) led to a meta-hypothesis: "quality metrics presuppose manifold membership and cannot distinguish low-quality content from structurally undefined state." A zero-vector query doesn't return bad results -- it's outside the space where results exist. The system formalized why certain classes of bugs are invisible to quality checks.

### Honest caveats

I want to be upfront about what's real and what's not:

- The "Active Inference gate" is a **weighted scoring function inspired by AIF**, not real free energy minimization. There's no generative model or belief updating. The field names (`aif_confidence`, `regime_tag`) reflect where I want to take this, not where it is today.
- The `tm_label` field references Tsetlin Machines but **there is no TM implementation**. It's a string tag. I have a provisional patent on combining HDC + TM + AIF but the TM and AIF parts aren't built yet.
- Confidence scores on LLM syntheses **inflate** because the confidence scorer is also an LLM -- it can't catch its own hallucinations. There's an [open PR](https://github.com/bcd532/one/pull/1) adding epistemic safety guards.
- The universal patterns (TRUST_LAUNDERING, SIGNAL_HOMOGRAPHY, etc.) are conceptual connections generated by Claude. Some map to real phenomena. Some might be creative naming of things practitioners already intuit. I haven't validated them against published literature.
- Morgoth is **token-intensive**. A 16-hour run uses significant API credits.

## Install

```bash
git clone https://github.com/bcd532/one.git
cd one
pip install -e .
```

Requires Python 3.10+ and [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) installed and authenticated.

## Usage

```bash
one                        # launch TUI
one "prompt"               # with initial prompt
one -m sonnet              # different model
one -c                     # continue last session
one -d /path/to/project    # set working directory
```

### Commands

| Command | Description |
|---------|-------------|
| `/recall` | Force memory recall |
| `/rules` | Show rule tree |
| `/stats` | Memory store statistics |
| `/entities` | Show knowledge graph |
| `/search <q>` | Search memories |
| `/sessions` | List past sessions |
| `/auto <goal>` | Autonomous agent loop |
| `/morgoth <goal>` | Research + build + verify loop |
| `/research <topic>` | Deep research with gap analysis |
| `/synthesize` | Cross-domain insight generation |
| `/verify` | Run Zero Hallucination Engine |
| `/health` | Knowledge graph health |
| `/audit` | Quality audit |

### CLI tools

```bash
one map       # map codebase: symbols, calls, deps
one verify    # verify all files against live schemas
one ground    # populate ground truths from introspection
one sync      # push everything to Palantir Foundry
one server    # REST API on :4111
```

## How memory works

1. **Encode** -- message encoded into 4096-dim hypervector (HDC: character trigrams + word vectors + bigrams)
2. **Gate** -- scoring function evaluates novelty, content quality, information type. Noise dropped.
3. **Store** -- qualifying messages stored with HDC vector and metadata in SQLite
4. **Recall** -- on topic shifts or periodic intervals, relevant memories retrieved by cosine similarity and injected into Claude's context
5. **Learn** -- repeated preferences promoted to rules in a contextual tree

## How morgoth works

```
UNDERSTAND --> RESEARCH --> SYNTHESIZE --> BUILD --> VERIFY --> ITERATE
                  |             |            |         |          |
           Claude researches   Cross-domain  Claude    Engine     Weak
           stores findings     pattern       writes    verifies   findings
           with confidence     discovery     code      against    pruned,
           scores                            from      schemas    strong
                  |             |            findings             reinforced
           Dialectic runs      Contradictions
           thesis/antithesis   mined and
           /synthesis          resolved
```

Each iteration self-prunes. The system argues with itself until only adversarially-tested knowledge remains.

## Optional: Foundry backend

```bash
mkdir -p ~/.one
echo "host=your-foundry-hostname" > ~/.one/config
echo "your-token" > ~/.one/token
one sync
```

## Optional: Gemma for condensation

```bash
ollama pull gemma3:4b
```

Used only for compressing recalled memories before injection. Claude is the primary reasoning engine for all research, dialectic, and synthesis operations.

## Status

| Feature | Status |
|---------|--------|
| Memory / recall | Stable |
| Entity graph | Stable |
| Rule learning | Stable |
| Sessions | Stable |
| Zero Hallucination Engine | Working, new |
| Foundry sync (objects) | Working |
| Foundry sync (links) | Needs config |
| `/auto` | Experimental |
| `/morgoth` | Experimental, token-intensive |
| Dialectic / synthesis | Working, confidence inflation issue |
| Epistemic safety | [PR #1](https://github.com/bcd532/one/pull/1) |

## License

MIT
