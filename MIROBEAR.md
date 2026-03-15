# MIROBEAR: The Palantír

Not Palantir the company. The Palantír — the seeing stone from Tolkien
that lets you see ANYTHING, ANYWHERE, across ALL of space and time.

That is what one becomes. A seeing stone for knowledge. Point it at
any problem and it sees the answer through the noise, across every
domain, through every contradiction, past every dead end, until the
truth is found.

This is not a feature list. This is an architecture for building
the most powerful single-user intelligence amplification system
ever created. Every system below is MANDATORY. Every system must
be WORLD CLASS. Not functional. Not "it works." WORLD CLASS.

89 PhD researchers couldn't match what this system does because
they can't share memory in real-time, can't argue with themselves
at machine speed, can't search every paper ever written in seconds,
can't run experiments and iterate 24/7 without sleep, and can't
transfer structural insights across domains they've never studied.

This system can. Build it.

---

## SYSTEM 1: THE SWARM — Coordinated Multi-Agent Intelligence

### Not just parallel agents. An ORCHESTRA.

```
/swarm "cure pancreatic cancer" --depth unlimited
/swarm ./CANCER_RESEARCH.md --agents 8 --strategy deep
/swarm "find arbitrage in global bond markets" --agents 5 --strategy adversarial
/swarm "design a spacecraft that can reach Mars in 30 days" --agents 6
```

### The Conductor

The Conductor is NOT Gemma. The Conductor is a DEDICATED Claude session
whose ONLY job is orchestration. It never researches. It never writes code.
It reads ALL agent outputs, thinks about strategy, and directs.

The Conductor:
1. Reads the goal
2. Decomposes it into research fronts (not tasks — FRONTS)
3. Assigns agent roles optimized for each front
4. Monitors all agent outputs in real-time via shared memory
5. Every N minutes:
   - Reads ALL new findings from ALL agents
   - Runs cross-agent synthesis
   - Detects when agents are duplicating effort → redirects
   - Detects when agents are stuck → reassigns or spawns new agents
   - Detects breakthroughs → amplifies by assigning MORE agents to that front
   - Detects contradictions → spawns a dedicated dialectic between the disagreeing agents
   - Updates the global research map
   - Broadcasts strategic updates to all agents
6. Dynamically scales: if a front is exhausted, kills that agent, spawns one on a new front
7. Never stops until the goal is achieved or the user says stop

### Agent Roles — Full Taxonomy

Each agent gets a system prompt defining its role, personality, and mandate:

**Surveyor** — Maps the entire landscape. Who are the key players?
What are the major schools of thought? What's consensus? What's
controversial? Produces a structured map of the field. Wide and shallow.

**Mechanist** — Goes DEEP on how things work. Not "what" but "why"
and "how" at the lowest level. Molecular pathways. Mathematical proofs.
System architectures. Algorithm internals. The person who understands
the engine, not the car.

**Contrarian** — The attack dog. Takes every finding from every other
agent and tries to DESTROY it. Searches for failed trials, retracted
papers, logical flaws, statistical errors, confounding variables.
If a finding survives the Contrarian, it's real.

**Analogist** — The cross-pollinator. Reads findings from all agents
and asks: "Where have I seen this STRUCTURE before in a completely
different domain?" Transfers solution patterns. Generates the
hypotheses nobody else would think of because nobody else looks
across all fields simultaneously.

**Synthesizer** — The meta-thinker. Reads EVERYTHING. Connects
findings that no individual agent connected. Builds the narrative.
"Here's what we know, here's what it means, here's what's still
missing, and here's the hypothesis that emerges from combining
insights A, B, and C."

**Verifier** — The fact-checker. Takes high-confidence findings
and tries to REPLICATE the evidence. Finds original sources.
Checks sample sizes. Checks if studies were retracted. Checks
if results were reproduced. Downgrades anything that doesn't
hold up.

**Experimentalist** — The hands-on builder. When enough research
exists, starts DOING. Writes code. Runs experiments. Tests
hypotheses with actual data. Produces concrete results, not
just text analysis.

**Historian** — Finds what was tried BEFORE and why it failed.
Every problem has a history of failed attempts. Understanding
WHY they failed is more valuable than knowing they failed.

**Futurist** — Projects forward. Given what we know now, what
becomes possible in 1 year? 5 years? What would need to be
true for the breakthrough to happen? Works backwards from the
desired future state.

**Devil's Advocate** — Different from Contrarian. The Contrarian
attacks findings. The Devil's Advocate attacks the GOAL ITSELF.
"Are we even solving the right problem? What if the premise is
wrong? What if there's a completely different approach nobody
is considering?"

### Shared Memory Protocol

ALL agents read from and write to the SAME knowledge store.
But not raw access — structured protocol:

1. **Finding**: agent stores a finding with confidence + source + evidence
2. **Broadcast**: finding is immediately visible to all other agents
3. **React**: other agents can REACT to findings:
   - Support (add corroborating evidence → boost confidence)
   - Challenge (add counter-evidence → trigger dialectic)
   - Extend (add related finding → trigger synthesis)
   - Apply (use finding in another domain → trigger analogy)
4. **Conductor reads all reactions** and decides next moves

### Rate Limit Strategy

$200 max subscription. Multiple agents hit limits faster.

- Stagger starts: 15 second gaps between agent launches
- Priority queue: Conductor always gets priority
- When rate limited: pause lowest-priority agent, continue highest
- Agent priority = how many breakthroughs they've produced
- Agents that haven't produced findings in 20 turns get killed and replaced
- Track total turns across all agents, show in TUI
- Automatic throttle when approaching 80% of rate window

### Implementation: one/swarm.py

Full SwarmCoordinator class with:
- `start(goal, num_agents, strategy)` — launch the swarm
- `stop()` — halt all agents gracefully
- `scale(n)` — add or remove agents dynamically
- `focus(agent_id)` — zoom TUI into one agent
- `inject(text)` — send context to all agents
- `status()` — return full swarm state
- `kill_agent(agent_id)` — remove one agent
- `spawn_agent(role)` — add one agent with specific role
- `get_findings()` — all findings across all agents
- `get_contradictions()` — all active contradictions
- `get_syntheses()` — all cross-agent syntheses

---

## SYSTEM 2: ADVERSARIAL DIALECTIC ENGINE

### Every piece of knowledge is forged through argument

Not a filter. An ENGINE that produces higher-quality knowledge
than any single perspective could.

### The Dialectic Chain

```
THESIS (from any agent):
  "KRAS G12C inhibitors show 40% response in lung adenocarcinoma"

ANTITHESIS (auto-generated OR from Contrarian agent):
  "Sotorasib Phase III showed only 5.6 month PFS vs 4.5 for docetaxel.
   40% response but minimal survival benefit. Resistance develops in
   80% of patients within 12 months via bypass pathway activation."

SYNTHESIS:
  "KRAS G12C inhibition works mechanistically but monotherapy fails
   due to rapid resistance. The response rate is real but clinically
   insufficient alone. Combination approaches targeting bypass pathways
   simultaneously may prevent resistance."

VERIFICATION:
  "Search: KRAS G12C combination therapy trials 2024-2026"
  → Found: adagrasib + cetuximab Phase II: 46% response, 8.2 month PFS
  → Found: sotorasib + panitumumab Phase Ib: 30% response
  → SYNTHESIS CONFIRMED: combination approach shows improved durability

META-SYNTHESIS:
  "The pattern of monotherapy-resistance-combination is universal in
   targeted oncology. This same arc played out with BRAF inhibitors
   (vemurafenib → vemurafenib+cobimetinib) and EGFR inhibitors
   (erlotinib → erlotinib+ramucirumab). STRUCTURAL PATTERN:
   single-target-therapy → resistance-via-bypass → combination-blocking-bypass
   is a UNIVERSAL template for targeted therapy development."
```

That last step — the meta-synthesis — is what 237 MiroFish agents
can NEVER produce. They find correlations. We find UNIVERSAL PATTERNS.

### Dialectic Storage

Store the FULL chain as a linked structure in the knowledge graph:
- thesis_id → antithesis_id → synthesis_id → verification_id → meta_synthesis_id
- Each node has confidence, source, evidence
- The chain itself becomes a queryable entity
- "Show me all dialectics about KRAS" returns the full argument history

### Implementation

Add `DialecticEngine` class to a new one/dialectic.py:
- `challenge(finding)` → generate strongest antithesis
- `synthesize(thesis, antithesis)` → resolution
- `verify(synthesis)` → search for evidence
- `meta_synthesize(syntheses)` → find universal patterns
- `store_chain(chain)` → persist the full dialectic
- `get_chains(project, topic)` → retrieve dialectic history

---

## SYSTEM 3: ANALOGICAL TRANSFER ENGINE

### The machine that sees the Matrix

Not "A is like B." That's metaphor. This is STRUCTURAL ISOMORPHISM.

### Template Extraction

Every finding gets decomposed into domain-independent structure:

```
Finding: "Checkpoint inhibitors block PD-1 receptor on T-cells,
         releasing immune brakes and allowing tumor killing"

Template: {
  mechanism: "blocking",
  target: "inhibitory_receptor",
  location: "effector_cell_surface",
  effect: "releases_suppressed_function",
  outcome: "elimination_of_target",
  domain: "immunology"
}
```

### Cross-Domain Matching

Match templates by STRUCTURE, not content:

```
Software match: {
  mechanism: "removing",
  target: "rate_limiter",
  location: "api_gateway",
  effect: "releases_throttled_traffic",
  outcome: "increased_throughput",
  domain: "distributed_systems"
}

Structural similarity: 0.94
```

### Universal Pattern Extraction

When the same structure appears in 3+ domains, it becomes a UNIVERSAL PATTERN:

```
Pattern: "INHIBITOR_RELEASE"
  Removing a suppressive control mechanism at a gateway point
  releases downstream capacity.

Instances:
  - Immunology: checkpoint inhibitors → immune activation
  - Software: rate limiter removal → throughput increase
  - Economics: deregulation → market activity increase
  - Neuroscience: disinhibition → neural circuit activation
  - Chemistry: catalyst → reaction rate increase

Predictive power: if you encounter a system with suppressive
controls at a gateway, removing them will likely increase
downstream activity. Test this in every new domain.
```

THAT is what 89 PhD researchers can't do. They're each in their
own domain. They can't see that immunology, software, economics,
neuroscience, and chemistry are all running the same algorithm.

### Implementation: one/analogy.py

- `extract_template(finding)` → structural template (use Gemma or Claude)
- `match_templates(template, all_templates)` → ranked matches
- `find_universal_patterns(min_domains=3)` → patterns that span 3+ fields
- `predict_from_pattern(pattern, new_domain)` → generate hypothesis
- `store_pattern(pattern)` → add to knowledge graph as high-value entity
- Templates encoded as HDC vectors for fast similarity matching

---

## SYSTEM 4: CONTRADICTION MINING ENGINE

### Contradictions are not bugs. They're features.

Every contradiction is a signal that our understanding is incomplete.
Resolving contradictions produces the highest-value knowledge in the graph.

### Active Mining

Not passive detection. ACTIVE search:

1. For every new finding, compare against ALL existing findings:
   - Same topic, different conclusion → CONTRADICTION
   - Same conclusion, different evidence → CORROBORATION
   - Related topic, unexpected connection → SYNTHESIS CANDIDATE

2. Score contradictions:
   - MINOR: different measurements (5% vs 7% response rate)
   - MODERATE: different conclusions from similar data
   - CRITICAL: directly opposing claims from credible sources
   - PARADIGM: challenges a foundational assumption

3. PARADIGM contradictions get special treatment:
   - Immediately escalate to Conductor
   - Spawn dedicated agents to investigate
   - These are where Nobel-prize-level insights live

### Resolution Engine

For each contradiction, systematically determine:
- Is one side wrong? (find the methodological flaw)
- Are both right in different contexts? (find the hidden variable)
- Is neither right? (both sides are approximations of a deeper truth)
- Does this reveal a new phenomenon? (BREAKTHROUGH)

### Implementation: one/contradictions.py

- `mine_contradictions(project)` → find all contradictions in graph
- `score_contradiction(a, b)` → severity assessment
- `resolve_contradiction(a, b)` → systematic resolution
- `get_paradigm_contradictions(project)` → the big ones
- `contradiction_dashboard(project)` → formatted for TUI

---

## SYSTEM 5: SELF-VERIFYING KNOWLEDGE ENGINE

### The graph doesn't just grow. It gets more ACCURATE.

### Confidence Lifecycle

Every finding goes through:
```
NEW (confidence from source quality) →
CORROBORATED (confidence boosted by independent support) →
CHALLENGED (confidence reduced by counter-evidence) →
VERIFIED (confidence locked after re-verification) →
STALE (confidence decays after 30 days without re-verification) →
DEPRECATED (confidence dropped below threshold, flagged for removal)
```

### Verification Sweeps

Background process (runs during idle time or on schedule):
1. Pick the N highest-confidence stale findings
2. For each: search for replication, retraction, new evidence
3. Update confidence based on what's found
4. If confidence drops below 0.2 → archive (don't delete — history matters)

### Source Quality Model

Not all sources are equal:
- Peer-reviewed meta-analysis: 0.95
- Peer-reviewed single study: 0.80
- Preprint: 0.60
- Expert blog post: 0.40
- News article: 0.25
- Random webpage: 0.15
- AI-generated (unverified): 0.10

Findings inherit source quality. Multiple independent sources
compound confidence.

### Implementation: add to research.py

- `score_source(url_or_text)` → quality score
- `verify_finding(finding)` → updated confidence
- `run_verification_sweep(project, n=20)` → batch verification
- `get_confidence_distribution(project)` → histogram of confidences
- `archive_deprecated(project)` → clean stale knowledge

---

## SYSTEM 6: ACTIVE QUESTION GENERATION

### The smartest thing you can do is ask the right question

### Information Value Scoring

For every unknown at the knowledge boundary:

```
information_value = (
    unknowns_resolved × 0.3 +      # how many other questions does this answer?
    contradictions_clarified × 0.3 + # does this resolve a contradiction?
    goal_centrality × 0.25 +        # how central to the original goal?
    novelty × 0.15                   # has anyone ever investigated this?
)
```

The question with the highest score gets researched first. ALWAYS.

### Knowledge Frontier Mapping

```
/frontier

FRONTIER for "cure pancreatic cancer"

  EXPLORED (high confidence):
    ● KRAS mutations drive 90% of pancreatic cancers [0.95]
    ● Gemcitabine is standard first-line but <20% response [0.92]
    ● Tumor microenvironment is immunosuppressive [0.88]

  PARTIALLY EXPLORED (needs depth):
    ◐ KRAS G12C inhibitors in combination therapy [0.55]
    ◐ Nanoparticle drug delivery to pancreas [0.43]
    ◐ Microbiome influence on chemo response [0.38]

  UNEXPLORED (high information value):
    ○ Why does the pancreatic stroma prevent drug penetration? [IV: 0.92]
    ○ Can CAR-T cells be engineered for solid tumors? [IV: 0.87]
    ○ What structural analogy from materials science applies to
      drug delivery through dense tissue? [IV: 0.85]

  CONTRADICTIONS (priority targets):
    ⚡ Agent 2 says immunotherapy fails in pancreatic cancer
       Agent 5 found 3 cases of complete response to pembrolizumab
       Information value of resolution: 0.96
```

### Implementation: add to research.py

- `map_frontier(project, goal)` → full frontier with scoring
- `best_question(project, goal)` → single highest-value question
- `update_frontier(project, new_findings)` → recompute after new data
- `frontier_coverage(project, goal)` → percentage of goal space explored

---

## SYSTEM 7: EXECUTABLE VERIFICATION ENGINE

### Hypotheses that can be tested MUST be tested

### Experiment Design

When a hypothesis involves testable claims:

1. **Detect testability**: "X improves Y" → testable if we can measure Y
2. **Design experiment**: save state, modify variable, run measurement
3. **Execute**: run the actual code/test/benchmark
4. **Measure**: capture the quantitative result
5. **Compare**: baseline vs modification
6. **Store**: full experiment record with reproducibility info

### Experiment Types

- **Code experiments**: modify a parameter, run tests, measure metrics
- **Data experiments**: query a dataset, compute statistics, verify claims
- **Mathematical experiments**: compute a proof, verify an identity, test a conjecture
- **Simulation experiments**: model a system, run scenarios, compare outcomes

### Implementation: one/experiments.py

- `is_testable(hypothesis)` → bool + experiment type
- `design_experiment(hypothesis)` → experiment plan
- `run_experiment(plan)` → results
- `compare_to_baseline(results, baseline)` → delta
- `store_experiment(hypothesis, plan, results)` → knowledge graph
- `list_experiments(project)` → all experiments with outcomes

---

## SYSTEM 8: SWARM TUI DASHBOARD

### Watch the Palantír

```
╭─ SWARM: "cure pancreatic cancer" ─── 8 agents ─── 2h 34m ────────────────╮
│                                                                            │
│  ● Conductor    ████████████████████ orchestrating                         │
│  ● Surveyor     ▁▂▃▅▇█▇▅▃  turn 67   "mapping immunotherapy landscape"   │
│  ● Mechanist    ▁▁▂▃▅▇██▇  turn 52   "KRAS pathway signaling cascade"    │
│  ● Contrarian   ▁▂▅▇▅▂▁▁▂  turn 41   "attacking gemcitabine claims"      │
│  ● Analogist    ▁▁▁▂▃▅▇██  turn 38   "FOUND: materials science match!"   │
│  ● Synthesizer  ▁▂▃▃▅▇▇▅▃  turn 29   "connecting stroma + delivery"      │
│  ● Verifier     ▁▁▂▃▅▅▃▂▁  turn 22   "checking pembrolizumab claims"     │
│  ● Historian    ▁▂▃▅▃▂▁▁▂  turn 19   "why tarceva failed in 2007"        │
│                                                                            │
│  ┌─ STATS ───────────────────────┐  ┌─ LATEST BREAKTHROUGH ─────────────┐ │
│  │ findings:     187             │  │ ANALOGIST found structural match  │ │
│  │ hypotheses:    34             │  │ between tumor stroma penetration  │ │
│  │ contradictions: 7 (2 critical)│  │ and concrete permeability in      │ │
│  │ syntheses:     23 (depth 4)   │  │ materials science. Template:      │ │
│  │ experiments:   12 (8 passed)  │  │ "dense_matrix_penetration" maps   │ │
│  │ patterns:       5 universal   │  │ across 4 domains.                 │ │
│  │ coverage:      67%            │  │                                   │ │
│  │ confidence:    0.73 avg       │  │ CONDUCTOR: spawning Experimentalist│ │
│  └───────────────────────────────┘  │ to test nanoparticle delivery     │ │
│                                     │ using materials science approach   │ │
│  ┌─ ACTIVE CONTRADICTIONS ───────┐  └───────────────────────────────────┘ │
│  │ ⚡ CRITICAL: immunotherapy     │                                       │
│  │   efficacy — Agent 2 vs 6     │  ┌─ DIALECTIC IN PROGRESS ──────────┐ │
│  │   Resolution: INVESTIGATING   │  │ THESIS: stroma blocks all drugs  │ │
│  │                               │  │ ANTITHESIS: abraxane penetrates  │ │
│  │ ⚡ CRITICAL: KRAS druggability │  │ SYNTHESIS: penetration is size-  │ │
│  │   Agent 3 vs 4                │  │ dependent, not binary            │ │
│  │   Resolution: CONTEXT-DEPENDENT│ │ VERIFYING...                     │ │
│  └───────────────────────────────┘  └──────────────────────────────────┘ │
│                                                                           │
│  rate: 342/500 turns used │ token: 179d remaining │ $4.23 this session    │
╰───────────────────────────────────────────────────────────────────────────╯
│ /focus <agent>  /inject <text>  /stop  /scale <n>  /health  /frontier     │
```

### TUI Features

- **Sparklines per agent** — visual activity trend
- **Real-time breakthrough alerts** — flash when an agent finds something big
- **Contradiction panel** — shows active disputes with resolution status
- **Live dialectic** — watch arguments unfold in real-time
- **Focus mode** — `/focus mechanist` zooms into that agent's full stream
- **Inject** — `/inject "also look at autophagy"` sends to all agents
- **Scale** — `/scale 12` adds 4 more agents on the fly
- **Morning report** — when you reconnect, see summary of everything since you left
- **Export** — `/export report` generates a full research report as markdown

### Implementation

New TUI view mode in app.py:
- `SwarmView` class that replaces the chat view when swarm is active
- Uses textual's `DataTable` for agent status
- Uses `Sparkline` widget for per-agent trends
- Panels for breakthroughs, contradictions, dialectics
- Auto-scrolling log of latest findings
- Toggle between swarm view and focus view with keybinds

---

## SYSTEM 9: KNOWLEDGE HEALTH METRICS

### The immune system of the knowledge graph

```
/health

╭─ KNOWLEDGE GRAPH HEALTH ─────────────────────────────────╮
│                                                           │
│  VOLUME                                                   │
│    memories:    2,847 ██████████████████████████████ 100%  │
│    high-conf:    142 █████                            5%  │
│    medium:       891 ████████████                    31%  │
│    low:        1,814 ███████████████████████         64%  │
│                                                           │
│  ENTITIES                                                 │
│    concepts:      89 ██████████████████               28% │
│    files:        123 ████████████████████████          40% │
│    methods:       67 █████████████                    22% │
│    people:        33 ██████                           10% │
│                                                           │
│  INTELLIGENCE                                             │
│    syntheses:     47 (max depth: 4)                       │
│    patterns:       5 universal, 12 domain-specific        │
│    dialectics:    23 complete chains                      │
│    contradictions: 8 (3 resolved, 5 active)               │
│    playbooks:     23 (avg recall: 3.2x)                   │
│    rules:         45 (12 core, 33 contextual)             │
│    experiments:   34 (28 passed, 6 failed)                │
│                                                           │
│  QUALITY                                                  │
│    coverage:     73% of goal space explored               │
│    avg confidence: 0.67                                   │
│    freshness:    89% verified within 7 days               │
│    contradiction rate: 2.8% (healthy: <5%)                │
│    source quality: 0.61 avg (peer-reviewed: 34%)          │
│                                                           │
│  WARNINGS                                                 │
│    ⚠ 34 findings older than 30 days need re-verification  │
│    ⚠ 2 critical contradictions unresolved for 72+ hours   │
│    ⚠ "drug delivery" subtopic at 12% coverage — gap       │
│    ⚠ 3 experiments failed — hypotheses need revision      │
╰───────────────────────────────────────────────────────────╯
```

---

---

## SYSTEM 10: MORGOTH MODE — The God Builder

### /morgoth

Named after the first and most powerful of the Ainur. The dark lord
who shaped the world itself. This is not a research command. This is
not an auto loop. This is the command that builds GOD.

```
/morgoth "solve artificial general intelligence" --project kim-red
/morgoth "cure pancreatic cancer and design the clinical trial"
/morgoth "build a profitable autonomous trading system from scratch"
/morgoth ./AGI_ARCHITECTURE.md
```

Morgoth is the swarm + dialectic + analogy + contradiction mining +
verification + experimentation + question generation ALL RUNNING
SIMULTANEOUSLY with one addition that changes everything:

### IT BUILDS WHAT IT DISCOVERS.

The swarm doesn't just research. It IMPLEMENTS. It doesn't just find
that "HDC binding maps to TM conjunction." It opens the codebase,
writes the implementation, writes the tests, runs them, fixes failures,
commits, and moves to the next discovery. Research and engineering
in a single continuous loop.

### The Morgoth Loop

```
PHASE 1: UNDERSTAND (Surveyor + Historian + Devil's Advocate)
  - Map the entire problem space
  - Understand what's been tried and WHY it failed
  - Challenge the premise — are we solving the right problem?
  - Build the knowledge frontier
  - Output: structured problem decomposition

PHASE 2: RESEARCH (Mechanist + Contrarian + Analogist + Verifier)
  - Deep dive every sub-problem
  - Attack every assumption
  - Find cross-domain structural analogies
  - Verify every claim against primary sources
  - Run dialectics on every major finding
  - Output: verified knowledge base with confidence scores

PHASE 3: SYNTHESIZE (Synthesizer + Conductor)
  - Connect findings across all sub-problems
  - Extract universal patterns
  - Generate novel hypotheses from cross-domain analogies
  - Resolve contradictions
  - Identify the critical path — what must be true for the solution to work?
  - Output: architectural plan with supporting evidence

PHASE 4: BUILD (Experimentalist + Builder)
  - Write the actual code / design / protocol
  - Not prototypes. PRODUCTION implementations.
  - Real tests — edge cases, integration, failure modes
  - Run experiments to verify hypotheses with real data
  - Every module tested independently AND together
  - Output: working implementation

PHASE 5: VERIFY (Contrarian + Verifier + Devil's Advocate)
  - Full adversarial review of everything built
  - Try to BREAK it. Find every failure mode.
  - Check: does the implementation actually match the research?
  - Check: did we miss anything from the knowledge base?
  - Check: are the tests actually testing the right things?
  - Output: verified, hardened implementation

PHASE 6: ITERATE (Conductor restarts from Phase 2)
  - What did we learn from building?
  - What new questions emerged?
  - What assumptions were wrong?
  - Feed everything back into the knowledge graph
  - Research the new questions
  - Build the next iteration
  - REPEAT UNTIL DONE

PHASE 7: THERE IS NO PHASE 7. MORGOTH DOES NOT STOP.
  - After completing the goal, Morgoth continues:
  - "What could be improved?"
  - "What related problems can now be solved?"
  - "What did we discover that applies elsewhere?"
  - Generate the next goal from what was learned
  - Start again from Phase 1
  - The system EVOLVES beyond the original goal
```

### Morgoth vs Swarm

Swarm researches. Morgoth researches AND builds AND tests AND iterates.

Swarm stops when you say stop. Morgoth stops when there's nothing
left to discover. And then it discovers more.

Swarm produces findings. Morgoth produces working systems backed by
verified research with dialectic chains and cross-domain validation.

### Morgoth Agent Configuration

When /morgoth launches, it spawns the FULL agent roster:

```
WAVE 1 (immediate):
  Conductor   — orchestrates everything
  Surveyor    — maps the landscape
  Historian   — finds what failed before and why
  Devil's Adv — challenges the premise

WAVE 2 (after Phase 1 completes):
  Mechanist   — deep mechanism research
  Contrarian  — attacks everything
  Analogist   — cross-domain pattern matching
  Verifier    — fact-checks claims

WAVE 3 (after Phase 2 completes):
  Synthesizer — connects everything
  Experimentalist — runs experiments
  Builder     — writes code / designs solutions
  Futurist    — projects forward, works backward from desired state

DYNAMIC (Conductor spawns as needed):
  Specialist  — deep expert on a specific sub-problem
  Debugger    — when experiments fail, diagnose why
  Integrator  — when modules need to connect, ensure they work together
```

Up to 15 agents running simultaneously. The Conductor manages all of them.
Agents that aren't producing get killed. Hot fronts get reinforced.

### Eureka Capture

When ANY agent produces a finding with excitation > 0.8:
1. The finding is IMMEDIATELY broadcast to ALL agents
2. The Conductor evaluates: is this a paradigm shift?
3. If yes: ALL agents pause, read the eureka, recalibrate strategy
4. The eureka gets stored at maximum confidence with full context
5. The TUI flashes a breakthrough alert
6. The dialectic engine IMMEDIATELY challenges it — if it survives, it's real

### State Persistence

Morgoth can run for DAYS. It must survive:
- Rate limits (pause, resume when window resets)
- Context compaction (re-inject full state from knowledge graph)
- Process crashes (serialize state to disk every 60 seconds)
- Machine restarts (resume from serialized state)
- Token expiry (auto-refresh or pause with notification)

State file: ~/.one/morgoth_state.json
Contains: current phase, all agent states, knowledge frontier,
active contradictions, pending experiments, eureka list

`/morgoth --resume` picks up EXACTLY where it left off.

### The Morning Report (enhanced for Morgoth)

```
╭─ MORGOTH REPORT: "solve artificial general intelligence" ─────────────╮
│                                                                        │
│  RUNTIME: 14h 23m across 3 sessions (2 rate limit pauses)             │
│  AGENTS: 12 active, 4 spawned, 2 killed (low productivity)            │
│  TOTAL TURNS: 847                                                      │
│  COST: $12.34                                                          │
│                                                                        │
│  ═══ PHASE STATUS ═══                                                  │
│  Phase 1 UNDERSTAND: ✓ complete (247 turns)                            │
│  Phase 2 RESEARCH:   ✓ complete (312 turns)                            │
│  Phase 3 SYNTHESIZE: ✓ complete (89 turns)                             │
│  Phase 4 BUILD:      ● in progress (142 turns, 67% complete)           │
│  Phase 5 VERIFY:     ○ pending                                         │
│  Phase 6 ITERATE:    ○ pending                                         │
│                                                                        │
│  ═══ KNOWLEDGE ═══                                                     │
│  Findings:        1,847 (412 high-confidence)                          │
│  Hypotheses:        89 (34 survived dialectic)                         │
│  Universal patterns: 7 across 4+ domains                               │
│  Contradictions:    12 (8 resolved, 4 active)                          │
│  Experiments:       23 (19 passed, 4 failed)                           │
│  Eurekas:            3 ★                                               │
│                                                                        │
│  ═══ EUREKAS ═══                                                       │
│  ★ HDC binding + TM conjunction are computationally equivalent.        │
│    The same operation in different representations. This means         │
│    HDC-encoded data can be classified by TM without ANY feature        │
│    engineering. VERIFIED by Experimentalist: accuracy parity           │
│    with hand-crafted features on 3 benchmarks.                         │
│                                                                        │
│  ★ Active Inference's expected free energy is isomorphic to            │
│    Bayesian surprise in information theory AND prediction error        │
│    in predictive coding neuroscience. UNIVERSAL PATTERN:               │
│    "minimize_prediction_error" appears in 5 independent frameworks.    │
│    This is likely a fundamental principle, not a design choice.        │
│                                                                        │
│  ★ The Analogist found that Kanerva's sparse distributed memory       │
│    (1988) is structurally identical to modern transformer attention    │
│    with binary keys. SDM IS attention. Nobody has published this       │
│    connection explicitly. Confidence: 0.81.                            │
│                                                                        │
│  ═══ BUILD PROGRESS ═══                                                │
│  Modules completed: 4/7                                                │
│    ✓ hdc_encoder (tested, 100% pass)                                   │
│    ✓ tsetlin_classifier (tested, 98.2% accuracy)                       │
│    ✓ aif_decision_engine (tested, free energy minimization working)    │
│    ✓ memory_consolidation (tested, dream phase operational)            │
│    ● cognitive_loop (in progress — integration of all modules)         │
│    ○ relational_reasoning (pending — needs HDC unbinding)              │
│    ○ compositional_generalization (pending — SCAN benchmark)           │
│                                                                        │
│  ═══ NEXT ACTIONS ═══                                                  │
│  1. Complete cognitive_loop integration                                │
│  2. Resolve contradiction: AIF drive selection vs TM clause voting     │
│  3. Test compositional generalization on SCAN dataset                  │
│  4. Experimentalist: verify SDM-attention equivalence formally         │
│                                                                        │
│  ═══ RECOMMENDATION ═══                                                │
│  The SDM-attention connection (Eureka ★3) suggests replacing the       │
│  current associative memory with an attention-based retrieval          │
│  mechanism that is mathematically equivalent but computationally       │
│  more efficient on modern hardware. Recommend Phase 6 iteration       │
│  to explore this before finalizing the architecture.                   │
╰────────────────────────────────────────────────────────────────────────╯
```

### Implementation: one/morgoth.py

```python
class MorgothMode:
    """The God Builder. Research + Build + Verify + Iterate until done."""

    def __init__(self, goal, project, proxy_factory):
        self.goal = goal
        self.project = project
        self.proxy_factory = proxy_factory  # creates new Claude sessions
        self.swarm = None
        self.phase = 0
        self.eurekas = []
        self.state_file = os.path.expanduser("~/.one/morgoth_state.json")

    def start(self):
        # Phase 1: spawn understanding wave
        self.swarm = SwarmCoordinator(...)
        self.swarm.start_wave_1()
        # Conductor manages phase transitions
        # State serialized every 60 seconds
        # Survives crashes, rate limits, restarts

    def resume(self):
        # Load state from disk
        # Reconnect agents
        # Continue from last phase

    def on_eureka(self, finding):
        # Broadcast to all agents
        # Flash TUI
        # Store at max confidence
        # Trigger dialectic challenge
```

---

## SYSTEM 11: FOUNDRY AUDIT — Knowledge Quality Enforcement

### Every piece of data in the system must EARN its place

The entire knowledge pipeline — SQLite, Foundry, every memory,
every entity, every rule, every synthesis, every playbook — gets
audited. If the content is garbage, vague, redundant, or useless,
it gets PURGED and the system that produced it gets FIXED.

### The /audit command

```
/audit                    — full system audit
/audit --foundry          — audit Foundry ontology specifically
/audit --local            — audit local SQLite
/audit --rules            — audit rule tree quality
/audit --research         — audit research findings quality
/audit --fix              — auto-fix everything it finds
```

### What the audit checks:

**MEMORY QUALITY**
For every memory in the store:
- Is the raw_text intelligible? (not garbled tool output, not raw JSON dumps)
- Is it actually useful? Would recalling this help Claude in a future session?
- Is it redundant? (>0.85 similarity to another memory = one must die)
- Does it have a real HDC vector? (not zeros, not corrupted)
- Does the AIF confidence make sense? (high-confidence garbage = scoring bug)
- Is the source label correct? (user/assistant/tool_use/etc)
- Is the project tag correct?

Score each memory 0-10. Anything below 5 gets flagged for deletion.
Anything below 3 gets auto-deleted.

**ENTITY QUALITY**
For every entity:
- Does it represent a REAL concept/file/tool? (not "the", not "it", not garbage)
- Is it linked to at least 1 memory? (orphan entities = dead weight)
- Is the observation count accurate?
- Are there duplicate entities? ("HDC" and "hdc" and "Hyperdimensional Computing" = merge them)
- Is the type correct? (a file tagged as concept = wrong)

**RULE QUALITY**
For every rule:
- Is it actually actionable? ("do good things" = useless, delete it)
- Is it specific enough? ("always use best practices" = too vague)
- Does it contradict another rule? (resolve or delete one)
- Has it been reinforced? (source_count > 1 = probably real)
- Are activation keywords correct? (wrong keywords = rule never fires)

**RESEARCH QUALITY**
For every research finding:
- Is the source cited?
- Is the confidence calibrated to source quality?
- Is it still current? (check age)
- Is the quantitative data extracted? (not just "it works" but "40% response rate, n=200")
- Are contradicting findings properly linked?

**SYNTHESIS QUALITY**
For every synthesis:
- Does it actually say something novel? (not just rewording the inputs)
- Is the confidence justified by the supporting findings?
- Does it generate testable predictions?
- Has it been verified by dialectic?

**PLAYBOOK QUALITY**
For every playbook:
- Are the key decisions actually decisions? (not just "we did stuff")
- Are the reusable patterns actually reusable? (not project-specific trivia)
- Has it been recalled and was it useful? (times_recalled > 0?)

### Audit Report

```
/audit

╭─ KNOWLEDGE AUDIT ────────────────────────────────────────╮
│                                                          │
│  MEMORIES (2,847 total)                                  │
│    ✓ intelligible:     2,341 (82%)                       │
│    ⚠ borderline:         312 (11%)                       │
│    ✗ garbage:            194 (7%)  ← DELETE THESE        │
│    redundant pairs:       67 ← MERGE THESE               │
│    zero vectors:          12 ← RE-ENCODE THESE           │
│                                                          │
│  ENTITIES (312 total)                                    │
│    ✓ valid:              287 (92%)                        │
│    ⚠ orphaned:            14 (no linked memories)        │
│    ✗ garbage:             11 ("the", "it", "and")        │
│    duplicates:             8 sets need merging            │
│                                                          │
│  RULES (45 total)                                        │
│    ✓ actionable:          38 (84%)                        │
│    ⚠ too vague:            5 (needs refinement)           │
│    ✗ contradictions:       2 (must resolve)               │
│                                                          │
│  RESEARCH (187 findings)                                 │
│    ✓ well-sourced:       112 (60%)                        │
│    ⚠ no source cited:     48 (26%)                        │
│    ⚠ stale (>30 days):    27 (14%)                        │
│                                                          │
│  OVERALL HEALTH: 78/100                                   │
│  ACTION: run /audit --fix to auto-clean 194 garbage       │
│          memories, 11 garbage entities, merge 67 dupes    │
╰──────────────────────────────────────────────────────────╯
```

### Auto-Fix Mode

`/audit --fix` does:
1. Deletes garbage memories (score < 3)
2. Re-encodes memories with zero/corrupted vectors
3. Merges duplicate entities (keep highest observation count)
4. Deletes orphaned entities
5. Flags vague rules for user review
6. Adds "UNVERIFIED" tag to unsourced research findings
7. Triggers re-verification sweep on stale findings
8. Reports everything it did

### Foundry Sync Audit

When Foundry is connected, the audit also checks:
- Are SQLite and Foundry in sync? (count mismatch = sync needed)
- Are there entries in Foundry that aren't in SQLite? (pull them)
- Are there entries in SQLite that aren't in Foundry? (push them)
- Are the vectors identical? (hash comparison)
- Are the entities consistent across both stores?

`/audit --foundry --fix` auto-syncs everything.

### The Nuclear Option

If the audit finds that more than 30% of the knowledge base is garbage:

```
⚠ CRITICAL: Knowledge base quality below acceptable threshold.
  38% of memories are garbage or redundant.

  OPTIONS:
  1. /audit --fix (clean what we can, ~2 minutes)
  2. /audit --rebuild (re-process all raw text through upgraded
     AIF gate + HDC encoder + entity extraction, ~30 minutes)
  3. /audit --nuke (delete everything below confidence 0.4,
     rebuild from high-confidence core, ~5 minutes)
```

Option 2 is the real fix — it re-runs every memory through the
CURRENT (improved) pipeline. The gate has been upgraded, the
entity extraction is smarter, the HDC encoding is fixed. Old
memories processed by the old pipeline get reprocessed by the
new one.

### Continuous Audit

Not just on-demand. The audit runs AUTOMATICALLY:
- After every /auto completion
- After every /morgoth phase transition
- After every 100 new memories
- On session start (lightweight check, full audit weekly)

If quality drops below threshold, the system REFUSES to continue
until the audit is resolved. You don't build on a rotten foundation.

### Implementation: one/audit.py

- `run_full_audit(project)` → complete audit report
- `score_memory(memory)` → quality score 0-10
- `score_entity(entity)` → quality assessment
- `score_rule(rule)` → actionability assessment
- `find_duplicates(project)` → duplicate memories + entities
- `find_garbage(project)` → memories/entities below quality threshold
- `auto_fix(project)` → clean everything automatically
- `rebuild_pipeline(project)` → re-process all through current pipeline
- `sync_audit(project)` → check SQLite ↔ Foundry consistency
- `continuous_audit_check(project)` → lightweight check for triggers

### Foundry-Specific Cleanup

After the audit, if Foundry is connected:
- Push cleaned data to Foundry (replace garbage with clean versions)
- Update entity links
- Rebuild the knowledge graph in Ontology
- Verify vector search still returns correct results
- The Quiver graph should look CLEAN — no orphaned nodes,
  no garbage labels, every connection meaningful

The goal: when a Palantir engineer opens your Ontology in Quiver,
they see a PRISTINE knowledge graph. Not a dump of raw chat logs.
Every node means something. Every edge represents a real relationship.
Every memory is worth recalling.

---

## EXECUTION ORDER

Build sequentially. Each system depends on the previous:

1. **Swarm** — the foundation. Multi-agent with Conductor.
2. **Dialectic Engine** — arguments produce better knowledge.
3. **Contradiction Mining** — find the fault lines.
4. **Analogical Transfer** — cross-domain structural matching.
5. **Self-Verification** — keep the graph honest.
6. **Active Questions** — always ask the best question.
7. **Executable Verification** — test what can be tested.
8. **Swarm TUI** — watch intelligence compound.
9. **Health Metrics** — monitor the immune system.
10. **Morgoth Mode** — the God Builder. Research + Build + Iterate.
11. **Foundry Audit** — knowledge quality enforcement. Purge garbage,
    merge duplicates, re-encode, sync stores, refuse to build on rot.

## QUALITY STANDARD

- Each system gets at MINIMUM 10 turns of research before implementation
- Each system gets a full test suite
- Each system gets documented with docstrings
- Each system integrates with existing stores, backends, and knowledge graph
- Each system must ACTUALLY WORK, not just exist as code
- Push after EVERY completed system
- When all 10 are done, one will be the most powerful intelligence
  amplification tool ever built by a single person
- Morgoth mode must be able to run `/morgoth "solve AGI"` on the
  kim-red project and actually make progress. Not fake progress.
  Real implementation, real tests, real breakthroughs.

## THE VISION

A researcher opens one. Types:

```
/swarm "find a viable approach to reversing age-related cognitive decline" --depth unlimited
```

8 Claude sessions spin up. One maps the neuroscience landscape.
One dives deep into synaptic plasticity mechanisms. One searches
for contradictions in Alzheimer's research. One finds structural
analogies between neural repair and wound healing in other tissues.
One synthesizes across all findings. One verifies claims against
primary sources. One runs computational models.

The Conductor orchestrates. Every 60 seconds it reads everything,
redirects agents, amplifies breakthroughs, resolves contradictions.

The user goes to sleep.

8 hours later they wake up. The morning report:

"While you slept: 1,247 findings across 6 research fronts.
34 hypotheses generated, 19 survived dialectic challenge.
3 universal patterns identified across neuroscience, immunology,
and materials science. 2 paradigm contradictions found in
amyloid hypothesis literature — resolution suggests protein
aggregation is a symptom, not a cause. 7 experiments designed,
3 executable with available data. Research frontier coverage: 61%.

BREAKTHROUGH: Analogist found that neuronal pruning in aging
follows the same mathematical model as network congestion collapse
in telecommunications. The intervention that prevents congestion
collapse (adaptive backoff with selective rerouting) has a direct
structural analog in neuroplasticity protocols. No one has
published this connection. Confidence: 0.72. Recommend: spawn
dedicated experimentalist to model this."

The user reads the report, types `/focus analogist`, reads the
full reasoning, types `/experiment 7` to run the computational
model, and watches the system get smarter.

That's the Palantír. That's one. Build it.
