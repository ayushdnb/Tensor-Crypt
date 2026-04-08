# The Field Guide to Tensor Crypt

This file is the intuitive world-manual for Tensor Crypt. It explains what the simulation **feels like from inside the world** and what an observer can reliably see from outside it. The goal is not to teach every internal detail. The goal is to give you a stable mental model before later documents move into architecture, math, and implementation contracts.

## What this file teaches

This file explains:

- what kind of world Tensor Crypt simulates
- what occupies the board
- what an agent can sense and do
- what a tick means in practical terms
- how health, damage, zones, metabolism, and death shape survival
- how births, parent roles, mutation, and rare mutation change lineages
- how catastrophes temporarily change the world
- what the viewer helps an operator notice
- which configuration groups most strongly change the feel of a run

## What this file deliberately leaves for later

This file does **not** try to teach:

- PPO internals or derivations
- policy/value network architecture in depth
- optimizer state, checkpoint manifest structure, or validation harnesses
- package/module architecture mapping
- full configuration reference coverage
- migration / schema / compatibility details

> **Reading posture**
>
> If you only want to understand what you are looking at in a live run, this file is enough.
> If you want to change the system safely, you will need the later documents.

## What kind of world this is

Tensor Crypt is a **grid-based multi-agent survival and reproduction simulation**.

The world is a rectangular board made of cells. Agents move across that board, sense it through directional rays, gain or lose health from the terrain, collide with one another, die, and sometimes get replaced by offspring. The simulation does not run as a loose narrative. It runs as a strict repeated cycle of **ticks**, and each tick updates the same shared world.

At the world level, three things matter immediately:

1. **Space matters.** Position, crowding, walls, and zone placement change outcomes.
2. **Life is expensive.** Agents constantly face damage pressure, terrain pressure, and metabolic cost.
3. **Population is regulated.** Death does not simply reduce the world forever. Birth logic can refill empty slots when the run allows it.

A helpful mental image is this:

```text
Tensor Crypt world
────────────────────────────────────────────────────────
Board geometry      → where walls and open cells exist
Field pressure      → where healing or harmful zones act
Live population     → which agents occupy which cells
Lineage pressure    → which bloodlines persist, mutate, or disappear
Shock layer         → temporary catastrophes that reshape conditions
────────────────────────────────────────────────────────
```

### The board is not just empty space

The grid has a few persistent layers:

| Layer | Plain-English meaning |
|---|---|
| Walls | Solid blocked cells, including border walls around the map |
| Zone field | A per-cell healing or harming rate |
| Agent occupancy | Which live agent currently occupies a cell |
| Agent mass imprint | The mass value stored with the occupying agent |

The board therefore acts like both a map and a pressure field. Two agents can stand on two nearby cells but still be living in very different local conditions.

## What exists in the world

Several things coexist on the board at the same time.

### 1. Walls

Walls are physical blockers. The world always starts with border walls, and procedural generation can carve additional internal wall segments. These change pathing, crowding, and collision risk.

Walls matter for two separate reasons:

- they block movement and ray sight lines
- colliding with them causes damage

### 2. Healing and harmful zones

The world carries a **zone-rate field**. A cell can contribute positive or negative health pressure.

At baseline, procedural generation creates rectangular healing zones. During a run, that field can become more complicated because:

- zones can overlap according to a configured overlap rule
- selected zones can be weakened, inverted, or erased by catastrophes
- new harmful patches can appear during catastrophes
- an operator can manually edit a selected zone's rate in the viewer

So the world does not just have “terrain.” It has **terrain with health consequences**.

### 3. Agents

An agent is a live unit on the board with:

- a position
- current health and maximum health
- mass
- vision range
- metabolism rate
- a bloodline family
- a policy/value brain
- lineage history such as parent roles and generation depth

In practical terms, an agent is a moving entity that senses, chooses one of nine actions, experiences environmental pressure, and may later become a parent in a future birth.

### 4. Bloodlines

Bloodlines are visible families of agents. They matter both visually and behaviorally.

They are visible because the viewer colors agents by bloodline and keeps family counts. They are behaviorally relevant because a child normally inherits the **brain parent's family**, unless a family-shift mutation is enabled and actually occurs.

So a bloodline is not only a cosmetic label. It is part of lineage continuity.

### 5. Temporary catastrophes

Catastrophes are bounded world shocks. They do not permanently rewrite the baseline map. Instead, they sit on top of the normal world for a limited time and temporarily alter:

- zone pressure
- movement danger
- sensing range
- metabolic burden
- mutation intensity
- whether reproduction is allowed

They are best understood as **temporary regimes** rather than permanent map edits.

## What an agent experiences

An agent does not “know” the whole simulation. It receives a structured local-and-global observation.

### Directional sensing

Each live agent casts a fixed number of rays outward in evenly spaced directions. Those rays can report, in normalized form:

- no hit
- another agent
- a wall
- how far away the first hit was
- the strongest zone pressure encountered along the path
- the terminal zone pressure at the ray end or hit location
- the mass of a hit agent
- the health ratio of a hit agent

Intuitively: the agent gets a **circular directional scan** of nearby structure and pressure.

### Self-state

An agent also receives information about itself, including:

- current health ratio
- current health deficit ratio
- normalized mass
- normalized maximum health
- normalized vision
- normalized metabolism
- normalized position
- distance from the board center
- normalized age
- the zone pressure of its current tile

So the agent is not blind to its own condition. It has a self-report alongside the outward scan.

### Global context

The observation also includes small context signals such as:

- fraction of total slots currently alive
- mean mass of the live population
- mean health ratio of the live population

That means the agent is not operating from pure local sensation alone. It also receives a coarse summary of the broader world state.

> **Where intuition stops**
>
> This file treats observation as a sensing surface. Later files explain the exact schema, normalization contract, and how the brain consumes it.

## What happens during one tick

A tick is one full world-update cycle. The easiest way to understand Tensor Crypt is to learn that cycle.

### Tick storyboard

```text
1. Existing catastrophes expire, and new ones may begin.
2. The baseline zone field is repainted.
3. Active catastrophes apply temporary overrides.
4. Live agents build observations from the current world.
5. Each live agent samples one action.
6. Physics resolves movement, wall hits, rams, and contested cells.
7. Zone effects and metabolism change health.
8. Deaths are detected and finalized.
9. Respawn logic may place offspring into freed slots.
10. Logs, summaries, and learning-facing buffers are updated.
```

### A more concrete narration

On each tick:

- catastrophe timing is checked first
- the world's zone field is rebuilt from its baseline zones
- active catastrophes then layer temporary world changes on top
- each live agent senses the resulting world
- each live agent chooses one of the available actions
- the physics layer resolves what can actually happen
- health changes from collisions, zones, and metabolism are applied
- any agents whose health has reached zero are retired
- the reproduction controller may fill dead slots with births
- telemetry and training state are updated from the tick that just happened

### Empty-world ticks still matter

If no agents are alive, the simulation does not become meaningless. The tick still advances through catastrophe status, logging, and respawn/extinction handling. This matters because a run can recover from near-collapse if its extinction policy permits it.

## How agents survive, suffer, and die

Survival pressure in Tensor Crypt comes from several places at once.

### Movement is not free of consequence

Agents choose from **nine actions**:

- one no-move action
- eight neighbor directions

That means movement is discrete and local. There is no long continuous glide. Every tick is a new directional commitment.

### Wall collisions hurt

If an agent moves into a wall, it does not pass through. It takes damage scaled by its mass and the wall penalty coefficient.

Heavier bodies therefore hit harder and get punished harder.

### Ramming hurts both sides

If a moving agent tries to enter a cell occupied by an agent that is staying put, or effectively not vacating it in time, the event is treated as a ram:

- the rammer takes ram damage
- the target takes idle-hit damage

This produces a world where crowding and aggressive movement can be costly even without full contests.

### Contests resolve who gets the cell

If multiple active contenders aim for the same destination cell, the cell becomes contested. The winner is determined by **strength**, where strength is based on:

- mass
- current health ratio

The winner moves into the contested cell. The others do not. But even the winner still takes damage, and the losers take more.

So winning a contest is not a free reward. It is a costly victory.

### The terrain changes health every tick

After movement resolution, the environment applies the current zone field:

- positive field values increase health
- negative field values decrease health

Positive gains are also tracked in a separate way that later matters for death-time fitness carryover.

### Metabolism is always active

Every living agent also loses health to metabolism. Metabolism is part of simply existing. Some catastrophes can increase that burden further, and one catastrophe adds an additional burden tied to mass.

This makes the world fundamentally hostile to passive indefinite survival.

### Death is a threshold, not a dramatic special mode

After collisions, zone effects, and metabolism, health is clamped into the legal range. Any live agent whose health has reached zero is marked dead and removed from the board.

Death reasons are tracked for diagnostics, including cases such as:

- wall collision
- ram damage
- contest damage
- poison zone exposure
- metabolism death

The important beginner intuition is simpler:

> An agent dies when the world pressure of that tick reduces its health to zero.

### Death also affects lineage and future selection

Death is not only removal. It is also a bookkeeping moment.

When a slot dies:

- its PPO-owned runtime state is cleared
- its UID is finalized as historical
- its stored fitness is updated using past fitness decay plus accumulated positive zone gain
- the slot becomes available for possible reuse by a future birth

That means death closes one life and prepares the substrate for the next.

### Agent life-cycle diagram

```text
spawned
  ↓
alive on board
  ↓
senses → acts → takes pressure → survives or weakens
  ↓
may become useful as:
  • brain parent
  • trait parent
  • anchor parent
  ↓
health reaches zero
  ↓
removed from board and finalized
  ↓
slot may later host a different child
```

## How new agents appear

New agents do not appear continuously at random. They appear through a **respawn controller** that watches population state and timing.

### When births are allowed

Births are considered only when all of the following make room for them:

- reproduction is currently enabled
- live population is below the population ceiling
- there are dead, reusable slots
- the run is either below the population floor or has reached the respawn period

This creates a deliberately regulated recovery pattern rather than constant unbounded spawning.

### Population floor and ceiling

The two most important intuitive thresholds are:

| Threshold | Meaning |
|---|---|
| Population floor | If live population drops below this, recovery becomes more urgent |
| Population ceiling | No births are added once population is at or above this size |

The floor is a recovery trigger. The ceiling is a hard stop.

### Binary parented reproduction

The active reproduction mode is binary parented. In plain language, that means a child is produced from **two living contributors plus one placement anchor role**.

The system keeps these roles separate on purpose.

#### Brain parent

The brain parent provides the child’s model lineage.

In the ordinary case, if the child stays in the same family, the child’s brain is copied from the brain parent and then perturbed with policy noise. This is the clearest sense in which “strategy lineage” persists.

#### Trait parent

The trait parent provides the latent trait template that is mutated into the child’s realized traits:

- mass
- maximum health
- vision
- metabolism

So the trait parent is the donor of the child’s biological-style profile.

#### Anchor parent

The anchor parent decides **where** the system first tries to place the child. The child is not spawned on top of the anchor. Instead, the controller searches nearby cells around that anchor.

The anchor can be configured to be:

- the brain parent
- the trait parent
- either one at random
- the fitter of the two

### Nearby placement first, fallback placement second

Birth placement is intentionally local-first.

The controller:

1. searches outward in shuffled square rings around the anchor parent
2. checks whether each candidate tile is acceptable
3. optionally falls back to a global free-cell search if local placement fails

A tile can be rejected for several reasons, depending on config:

- it is a wall
- it is already occupied
- it is in a harmful zone

So “a birth failed” does not necessarily mean reproduction logic broke. It often means the world did not offer a legal cell.

### Floor recovery

When the population is below the floor, parent selection becomes more permissive if floor-recovery threshold suspension is enabled. In practice, this means the controller relaxes normal eligibility thresholds so the run can recover more aggressively.

### Extinction handling

If live population drops below two agents, normal binary reproduction is impossible. At that point the extinction policy decides what happens.

Possible outcomes include:

- the run fails immediately
- bootstrap agents are spawned from a default latent template
- bootstrap agents are spawned under an administrative default policy

This is the system’s answer to total or near-total collapse.

### Birth / inheritance diagram

```text
living population
   │
   ├── select brain parent  → strategy lineage
   ├── select trait parent  → trait lineage
   └── select anchor parent → placement reference
                │
                ↓
      try nearby legal cells first
                │
        local placement succeeds?
          ├── yes → place child nearby
          └── no  → optional global fallback
                │
                ↓
       instantiate child with:
       - brain family
       - mutated traits
       - birth HP rule
       - new UID / new life
```

## How mutation changes lineages

Mutation is the main source of controlled novelty during reproduction.

### Ordinary mutation

On a normal birth, the trait-parent latent values are copied and then perturbed by small random changes. Those changes alter how the child’s trait budget is allocated across:

- maximum health
- mass
- vision
- metabolism

Separately, the child brain also receives policy noise.

So a child is usually **similar but not identical** to its lineage sources.

### Rare mutation

A small rare-mutation probability exists. When that rare path triggers, the system uses much larger mutation strengths than usual.

In practice, that means rare mutation is the “occasionally disruptive” path. It is not the common background drift. It is the sharper break from local family history.

### Family-shift mutation

Family shift is optional and disabled by default. When enabled, a birth can very rarely switch into a different bloodline family instead of remaining in the brain parent’s family.

Conceptually, this matters because the child is no longer simply “another member of the same house.” The lineage may branch into another family identity.

### Lineage is split, not monolithic

One of the most important ideas in this repository is that inheritance is deliberately split:

- strategy can come from one parent
- traits can come from another
- placement reference can come from either

This is why the birth system is more precise than a single “one parent makes one child” story.

> **Good beginner summary**
>
> Mutation changes what the child is like.
> Parent roles change which part of the child came from where.

## How catastrophes reshape the world temporarily

Catastrophes are timed shocks managed by a dedicated scheduler. They are **temporary bounded events**, not permanent map rewrites.

Two design facts matter most:

1. baseline zones are repainted every tick
2. catastrophe effects are reapplied on top of that baseline while the event is active

So when a catastrophe ends, the world falls back to its ordinary underlying field unless another catastrophe is still active.

### Scheduler modes

The catastrophe manager supports four overall modes:

| Mode | Meaning |
|---|---|
| `off` | no catastrophe activity |
| `manual_only` | only operator-triggered catastrophes |
| `auto_dynamic` | weighted random auto scheduling with random gaps |
| `auto_static` | interval-based auto scheduling with ordered type selection |

It can also be configured to pause the scheduler, allow or disallow overlap, and cap concurrent active catastrophes.

### Catastrophe summary table

| Catastrophe | Plain-English effect | Pressure added |
|---|---|---|
| Ashfall of Nocthar | Removes healing from a selected share of positive zones | healing collapse |
| Sanguine Bloom | Turns selected positive zones into harmful zones | localized poison |
| The Woundtide | Sweeps a damaging front across the map | traveling hazard band |
| The Hollow Fast | Reduces the strength of positive zones | healing drought |
| Mirror of Thorns | Inverts selected zone regions | zone reversal / trap field |
| Veil of Somnyr | Shrinks effective vision while active | sensory fog |
| Graveweight | Raises metabolic burden and penalizes heavier bodies more | survival tax |
| Glass Requiem | Makes collision damage harsher | contact danger |
| The Witchstorm | Increases mutation intensity and mutation-related probabilities | lineage instability |
| The Thorn March | Pushes a harmful border inward and shrinks the safe interior | space compression |
| The Barren Hymn | Disables reproduction while active | recovery shutdown |
| Crimson Deluge | Creates several harmful patches on the board | patchy lethal terrain |

### What catastrophes can change

Across the active catastrophe roster, the system can temporarily change:

- which cells heal and which cells harm
- how much healing remains available
- whether vision is shortened
- how punishing collisions are
- how strong metabolism pressure becomes
- whether births are allowed
- how aggressive mutation becomes

That means catastrophes do not all “do damage” in the same way. Some attack sensing. Some attack terrain. Some attack lineage stability. Some attack recovery.

### A good mental model

Think of a catastrophe as a **temporary rule regime** layered onto the same board.

The board is still the same board.
The agents are still the same agents.
But for a bounded window, the board behaves under harsher or stranger local rules.

## What an observer can learn from the viewer

The viewer is the operator’s main intuition surface. It does not merely animate the run. It exposes the state in a way that makes patterns legible.

### What the world view shows well

The main world view lets an observer see:

- walls and open corridors
- healing versus harmful tiles
- agent positions
- agent bloodline colors
- low-health visual darkening within bloodline coloring
- HP bars, when enabled
- selected-agent rays, when enabled
- catastrophe overlays, when enabled

This is enough to understand whether the run is sparse, crowded, choking, recovering, or currently under shock.

### What the HUD gives you quickly

The HUD summarizes high-level state such as:

- tick number
- pause / speed state
- live population versus slot capacity
- alive counts by bloodline
- catastrophe mode, active names, and next scheduled event

This is the fast “how is the world doing right now?” surface.

### What the inspector gives you

When an agent is selected, the side inspector can show:

- slot and UID
- bloodline
- age and birth tick
- generation depth
- brain / trait / anchor parent UIDs
- health
- position
- mass, vision, metabolism
- parameter count of the selected brain

When enrichment is enabled, it can also show trait-budget allocation details, PPO counters, and catastrophe exposure summaries.

When a zone is selected, the inspector shows its bounds and rate.

### What the operator is usually looking for

A careful observer often looks for patterns like:

- which bloodlines are holding territory or collapsing
- whether healing zones are being used or inverted
- whether one region has become too punishing
- whether births are keeping up with deaths
- whether catastrophes are narrowing safe space
- whether a selected agent’s parent roles and visible condition explain its behavior

### The viewer is also an intervention surface

The viewer is not only passive.

It supports operator actions such as:

- pausing and single-stepping
- changing speed
- toggling rays, HP bars, zones, and grid lines
- selecting agents and zones
- editing the rate of a selected zone
- triggering, clearing, and managing catastrophes

This is why it is best understood as an **operator console**, not only a renderer.

## Which configuration groups change the feel of the simulation

This file is not the full config manual, but several groups strongly alter the world experience.

### World size and density

These are the knobs that most obviously change spatial feel:

- `GRID.W`, `GRID.H`
- `MAPGEN.RANDOM_WALLS`
- `MAPGEN.WALL_SEG_MIN`, `MAPGEN.WALL_SEG_MAX`
- `MAPGEN.HEAL_ZONE_COUNT`
- `MAPGEN.HEAL_ZONE_SIZE_RATIO`
- `GRID.HZ_OVERLAP_MODE`

Increase board size and you usually reduce encounter density.
Increase walls and you create chokepoints.
Increase zone count or size and terrain pressure becomes more central.

### Population pressure and recovery behavior

These decide how forgiving or brittle the population dynamics feel:

- `AGENTS.N`
- `RESPAWN.RESPAWN_PERIOD`
- `RESPAWN.MAX_SPAWNS_PER_CYCLE`
- `RESPAWN.POPULATION_FLOOR`
- `RESPAWN.POPULATION_CEILING`
- `RESPAWN.EXTINCTION_POLICY`
- `RESPAWN.EXTINCTION_BOOTSTRAP_SPAWNS`

Lower respawn delay and the world recovers faster.
Raise the ceiling and runs can stay denser.
Choose a permissive extinction policy and collapse may become reversible.

### Survival pressure

These make ordinary life harsher or milder:

- `MAPGEN.HEAL_RATE`
- `PHYS.K_WALL_PENALTY`
- `PHYS.K_RAM_PENALTY`
- `PHYS.K_IDLE_HIT_PENALTY`
- `PHYS.K_WINNER_DAMAGE`
- `PHYS.K_LOSER_DAMAGE`
- `TRAITS.METAB_COEFFS`

If you change these, the world can shift from exploratory to brutally attritional very quickly.

### Perception feel

These change how much local structure an agent can use:

- `PERCEPT.NUM_RAYS`
- trait clamp ranges, especially vision bounds
- catastrophe settings such as `veil_of_somnyr`

More rays and more vision change the informational geometry of the run.

### Mutation pressure

These change how conservative or volatile births become:

- `EVOL.POLICY_NOISE_SD`
- `EVOL.TRAIT_LOGIT_MUTATION_SIGMA`
- `EVOL.TRAIT_BUDGET_MUTATION_SIGMA`
- `EVOL.RARE_MUT_PROB`
- rare-mutation sigma knobs
- `EVOL.ENABLE_FAMILY_SHIFT_MUTATION`
- `EVOL.FAMILY_SHIFT_PROB`

This group changes whether lineages drift slowly or break sharply.

### Catastrophe behavior

These control how often shocks arrive and how severe they feel:

- `CATASTROPHE.DEFAULT_MODE`
- dynamic or static scheduling gaps
- per-type enable table
- per-type durations
- per-type weights
- `CATASTROPHE.TYPE_PARAMS`
- overlap and concurrency settings

This group decides whether catastrophes are occasional interruptions or a constant strategic fact of life.

### Viewer visibility

These do not change simulation mechanics, but they strongly change what a human can understand during a run:

- `VIEW.SHOW_OVERLAYS`
- `VIEW.SHOW_BLOODLINE_LEGEND`
- `VIEW.SHOW_CATASTROPHE_PANEL`
- `VIEW.SHOW_CATASTROPHE_OVERLAY`
- `VIEW.SHOW_CATASTROPHE_STATUS_IN_HUD`
- `VIEW.FPS`

In the current viewer, `VIEW.SHOW_OVERLAYS` only seeds startup state for `h_rate` and `rays`; it is not a general overlay registry.

Runs can feel “opaque” or “legible” to an operator even when their underlying mechanics are identical.

## Common beginner questions

### Why did the population suddenly rise or fall?

Because births are regulated, not continuous. Population can drop quickly from collisions, poison pressure, metabolism, or catastrophes, and then rise only when the respawn controller is allowed to refill dead slots under the floor/ceiling/timer rules.

### Why do some agents seem more resilient?

Because resilience is not one number. Mass, maximum health, current health, vision, metabolism, local zone pressure, and current catastrophe conditions all contribute to how long an agent lasts.

### Why did a birth fail or appear somewhere else?

Birth placement tries nearby legal cells around the anchor parent first. If those are blocked or unsafe, the controller may fall back to a global free-cell search. If that also fails, the birth can fail altogether.

### Why does the world suddenly become harsher?

Because a catastrophe may have started, or because the local zone field, crowding, and collision pattern shifted into a worse regime. Harshness in Tensor Crypt is often spatially local before it is globally obvious.

### Why does the same run feel different after configuration changes?

Because configuration changes alter the world itself: map size, wall density, zone overlap, respawn cadence, mutation strength, catastrophe intensity, and viewer visibility all change the practical behavior of the same underlying simulation loop.

### Are rewards the same thing as evolutionary fitness here?

No. The PPO-facing reward is currently based on the agent’s health ratio, optionally with a gate. The stored evolutionary fitness used in parent selection is updated at death from accumulated positive zone gain plus fitness decay. They are related to survival, but they are not the same signal.

### What should I read next if I want the real internal mechanics?

Read the next document in the suite, which moves from world intuition into the deeper mechanics and internal runtime structure.

## End-of-file recap

Tensor Crypt is a grid-based world where agents live under constant pressure.

They sense with rays, choose one of nine actions, collide, heal, suffer, burn health to metabolism, and die when pressure becomes too great. Dead slots can later be reused by births, but births are regulated by explicit recovery rules rather than being continuous. Each child can inherit strategy, traits, and placement reference from different parent roles. Mutation keeps lineages moving, and rare mutation can break them more sharply. Catastrophes temporarily rewrite the practical conditions of the world without permanently replacing the underlying map. The viewer turns all of this into something an operator can actually inspect, compare, and reason about.

If you can now mentally narrate a tick, a death, a birth, and a catastrophe window, this file has done its job.

## Read next

Continue to the next document in the technical document set for the deeper mechanics: internal runtime flow, subsystem responsibilities, and the stricter technical picture that sits underneath this field guide.
