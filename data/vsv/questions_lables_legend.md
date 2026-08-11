# Objective legend for the diagnostic S, T, and C labels

This legend defines a fixed, evidence-based three-level complexity scale for
spatial, temporal, and causal questions.

The scale evaluates the **minimum operations and evidence required by the
question**, rather than assigning a subjective estimate of the difficulty of
the corresponding video.

The same 0–2 progression is applied independently to the spatial, temporal,
and causal dimensions.

---

# Shared 0–2 complexity scale

## Level 0 — Direct retrieval

The answer can be obtained by retrieving one directly available piece of
evidence.

Objective criteria:

- exactly one relevant evidence unit is sufficient;
- no comparison between different evidence units is required;
- no temporal alignment between distinct events is required;
- no reconstruction of a change of state is required;
- no implicit inference is required.

Typical evidence scope:

`single_frame` or one directly observable event/state.

This level corresponds to the previous Level 0 and remains unchanged.

---

## Level 1 — Explicit relational integration

The answer requires establishing an explicit relation, alignment, or
dependency involving one or two evidence units.

A question is assigned Level 1 if its solution requires at least one of the
following operations:

- identifying one explicit relation between entities;
- conditioning the retrieval of information on one explicitly identifiable
  event or phase;
- comparing or aligning two explicit evidence units;
- ordering two explicitly observable events;
- connecting one explicit antecedent, action, instruction, or goal to one
  explicit consequence.

Objective constraints:

- the relevant evidence is directly observable;
- at most two evidence units need to be integrated;
- no implicit causal, intentional, or social inference is necessary;
- no multi-event state-transition reconstruction is necessary.

Typical evidence scope:

`local_interval` or `two_intervals`.

This level merges the previous Levels 1 and 2.

---

## Level 2 — Multi-step, implicit, or state-transition integration

The answer requires integrating information beyond a simple explicit
pairwise relation.

A question is assigned Level 2 if **at least one** of the following conditions
holds:

- evidence must be integrated across more than two relevant units or events;
- the question requires reconstructing a sequence of states or events;
- the answer depends on a change of state;
- the answer requires tracking how an entity or relation changes over time;
- duration or repeated occurrences must be accumulated or counted;
- at least one relevant causal, intentional, social, or contextual relation
  is not explicitly stated and must be inferred;
- solving the question requires combining multiple operations sequentially.

Typical evidence scope:

`multi_event_sequence`.

This level merges the previous Levels 3 and 4.

---

# Operational decision rule

The following decision procedure should be applied in order.

1. **Can the answer be obtained from one directly available evidence unit,
   without comparison, alignment, or inference?**

   Yes → Level 0.

2. **Otherwise, can the answer be obtained by integrating at most two explicit
   evidence units through one observable relation, comparison, temporal
   alignment, or explicit cause–effect connection?**

   Yes → Level 1.

3. **Otherwise**, if the question requires multi-event integration,
   state-transition reconstruction, duration/counting, tracking across phases,
   or an implicit inferential step:

   → Level 2.

The highest required operation determines the complexity level.

---

# S labels — Spatial reasoning

## S0 — Direct spatial retrieval

Retrieve the location of an entity from one frame or stable scene state.

**Objective criteria:**

- one entity or target spatial property;
- one evidence unit;
- no event needs to be used to condition the retrieval;
- no comparison between different spatial states is required.

**Evidence scope:** `single_frame`

**Typical question:**  
“Where is the object?”

---

## S1 — Explicit or event-conditioned spatial relation

Retrieve a spatial relation that is directly observable either within one
action interval or after aligning the spatial information with one explicitly
identifiable event.

This category includes both local spatial relations and event-conditioned
spatial retrieval.

**Objective criteria:**

At least one of the following applies:

- retrieve one explicit source, destination, direction, support surface,
  body part, containment relation, or placement relation;
- identify one event and retrieve the spatial configuration associated with
  that event;
- compare at most two explicitly observable spatial evidence units.

The spatial relation itself must remain directly observable.

**Maximum evidence units:** 2

**Evidence scope:** `local_interval` or `two_intervals`

**Possible auxiliary ability:** `T0` or `T1`

**Typical questions:**

“Where does the person put the object?”

“Where is the child when she starts crying?”

---

## S2 — Spatial reconstruction across phases or state transitions

Reconstruct a spatial configuration by integrating information across
multiple temporal phases, events, or states.

**Objective criteria:**

At least one of the following applies:

- reconstruct where an entity was before or after another event when the
  relevant state must be recovered from different phases;
- track the movement or relocation of an entity;
- compare the spatial state of the same entity across multiple phases;
- determine how an event changes a spatial configuration;
- integrate more than two relevant evidence units.

**Evidence scope:** `multi_event_sequence`

**Required auxiliary ability:** temporal reasoning, normally `T1` or `T2`

**Typical questions:**

“Where was the phone before the woman answered?”

“Where was the object before it was picked up, and where is it afterwards?”

---

# T labels — Temporal reasoning

## T0 — Event recognition

Recognize one event or state.

No ordering, duration estimation, temporal comparison, or reconstruction is
required.

**Objective criteria:**

- one event/state;
- one evidence unit;
- no comparison with another event;
- no temporal sequence reconstruction.

In this dataset, T0 can also be used as an auxiliary prerequisite for other
dimensions.

---

## T1 — Explicit temporal localization or pairwise ordering

Locate an event relative to an explicit temporal anchor or determine the
temporal relation between two explicitly observable events.

This category includes both temporal localization and pairwise event ordering.

**Objective criteria:**

At least one of the following applies:

- locate one event relative to an explicit time, date, phase, or temporal
  anchor;
- identify two events and determine which occurs first;
- determine a direct `before`, `after`, or equivalent pairwise temporal
  relation.

**Maximum relevant evidence units:** 2

No reconstruction of a longer event sequence is necessary.

**Evidence scope:** `local_interval` or `two_intervals`

**Typical questions:**

“When did the event happen?”

“Did the event happen before or after 11:30?”

“Does the woman sigh before or after the phone rings?”

---

## T2 — Temporal sequence or state reconstruction

Reconstruct temporal information distributed across a sequence rather than a
single pair of events.

**Objective criteria:**

At least one of the following applies:

- count repetitions of an action;
- estimate or compare duration across an interval;
- identify the onset, continuation, or completion of a state;
- distinguish an earlier state from a later state;
- determine whether an action was already in progress when another event
  occurred;
- reconstruct an ordered sequence involving more than two relevant
  evidence units.

**Evidence scope:** `multi_event_sequence`

**Typical questions:**

“After how many repetitions does the assistant help?”

“Was the action already in progress, or did it begin after the other event?”

---

# C labels — Causal reasoning

## C0 — Explicitly stated cause

Retrieve a cause that is directly and unambiguously stated in the available
evidence.

No causal reconstruction is required.

**Objective criteria:**

- the causal relation is explicitly expressed in the available evidence;
- one evidence unit is sufficient;
- no connection between separate events needs to be inferred.

No current item is necessarily assigned C0 solely from question form and
existing metadata; the label remains available when explicit causal evidence
is present.

---

## C1 — Explicit causal relation

Establish a causal relation between directly observable or explicitly
available antecedent and consequence information.

This category includes both direct local causation and causal links distributed
across two explicit evidence units.

**Objective criteria:**

At least one of the following applies:

- identify one directly observable physical or behavioral cause–effect pair;
- connect an explicit antecedent to an explicit consequence;
- connect an explicit instruction or goal to the corresponding action or
  outcome.

**Maximum relevant evidence units:** 2

The causal link must be supported without reconstructing an unstated mental,
social, or contextual explanation.

**Evidence scope:** `local_interval` or `two_intervals`

**Possible auxiliary ability:** normally `T1`

**Typical questions:**

“Why does the paper make a bang?”

“Why does the trainer tell the athlete to perform the exercise?”

---

## C2 — Implicit or multi-event causal reconstruction

Infer a causal, intentional, social, or motivational relation by integrating
context distributed across multiple events or states.

**Objective criteria:**

At least one of the following applies:

- the causal relation is not directly stated or locally observable;
- an unstated intention, motivation, mental state, or social reason must be
  inferred;
- more than two evidence units must be integrated;
- a causal explanation depends on reconstructing a change of state;
- multiple preceding events must be connected to explain a later reaction or
  consequence;
- the solution requires an implicit causal chain.

**Evidence scope:** `multi_event_sequence`

**Required auxiliary ability:** normally temporal reconstruction (`T2`)

**Typical questions:**

“Why is the person worried?”

“Why does the character react this way after the sequence of preceding events?”

---

# Mapping from the previous scale

The previous five-level scale is converted deterministically as follows:

| Previous level | New level |
|----------------|-----------|
| 0              | 0         |
| 1              | 1         |
| 2              | 1         |
| 3              | 2         |
| 4              | 2         |

Accordingly:

- `S0 → S0`
- `S1, S2 → S1`
- `S3, S4 → S2`

- `T0 → T0`
- `T1, T2 → T1`
- `T3, T4 → T2`

- `C0 → C0`
- `C1, C2 → C1`
- `C3, C4 → C2`

---

# CSV columns

## `diagnostic_label`

Principal label followed by any auxiliary labels.

Examples:

`S1|T0`

`S2|T1`

`C2|T2`

---

## `complexity_level`

Numerical complexity of the principal dimension:

- `0` = direct retrieval;
- `1` = explicit relational integration;
- `2` = multi-step, implicit, or state-transition integration.

---

## `principal_label`

The `S`, `T`, or `C` label corresponding to the primary operation required by
the question.

Examples:

`S0`, `S1`, `S2`

`T0`, `T1`, `T2`

`C0`, `C1`, `C2`

---

## `auxiliary_labels`

Prerequisite operations belonging to dimensions different from the principal
one.

Auxiliary labels describe necessary supporting operations but do not determine
the principal complexity level.

---

## `objective_criterion`

Short textual justification identifying the observable rule that triggered the
classification.

The justification should refer to operations such as:

- direct retrieval;
- one explicit relation;
- alignment of two evidence units;
- pairwise temporal ordering;
- multi-event reconstruction;
- state-transition tracking;
- implicit causal inference.

It should not contain subjective expressions such as “easy”, “difficult”,
“complex”, or “challenging”.

---

## `evidence_scope`

One of:

- `single_frame`
- `local_interval`
- `two_intervals`
- `multi_event_sequence`

---

## `evidence_units`

Minimum number of distinct evidence units that must be retrieved and integrated
to answer the question.

This value should refer to logically necessary evidence rather than the total
number of events visible in the video.

---

## `dimension_combination`

List of reasoning dimensions that must be integrated to solve the question.

Examples:

`[S]`

`[S,T]`

`[T]`

`[C,T]`

`[C,S,T]`

---

# Methodological limitation

The scale measures the **structural complexity of the operations required by
the question**.

It does not measure the perceptual difficulty of extracting the corresponding
evidence from a particular video.

Consequently, factors such as:

- occlusion;
- video quality;
- number of visually similar entities;
- event speed;
- speech intelligibility;
- acoustic noise;
- visual ambiguity;

do not affect the S/T/C complexity level.

These properties should be represented separately if the objective is to
measure the empirical perceptual difficulty of individual video instances.