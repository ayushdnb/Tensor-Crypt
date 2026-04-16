# High-School Math for Scaling, Rates, and Normalization

> Scope: Explain the arithmetic and ratio concepts that recur throughout Tensor Crypt observations, traits, health logic, and telemetry.

## Who this document is for
Readers building prerequisite knowledge before entering repository-specific chapters.

## What this document covers
- ratios and proportions
- clamping
- min-max normalization
- signed normalization
- distance and center measures
- why bounded features matter

## What this document does not cover
- detailed subsystem auditing
- operator instructions unless explicitly included

## Prerequisite reading
- [Documentation index](../00_program/00_documentation_index_and_reading_guide.md)
- [Glossary and notation legend](../00_program/02_glossary_notation_and_schema_legend.md)

## 1. Ratios are the basic language of the repository

The code dump repeatedly converts raw quantities into bounded fractions:
- HP becomes `hp / hp_max`
- position becomes `x / (W - 1)` and `y / (H - 1)`
- age becomes `age_ticks / AGE_NORM_TICKS`
- zone rates become a signed value divided by a configured absolute maximum

A ratio is useful because it allows very different raw scales to be compared. A mass of `7.5` and a vision of `14.0` are not naturally comparable, but two numbers each mapped into `[0, 1]` are.

## 2. Clamping

A clamp prevents a value from leaving a legal interval.

```text
clamp(value, low, high)
= low    if value < low
= value  if low <= value <= high
= high   if value > high
```

Clamping appears whenever the repository wants a stable feature range or wants to prevent impossible state:
- HP is clamped between `0` and `HP_MAX`
- normalized ratios are clamped into `[0, 1]`
- signed zone features are clamped into `[-1, 1]`

## 3. Min-max normalization

A raw value `x` with known lower and upper bounds can be normalized as

```text
(x - lower) / (upper - lower)
```

When the repository normalizes mass, HP max, vision, or metabolism using configured trait clamps, it is using this idea. After normalization:
- `0` means “at the configured lower bound”
- `1` means “at the configured upper bound”
- intermediate values express position between those bounds

## 4. Signed normalization

Not all fields are one-sided. Zone rates can be healing or harmful. A convenient map is:

```text
normalized_signed = clamp(raw / max_abs, -1, 1)
```

That preserves sign:
- positive means healing
- negative means harm
- magnitude expresses intensity relative to the chosen maximum

## 5. Distance and center

The repository uses position not only as raw coordinates but also as derived spatial context. A useful question is: how far is an agent from the world center? That quantity can also be normalized, turning a raw geometric distance into a bounded scalar suitable for observations.

## 6. Why bounded features matter

A neural policy behaves better when feature scales are reasonably controlled. If one feature lives near `0.2` and another near `10,000`, the larger one may dominate early optimization unless the architecture and training process compensate. Tensor Crypt’s observation schema therefore converts many raw state fields into bounded or standardized quantities before the brain consumes them.


## Why this matters for Tensor Crypt
The canonical observation bundle contains normalized self and context features, and the perception path produces signed zone-rate features and normalized target statistics. Without this layer, the observation contract would look arbitrary rather than deliberate.

## Read next
- [Probability, statistics, and expected value for RL](02_probability_statistics_and_expected_value_for_rl.md)
- [Linear algebra, tensors, shapes, and batching](04_linear_algebra_tensors_shapes_and_batching.md)
- [Observation schema, perception, and ray semantics](../03_mechanics/04_observation_schema_perception_and_ray_semantics.md)

## Related reference
- [Game manual and rulebook](../06_game_manual/00_tensor_crypt_game_manual_and_rulebook.md)
- [Repository identity and public contract](../02_system/00_repository_identity_entry_surfaces_and_public_contract.md)

## If debugging this, inspect…
- [Foundations learning roadmap](00_foundations_learning_roadmap.md)
