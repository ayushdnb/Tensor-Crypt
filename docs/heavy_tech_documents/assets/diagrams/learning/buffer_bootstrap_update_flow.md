# Buffer / Bootstrap / Update Flow

> Owning document: [PPO buffers, bootstrap, and update cadence](../../../04_learning/01_ppo_buffers_bootstrap_and_update_cadence.md)

## What this asset shows
- the buffer fields and closure paths

## What this asset intentionally omits
- exact per-field tensor dtypes

```mermaid
flowchart TD
    A[transition] --> B[obs/actions/log_probs/rewards/values/dones]
    B --> C{tail terminal?}
    C -- yes --> D[bootstrap_done = 1, no obs]
    C -- no --> E[stage bootstrap obs + done=0]
    D --> F[returns and advantages]
    E --> F
    F --> G[update if ready]
    G --> H[clear buffer]

```
