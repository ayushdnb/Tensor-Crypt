"""Durable lineage export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_lineage_graph(registry, *, life_rows_by_uid: dict[int, dict] | None = None) -> dict[str, Any]:
    life_rows_by_uid = life_rows_by_uid or {}
    nodes = []
    edges = []

    for uid, record in sorted(registry.uid_lifecycle.items()):
        node = {
            "uid": int(uid),
            "family": registry.uid_family.get(uid),
            "birth_tick": int(record.birth_tick),
            "death_tick": None if record.death_tick is None else int(record.death_tick),
            "lineage_depth": int(registry.uid_generation_depth.get(uid, 0)),
            "brain_parent_uid": int(registry.uid_parent_roles.get(uid, {}).get("brain_parent_uid", record.parent_uid)),
            "trait_parent_uid": int(registry.uid_parent_roles.get(uid, {}).get("trait_parent_uid", record.parent_uid)),
            "anchor_parent_uid": int(registry.uid_parent_roles.get(uid, {}).get("anchor_parent_uid", record.parent_uid)),
        }
        node.update(life_rows_by_uid.get(uid, {}))
        nodes.append(node)

        roles = registry.uid_parent_roles.get(
            uid,
            {
                "brain_parent_uid": int(record.parent_uid),
                "trait_parent_uid": int(record.parent_uid),
                "anchor_parent_uid": int(record.parent_uid),
            },
        )
        for role_name in ("brain_parent_uid", "trait_parent_uid", "anchor_parent_uid"):
            parent_uid = int(roles.get(role_name, -1))
            if parent_uid == -1:
                continue
            edges.append(
                {
                    "parent_uid": parent_uid,
                    "child_uid": int(uid),
                    "edge_type": role_name.removesuffix("_uid"),
                }
            )

    return {
        "format": "tensor_crypt_lineage_graph_v1",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def export_lineage_json(path: str | Path, registry, *, life_rows_by_uid: dict[int, dict] | None = None) -> dict[str, Any]:
    payload = build_lineage_graph(registry, life_rows_by_uid=life_rows_by_uid)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload
