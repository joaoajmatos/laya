"""`golden weights-inventory`: every state_dict key with shape and dtype (laya:004 US1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from . import common


def inventory_of(model) -> Dict[str, Any]:
    entries = {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
               for k, v in model.state_dict().items()}
    return {"n_keys": len(entries), "n_parameters": sum(int(v.numel()) for v in model.state_dict().values()),
            "keys": entries}


def write_inventory(out: Path, checkpoint: Optional[str] = None, hidden: int = 64) -> Path:
    """Fixture model by default; a real local checkpoint only when `checkpoint` is given."""
    import laya
    out = Path(out)
    common.make_deterministic()
    if checkpoint:
        agent = laya.Agent(checkpoint, device="cpu")
        body = inventory_of(agent.model)
        meta = {"model": "checkpoint", "checkpoint_dir_name": Path(checkpoint).name}
    else:
        with common.FixtureAgent(hidden) as fx:
            body = inventory_of(fx.agent.model)
            meta = {"model": "fixture", "fixture_hidden": hidden,
                    "agent_config": fx.agent.cfg, "encoder_config": common.encoder_config(fx.agent.model)}
    body["meta"] = common.base_meta("weights-inventory", **meta)
    return common.write_json(out / "inventory.json", body)
