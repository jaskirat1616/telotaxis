"""Build the visual pathway from the public MaleCNS v1.0 tables.

Keeps every traced optic-lobe intrinsic neuron (lamina L1–L5 included), every
traced visual projection neuron, and every traced descending neuron. Keeps
every synapse of weight at least 3 between them whose presynaptic transmitter
is acetylcholine, GABA, glutamate, or histamine. Modulatory and unresolved
transmitters are dropped, not guessed.

Writes data/pathway.npz. The raw feather files are not copied into the repo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "pathway.npz"

EXCITATORY = {"acetylcholine"}
INHIBITORY = {"gaba", "glutamate", "histamine"}
MIN_SYNAPSES = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ann_path = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    nt_path = RAW / "body-neurotransmitters-male-cns-v1.0.feather"
    w_path = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    for path in (ann_path, nt_path, w_path):
        if not path.exists():
            raise SystemExit(f"missing {path}. See README for the public URLs.")

    print("reading annotations")
    ann = feather.read_table(
        ann_path,
        columns=[
            "bodyId", "status", "superclass", "type", "somaSide",
            "somaLocation", "assignedOlHex1", "assignedOlHex2",
        ],
    ).to_pandas()
    traced = ann[ann.status == "Traced"].copy()
    lamina = traced["type"].isin(["L1", "L2", "L3", "L4", "L5"])
    optic = traced["superclass"] == "ol_intrinsic"
    vpn = traced["superclass"] == "visual_projection"
    descending = traced["superclass"] == "descending_neuron"
    chosen = traced[optic | vpn | descending].copy()
    chosen = chosen.drop_duplicates("bodyId")
    chosen["bodyId"] = chosen["bodyId"].astype(np.int64)
    print(f"  neurons {len(chosen)}  lamina {int(lamina.sum())}  vpn {int(vpn.sum())}  dn {int(descending.sum())}")

    print("reading transmitters")
    nt = feather.read_table(nt_path, columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body")
    sign_by_name = {name: 1 for name in EXCITATORY}
    sign_by_name.update({name: -1 for name in INHIBITORY})
    nt["sign"] = nt["consensus_nt"].map(sign_by_name).fillna(0).astype(np.int8)
    sign_of = dict(zip(nt["body"].astype(np.int64), nt["sign"].astype(np.int8)))

    ids = np.sort(chosen["bodyId"].to_numpy(np.int64))
    sign = np.array([sign_of.get(int(body), np.int8(0)) for body in ids], dtype=np.int8)

    def locate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.searchsorted(ids, values)
        pos = np.minimum(pos, len(ids) - 1)
        ok = ids[pos] == values
        return pos, ok

    print("scanning synapses")
    pres: list[np.ndarray] = []
    posts: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    table = feather.read_table(w_path, columns=["body_pre", "body_post", "weight"], memory_map=True)
    seen = 0
    for batch in table.to_batches(max_chunksize=4_000_000):
        pre = batch.column(0).to_numpy(zero_copy_only=False)
        post = batch.column(1).to_numpy(zero_copy_only=False)
        weight = batch.column(2).to_numpy(zero_copy_only=False)
        ip, ok_pre = locate(pre)
        iq, ok_post = locate(post)
        keep = ok_pre & ok_post & (weight >= MIN_SYNAPSES) & (sign[ip] != 0)
        if np.any(keep):
            pres.append(ip[keep].astype(np.int32))
            posts.append(iq[keep].astype(np.int32))
            counts.append(np.clip(weight[keep], 0, 65535).astype(np.uint16))
        seen += len(pre)
        if seen % 40_000_000 < 4_000_000:
            print(f"  {seen / 1e6:.0f}M rows", flush=True)
    pre_i = np.concatenate(pres)
    post_i = np.concatenate(posts)
    count = np.concatenate(counts)
    print(f"  kept {len(count):,} synapses")

    by_id = chosen.set_index("bodyId")
    types = by_id.loc[ids, "type"].fillna("").astype(str).to_numpy()
    sides_raw = by_id.loc[ids, "somaSide"].fillna("").astype(str).to_numpy()
    side = np.zeros(len(ids), np.int8)
    side[sides_raw == "L"] = -1
    side[sides_raw == "R"] = 1
    role = np.zeros(len(ids), np.uint8)
    role[np.isin(types, ["L1", "L2", "L3", "L4", "L5"])] = 1
    role[by_id.loc[ids, "superclass"].to_numpy() == "visual_projection"] = 2
    role[by_id.loc[ids, "superclass"].to_numpy() == "descending_neuron"] = 3
    dnp = np.array([str(name).startswith("DNp") for name in types], dtype=np.uint8)

    hex1 = by_id.loc[ids, "assignedOlHex1"].to_numpy(dtype=np.float64)
    hex2 = by_id.loc[ids, "assignedOlHex2"].to_numpy(dtype=np.float64)
    column, column_azimuth, column_side = _columns(role, side, hex1, hex2)

    soma = by_id.loc[ids, "somaLocation"]
    position = _positions(soma.to_numpy(), side, role, hex1, hex2)

    meta = {
        "dataset": "MaleCNS v1.0",
        "licence": "CC-BY-4.0",
        "source": "https://male-cns.janelia.org/download/",
        "paper": "10.1016/j.cell.2026.08.015",
        "neurons": int(len(ids)),
        "synapses": int(len(count)),
        "min_synapses": MIN_SYNAPSES,
        "lamina": int(np.count_nonzero(role == 1)),
        "visual_projection": int(np.count_nonzero(role == 2)),
        "descending": int(np.count_nonzero(role == 3)),
        "dnp": int(np.count_nonzero(dnp)),
        "columns": int(len(column_azimuth)),
        "omitted": "central-brain intrinsic, ventral-cord intrinsic, sensory neurons outside the optic lobe, photoreceptors",
        "sign": "acetylcholine +1; GABA, glutamate, histamine -1; modulatory and unclear dropped",
        "sha256": {
            ann_path.name: sha256(ann_path),
            nt_path.name: sha256(nt_path),
            w_path.name: sha256(w_path),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        ids=ids,
        pre=pre_i,
        post=post_i,
        count=count,
        sign=sign,
        role=role,
        side=side,
        dnp=dnp,
        column=column,
        column_azimuth=column_azimuth.astype(np.float32),
        column_side=column_side,
        position=position.astype(np.float32),
        types=np.array(types, dtype="U24"),
        meta=np.array(json.dumps(meta)),
    )
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps({k: meta[k] for k in ("neurons", "synapses", "lamina", "visual_projection", "descending", "dnp", "columns")}, indent=2))


def _columns(role, side, hex1, hex2):
    """One column per optic-lobe hex. Left eye maps to positive azimuth."""

    lamina = np.flatnonzero(role == 1)
    keys = {}
    column = np.full(len(role), -1, np.int32)
    azimuths = []
    sides = []
    for index in lamina:
        if not np.isfinite(hex1[index]) or not np.isfinite(hex2[index]):
            continue
        eye = int(side[index])
        if eye == 0:
            continue
        key = (eye, int(hex1[index]), int(hex2[index]))
        if key not in keys:
            keys[key] = len(azimuths)
            azimuths.append(0.0)
            sides.append(eye)
        column[index] = keys[key]
    if not azimuths:
        raise RuntimeError("no lamina hexes")
    # Horizontal axis of the hex lattice, per eye, mapped into that eye's field.
    buckets: dict[int, list[int]] = {}
    for key, col in keys.items():
        buckets.setdefault(key[0], []).append(col)
    az = np.zeros(len(azimuths), np.float64)
    for eye, cols in buckets.items():
        xs = np.array([_hex_x(k) for k, col in keys.items() if col in set(cols)], dtype=np.float64)
        # keys iteration order is insertion order, which matches cols. Rebuild explicitly.
        pairs = [(key, col) for key, col in keys.items() if key[0] == eye]
        # hex2 is the axis that lateralises both eyes. The left eye's lattice
        # is mirrored relative to the right, so that eye's unit is flipped.
        xs = np.array([key[2] + 0.5 * key[1] for key, _ in pairs], dtype=np.float64)
        lo, hi = float(xs.min()), float(xs.max())
        span = max(hi - lo, 1e-6)
        unit = (xs - lo) / span
        if eye < 0:
            unit = 1.0 - unit
        for (key, col), u in zip(pairs, unit):
            # Left eye (side -1) sees the agent's left, azimuth > 0.
            if eye < 0:
                az[col] = 6.0 + float(u) * 124.0
            else:
                az[col] = -6.0 - float(u) * 124.0
    return column, az, np.array(sides, np.int8)


def _hex_x(key) -> float:
    return key[1] + 0.5 * key[2]


def _positions(soma, side, role, hex1, hex2) -> np.ndarray:
    """Soma coordinates where the paper has them. Hex shell where it does not."""

    n = len(side)
    raw = np.full((n, 3), np.nan, np.float64)
    for index, value in enumerate(soma):
        if isinstance(value, np.ndarray) and value.shape == (3,):
            raw[index] = value.astype(np.float64)
    known = np.isfinite(raw[:, 0])
    center = np.median(raw[known], axis=0)
    left = np.nanmean(raw[(side < 0) & known], axis=0)
    right = np.nanmean(raw[(side > 0) & known], axis=0)
    lateral = left - right
    lateral /= np.linalg.norm(lateral)
    lamina_known = known & (role == 1)
    dn_known = known & (role == 3)
    anterior = np.nanmean(raw[lamina_known], axis=0) - np.nanmean(raw[dn_known], axis=0)
    anterior = anterior - lateral * np.dot(anterior, lateral)
    anterior /= np.linalg.norm(anterior)
    dorsal = np.cross(lateral, anterior)
    dorsal /= np.linalg.norm(dorsal)
    basis = np.stack([lateral, anterior, dorsal], axis=1)
    placed = (raw - center) @ basis
    # Lamina neurons without a soma sit on the outer shell of their eye.
    missing = (role == 1) & ~known & np.isfinite(hex1) & np.isfinite(hex2)
    for eye in (-1, 1):
        members = np.flatnonzero(missing & (side == eye))
        if len(members) == 0:
            continue
        anchor = placed[known & (side == eye)]
        origin = np.median(anchor, axis=0) if len(anchor) else np.array([1.2 * eye * -1, 0.2, 0.4])
        hx = hex1[members]
        hy = hex2[members]
        hx = (hx - np.nanmean(hx)) / (np.nanstd(hx) + 1e-6)
        hy = (hy - np.nanmean(hy)) / (np.nanstd(hy) + 1e-6)
        placed[members, 0] = origin[0] + eye * -0.15
        placed[members, 1] = origin[1] + hy * 0.12
        placed[members, 2] = origin[2] + hx * 0.08
    # Anyone still missing goes to the centre of their side, faint and medial.
    still = ~np.isfinite(placed[:, 0])
    placed[still, 0] = side[still] * -0.2
    placed[still, 1] = 0.0
    placed[still, 2] = 0.0
    scale = 1.15 / np.percentile(np.linalg.norm(placed, axis=1), 98)
    return placed * scale


if __name__ == "__main__":
    main()
