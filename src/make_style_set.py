"""Build style splits from one WikiArt shard. Split is BY ARTIST so test styles come from unseen artists."""
import io, json, random, shutil
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

OUT = Path("data/styles"); SEED = 0
N = {"train": 5000, "val": 30, "test": 40}
CAP = {"train": 20, "val": 3, "test": 3}      # max paintings per artist (val/test unchanged from v1)
pf = pq.ParquetFile("data/wikiart_shard0.parquet")
rows = []
for g in range(pf.num_row_groups):
    t = pf.read_row_group(g, columns=["image", "artist", "style", "title"]).to_pylist()
    for i, r in enumerate(t):
        rows.append((g, i, r["artist"], r["style"], r["title"]))
by_artist = {}
for g, i, a, s, ti in rows:
    by_artist.setdefault(a, []).append((g, i, s, ti))
artists = sorted(by_artist); random.Random(SEED).shuffle(artists)
assign = {"test": [], "val": [], "train": []}
counts = {"test": 0, "val": 0, "train": 0}
rng = random.Random(SEED + 1)
for a in artists:
    for split in ("test", "val", "train"):
        if counts[split] < N[split]:
            take = rng.sample(by_artist[a], min(CAP[split], len(by_artist[a])))
            assign[split] += [(a, *x) for x in take]; counts[split] += len(take); break
if OUT.exists(): shutil.rmtree(OUT)
meta = {}
cache = {}
for split, items in assign.items():
    (OUT / split).mkdir(parents=True)
    for k, (a, g, i, s, ti) in enumerate(items[:N[split]]):
        if g not in cache: cache = {g: pf.read_row_group(g, columns=["image"]).to_pylist()}
        im = Image.open(io.BytesIO(cache[g][i]["image"]["bytes"])).convert("RGB")
        name = f"{split}_{k:04d}.png"; im.save(OUT / split / name)
        meta[f"{split}/{name}"] = {"artist": a, "style": s, "title": ti, "size": im.size}
# the three original project styles as extra showcase styles (native resolution)
(OUT / "showcase").mkdir()
for n in ["cuphead", "mosaic", "starry_night"]:
    shutil.copy(f"../Fast-Neural-Style-Transfer/images/styles/{n}.jpg", OUT / "showcase" / f"{n}.jpg")
json.dump(meta, open(OUT / "styles_meta.json", "w"), indent=1)
for s in N: print(s, len(list((OUT / s).glob("*.png"))), "images,", len({m["artist"] for k, m in meta.items() if k.startswith(s)}), "artists")
tr = {m["artist"] for k, m in meta.items() if k.startswith("train")}
te = {m["artist"] for k, m in meta.items() if k.startswith("test")}
print("artist overlap train/test:", len(tr & te))
