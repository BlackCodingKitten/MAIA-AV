import argparse, itertools, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path("data/preliminar_analysis")
QUESTIONS = Path("data/vsv/question_classification.csv")
OUT = ROOT / "final"
MODELS = ("gemma", "gemma-27B", "qwen", "qwen-30B")


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def txt(*x):
    return " ".join(str(v).strip() for v in x if v not in (None, "", []))


def conf(x):
    try:
        return float(x.get("confidence", 1))
    except (TypeError, ValueError):
        return 0.0


def semantic(path, min_conf):
    entities, spatial = [], []
    for s in load(path).get("segments", []):
        ids = {}
        for e in s.get("entities", []):
            if conf(e) < min_conf:
                continue
            t = txt(e.get("label"), e.get("role"), *e.get("attributes", []))
            if t:
                ids[e.get("entity_id")] = t
                entities.append({"text": t})
        for r in s.get("spatial_relations", []):
            if conf(r) >= min_conf:
                spatial.append({"text": txt(ids.get(r.get("subject"), r.get("subject")),
                                             r.get("relation"),
                                             ids.get(r.get("object"), r.get("object")))})
        for c in s.get("state_changes", []):
            if conf(c) >= min_conf:
                spatial.append({"text": txt(ids.get(c.get("entity"), c.get("entity")),
                                             c.get("before"), "->", c.get("after"))})
    return entities, spatial


def events(path, min_conf):
    return [{
        "id": e.get("event_id"),
        "text": txt(e.get("description"), *e.get("participants", [])),
        "start": float(e.get("start_time", 0)),
        "end": float(e.get("end_time", e.get("start_time", 0))),
    } for e in load(path).get("events", []) if conf(e) >= min_conf]


def causal(path, min_conf):
    return [r for r in load(path).get("causal_relations", [])
            if conf(r) >= min_conf and r.get("evidence_type") != "uncertain"]


class Match:
    def __init__(self, name, device):
        self.model, self.cache = SentenceTransformer(name, device=device), {}

    def vec(self, strings):
        strings = [str(x).strip() for x in strings if str(x).strip()]
        missing = list(dict.fromkeys(x for x in strings if x not in self.cache))
        if missing:
            self.cache.update(zip(missing, self.model.encode(
                missing, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)))
        return [self.cache[x] for x in strings]

    def best(self, query, items, exclude=()):
        items = [x for x in items if x.get("id") not in set(exclude) and x.get("text")]
        if not query or not items:
            return 0.0, None
        q, *v = self.vec([query] + [x["text"] for x in items])
        scores = [float(q @ x) for x in v]
        i = int(np.argmax(scores))
        return scores[i], items[i]

    def count(self, query, items, threshold):
        if not query or not items:
            return 0
        q, *v = self.vec([query] + [x["text"] for x in items])
        return sum(float(q @ x) >= threshold for x in v)


PREFIX = re.compile(
    r"^(evento(?:/stato)?(?:\s+[AB])?|evento target|evento contestuale|"
    r"evento o stato iniziale|stato target|insieme di eventi|effetto da spiegare|"
    r"relazione causale/finalistica esplicitata nella domanda)\s*:\s*", re.I)

GENERIC = ("luogo ", "posizione ", "localizzazione ", "destinazione ",
           "referente ", "contenuto ", "informazione ", "riferimento ", "misura ")


def split(x):
    return [v.strip() for v in str(x or "").split(";") if v.strip()]


def entity_req(x):
    return [v for v in split(x) if not v.lower().startswith(GENERIC)]


def event_req(x):
    return [PREFIX.sub("", v).strip() for v in split(x)
            if "da identificare nell'evidenza" not in v.lower()
            and not v.lower().startswith(("target:", "riferimento temporale:",
                                          "misura temporale/ordinale:"))]


def units(x):
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else 1


def relation(a, b):
    return "before" if a["end"] < b["start"] else "after" if b["end"] < a["start"] else "overlap"


def entity_match(req, evidence, m, threshold):
    if not req:
        return 1.0, True
    scores = [m.best(q, evidence)[0] for q in req]
    return min(scores), all(x >= threshold for x in scores)


def diagnose(q, ent, spa, eve, cau, m, threshold):
    label = str(q.principal_label)
    es, eok = entity_match(entity_req(q.entity), ent, m, threshold)
    er = event_req(q.event)
    vs, ve = m.best(er[0] if er else q.question_text, eve)
    vok, rs, rv = vs >= threshold, 0.0, ""

    if label.startswith("S"):
        rs, _ = m.best(q.event, spa)
        rok = rs >= threshold
        ok = (eok and rok if label == "S0" else
              eok and vok and rok if label == "S1" else
              eok and vok and rok and m.count(q.event, spa, threshold) >= units(q.evidence_units))

    elif label == "T0":
        rs, rv, ok = vs, "single_event", vok

    elif label == "T1":
        if len(er) < 2:
            ok = False
        else:
            s1, a = m.best(er[0], eve)
            s2, b = m.best(er[1], eve, {a["id"]} if a else ())
            vs, rs, ve = min(s1, s2), min(s1, s2), a
            rv = relation(a, b) if a and b else ""
            ok = bool(a and b and rs >= threshold)

    elif label == "T2":
        rs, rv, ok = vs, "multi_event_sequence", vok and len(eve) >= units(q.evidence_units)

    elif label == "C0":
        rs, rv, ok = es, "explicit_in_question", eok

    else:
        vs, ve = m.best(er[0] if er else q.question_text, eve)
        vok = vs >= threshold
        wanted = "direct" if label == "C1" else "inferred"
        found = [r for r in cau if ve and r.get("effect_event") == ve.get("id")
                 and r.get("evidence_type") == wanted]
        if found:
            best = max(found, key=conf)
            rs, rv = conf(best), best.get("relation", "")
        ok = vok and bool(found)

    return {
        "entity_score": round(es, 4), "entity_ok": int(eok),
        "event_score": round(vs, 4), "event_ok": int(vok),
        "relation_score": round(rs, 4), "relation_value": rv,
        "matched_event": ve["text"] if ve else "",
        "prerequisites_recovered": int(ok),
    }


def agreement(df):
    rows = []
    for qid, g in df.groupby("question_id"):
        v = g.loc[g.analysis_valid == 1, "prerequisites_recovered"].tolist()
        pairs = list(itertools.combinations(v, 2))
        rows.append({
            "question_id": qid, "video_id": g.video_id.iloc[0],
            "principal_label": g.principal_label.iloc[0],
            "models_valid": len(v), "models_recovered": sum(v),
            "recovery_fraction": np.mean(v) if v else np.nan,
            "pairwise_agreement": np.mean([a == b for a, b in pairs]) if pairs else np.nan,
            "stability": ("no_valid_models" if not v else "stable_positive" if all(v)
                          else "stable_negative" if not any(v) else "mixed"),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=.45)
    p.add_argument("--min-confidence", type=float, default=.25)
    p.add_argument("--embedding-model",
                   default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    questions = pd.read_csv(QUESTIONS, sep=";", encoding="utf-8-sig")
    matcher, rows = Match(a.embedding_model, a.device), []

    for model in MODELS:
        for _, q in questions.iterrows():
            video = Path(q.video_name).stem
            sp = ROOT / "entity" / model / f"{video}_semantic.json"
            ep = ROOT / "event" / model / f"{video}_events.json"
            cp = ROOT / "causal" / model / f"{video}_causal.json"
            raw = [load(x) for x in (sp, ep, cp)]
            ent, spa = semantic(sp, a.min_confidence)
            eve, cau = events(ep, a.min_confidence), causal(cp, a.min_confidence)

            rows.append({
                "question_id": f"{video}_q{q.question_order}", "video_id": video,
                "question_order": q.question_order, "question_text": q.question_text,
                "principal_dimension": q.principal_dimension, "principal_label": q.principal_label,
                "diagnostic_label": q.diagnostic_label, "complexity_level": q.complexity_level,
                "evidence_scope": q.evidence_scope, "evidence_units": q.evidence_units,
                "model": model,
                "analysis_valid": int(all(raw) and not any("error" in x for x in raw)),
                **diagnose(q, ent, spa, eve, cau, matcher, a.threshold),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "question_diagnostics.csv", index=False)
    agreement(df).to_csv(OUT / "inter_model_agreement.csv", index=False)
    print(f"Creati:\n{OUT/'question_diagnostics.csv'}\n{OUT/'inter_model_agreement.csv'}")


if __name__ == "__main__":
    main()