#!/usr/bin/env python3
#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

MODEL_ID = "google/gemma-3n-E4B-it"
SEMANTIC_DIR = Path("data/preliminar_analysis/entity/entity_semantic/gemma")
OUTPUT_DIR = Path("data/preliminar_analysis/event/gemma")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("La risposta non contiene un oggetto JSON.")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        from json_repair import repair_json
        return json.loads(repair_json(candidate))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def compact(payload):
    keys = (
        "segment_id", "start_time", "end_time", "entities", "actions", "events",
        "spatial_relations", "state_changes", "temporal_relations", "causal_hypotheses"
    )
    return {
        "id_video": payload.get("id_video"),
        "segments": [{key: segment.get(key, []) for key in keys} for segment in payload.get("segments", [])],
    }


def build_prompt(payload):
    return f"""Consolida i segmenti dell'analisi semantica precedente in una rappresentazione cronologica degli eventi.

Regole:
- usa esclusivamente le informazioni presenti nell'analisi semantica;
- unisci soltanto i duplicati causati dalla sovrapposizione delle finestre;
- mantieni separati eventi realmente distinti o ripetuti;
- ricava i tempi soltanto dai segmenti forniti;
- non inventare intenzioni, cause, emozioni o azioni mancanti;
- usa evidence_type="inferred" soltanto se l'input contiene esplicitamente un'inferenza;
- assegna gli ID E0001, E0002, ... in ordine temporale;
- restituisci esclusivamente JSON valido.

Schema:
{{
  "events": [
    {{
      "event_id": "E0001",
      "description": "string",
      "start_time": 0.0,
      "end_time": 0.0,
      "participants": ["string"],
      "evidence_segments": ["segment_0000"],
      "evidence_type": "observed|inferred|uncertain",
      "confidence": 0.0
    }}
  ],
  "temporal_relations": [
    {{
      "first_event": "E0001",
      "relation": "before|after|overlaps|during|simultaneous",
      "second_event": "E0002",
      "confidence": 0.0
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}"""


class Inferencer:
    def __init__(self, model_id, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def __call__(self, prompt):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device, dtype=self.model.dtype) if value.is_floating_point() else value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)

        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Event extraction Gemma-3n-E4B dalla propria analisi semantica.")
    parser.add_argument("semantic_directory", nargs="?", type=Path, default=SEMANTIC_DIR)
    parser.add_argument("output_directory", nargs="?", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.semantic_directory.glob("*_semantic.json"))
    if args.limit_videos:
        files = files[:args.limit_videos]
    if not files:
        parser.error(f"Nessun *_semantic.json trovato in {args.semantic_directory}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    infer = Inferencer(args.model, args.max_new_tokens)

    for index, path in enumerate(files, 1):
        payload = read_json(path)
        video_id = payload.get("id_video") or path.stem.removesuffix("_semantic")
        output_path = args.output_directory / f"{video_id}_events.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] SKIP {video_id}")
            continue

        print(f"[{index}/{len(files)}] {video_id} su CUDA:0")
        raw = infer(build_prompt(compact(payload)))
        result = parse_json(raw)
        result.update({"id_video": video_id, "model": args.model, "source_semantic_file": str(path)})
        write_json(output_path, result)

    print(f"Creato output in: {args.output_directory}")


if __name__ == "__main__":
    main()