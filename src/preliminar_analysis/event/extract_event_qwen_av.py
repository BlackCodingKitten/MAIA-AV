from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
SEMANTIC_DIR = Path("data/preliminar_analysis/entity/qwen")
OUTPUT_DIR = Path("data/preliminar_analysis/event/qwen")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("La risposta non contiene un oggetto JSON.")

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(candidate))
        except Exception as error:
            raise ValueError("La risposta contiene JSON non valido.") from error


def compact(payload):
    keys = (
        "segment_id", "start_time", "end_time", "entities", "actions", "events",
        "spatial_relations", "state_changes", "temporal_relations", "causal_hypotheses",
    )
    return {
        "id_video": payload.get("id_video"),
        "segments": [
            {key: segment.get(key, []) for key in keys}
            for segment in payload.get("segments", [])
        ],
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
- restituisci ESCLUSIVAMENTE un oggetto JSON valido;
- il primo carattere della risposta deve essere {{ e l'ultimo deve essere }}.

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
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        self.model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    def __call__(self, prompt):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                return_audio=False,
            )

        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def infer_json(inferencer, prompt, attempts=2):
    raw = ""
    error = None

    for attempt in range(attempts):
        raw = inferencer(
            prompt if attempt == 0 else
            prompt + "\n\nLa risposta precedente non era JSON valido. Rispondi soltanto con l'oggetto JSON richiesto."
        )
        try:
            return parse_json(raw), raw
        except Exception as current_error:
            error = current_error

    raise ValueError(f"JSON non ottenuto dopo {attempts} tentativi. Ultima risposta: {raw[:500]!r}") from error


def main():
    parser = argparse.ArgumentParser(
        description="Event extraction Qwen2.5-Omni-3B dalla propria analisi semantica."
    )
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
    inferencer = Inferencer(args.model, args.max_new_tokens)

    for index, path in enumerate(files, 1):
        payload = read_json(path)
        video_id = payload.get("id_video") or path.stem.removesuffix("_semantic")
        output_path = args.output_directory / f"{video_id}_events.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] SKIP {video_id}")
            continue

        print(f"[{index}/{len(files)}] {video_id} su CUDA:6")

        try:
            result, raw = infer_json(inferencer, build_prompt(compact(payload)))
            result.update({
                "id_video": video_id,
                "model": args.model,
                "source_semantic_file": str(path),
            })
        except Exception as error:
            print(f"ERRORE {video_id}: {type(error).__name__}: {error}")
            result = {
                "id_video": video_id,
                "model": args.model,
                "source_semantic_file": str(path),
                "events": [],
                "temporal_relations": [],
                "error": f"{type(error).__name__}: {error}",
            }

        write_json(output_path, result)

    print(f"Creato output in: {args.output_directory}")


if __name__ == "__main__":
    main()