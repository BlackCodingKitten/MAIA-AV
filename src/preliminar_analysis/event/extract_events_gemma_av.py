#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

MODEL_ID = "google/gemma-3n-E4B-it"
SEMANTIC_DIR = Path("data/preliminar_analysis/entity/entity_semantic/gemma")
PREPROCESSING_DIR = Path("data/preliminar_analysis/preprocessing")
VIDEO_DIR = Path("data/video")
OUTPUT_DIR = Path("data/preliminar_analysis/event_analysis/gemma")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def find_video(directory: Path, video_id: str):
    return next((p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and p.stem == video_id), None)


def extract_audio(source: Path, start: float, end: float, output: Path) -> bool:
    result = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(max(0.1, end - start)),
        "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)
    ])
    return result.returncode == 0 and output.exists() and output.stat().st_size > 44


def silent_audio(path: Path, duration: float):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * max(1, int(duration * 16000)))


def build_prompt(segment: dict) -> str:
    semantic = {k: segment.get(k, []) for k in (
        "entities", "actions", "events", "spatial_relations",
        "state_changes", "temporal_relations", "causal_hypotheses"
    )}
    return f"""Analizza congiuntamente i frame e l'audio del segmento usando come contesto l'analisi semantica prodotta precedentemente dallo stesso modello.

Segmento: {segment["segment_id"]}
Intervallo: {segment["start_time"]:.3f}-{segment["end_time"]:.3f} secondi
Analisi semantica precedente:
{json.dumps(semantic, ensure_ascii=False)}

Estrai soltanto eventi osservabili o udibili. Non inventare intenzioni, emozioni, cause invisibili o conseguenze future.
Usa gli entity_id già presenti quando disponibili. Restituisci esclusivamente JSON valido:

{{
  "events": [
    {{
      "event_id": "E1",
      "description": "descrizione breve in italiano",
      "participants": ["e1"],
      "start_time": 0.0,
      "end_time": 0.0,
      "modalities": ["video", "audio"],
      "evidence_frames": [],
      "confidence": 0.0
    }}
  ],
  "temporal_relations": [
    {{"first_event": "E1", "relation": "before|after|overlaps|during|simultaneous", "second_event": "E2"}}
  ]
}}"""


def load_model(model_id: str):
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True
    ).eval()
    return model, AutoProcessor.from_pretrained(model_id)


def infer(model_data, frame_paths: list[Path], audio_path: Path, prompt: str, max_new_tokens: int):
    model, processor = model_data
    content = [{"type": "image", "url": str(path)} for path in frame_paths]
    content += [{"type": "audio", "audio": str(audio_path)}, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    generated = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Event extraction audiovisiva Gemma-3n dai segmenti semantici.")
    parser.add_argument("semantic_directory", nargs="?", type=Path, default=SEMANTIC_DIR)
    parser.add_argument("preprocessing_directory", nargs="?", type=Path, default=PREPROCESSING_DIR)
    parser.add_argument("video_directory", nargs="?", type=Path, default=VIDEO_DIR)
    parser.add_argument("output_directory", nargs="?", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=8000)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.semantic_directory.glob("*_semantic.json"))
    if args.limit_videos:
        files = files[:args.limit_videos]
    if not files:
        parser.error(f"Nessun *_semantic.json trovato in {args.semantic_directory}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    model_data = load_model(args.model)

    for i, semantic_path in enumerate(files, 1):
        payload = read_json(semantic_path)
        video_id = payload.get("id_video") or semantic_path.stem.removesuffix("_semantic")
        output_path = args.output_directory / f"{video_id}_events.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{i}/{len(files)}] SKIP {video_id}")
            continue

        source_video = find_video(args.video_directory, video_id)
        if source_video is None:
            print(f"[{i}/{len(files)}] Video non trovato: {video_id}")
            continue

        result = {"id_video": video_id, "model": args.model, "source_semantic_file": str(semantic_path), "segments": []}

        for j, segment in enumerate(payload.get("segments", []), 1):
            print(f"[{i}/{len(files)}] {video_id} - [{j}/{len(payload.get('segments', []))}] {segment['segment_id']} su CUDA:0")
            item = {
                "segment_id": segment["segment_id"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "events": [],
                "temporal_relations": [],
            }
            try:
                frame_paths = [
                    args.preprocessing_directory / video_id / "dense_frames" / name
                    for name in segment.get("input_frames", [])
                ]
                frame_paths = [p for p in frame_paths if p.exists()]
                if not frame_paths:
                    raise FileNotFoundError("Nessun frame semantico trovato.")

                with tempfile.TemporaryDirectory() as tmp:
                    audio_path = Path(tmp) / "segment.wav"
                    audio_present = extract_audio(source_video, segment["start_time"], segment["end_time"], audio_path)
                    if not audio_present:
                        silent_audio(audio_path, segment["end_time"] - segment["start_time"])
                    raw = infer(model_data, frame_paths, audio_path, build_prompt(segment), args.max_new_tokens)

                data = extract_json(raw)
                item["events"] = data.get("events", [])
                item["temporal_relations"] = data.get("temporal_relations", [])
                item["audio_originale_presente"] = audio_present
            except Exception as error:
                item["error"] = f"{type(error).__name__}: {error}"

            result["segments"].append(item)
            write_json(output_path, result)

    print(f"Creato output in: {args.output_directory}")


if __name__ == "__main__":
    main()