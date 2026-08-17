from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# CONFIGURAZIONE AMBIENTE (ANTI-DEADLOCK & SHM FIX)
# Deve essere eseguita PRIMA di importare vLLM o PyTorch
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Disabilita P2P per evitare blocchi hardware PCIe
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# PUNTO 3: Forza il metodo spawn e disabilita le stats per mitigare problemi di shared memory
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ora possiamo importare i moduli pesanti
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# =============================================================================
# COSTANTI GLOBALI
# =============================================================================
MODEL_ID = "google/gemma-4-12B-it"
SEMANTIC_DIR = Path("data/preliminar_analysis/entity/gemma-4-12B")
OUTPUT_DIR = Path("data/preliminar_analysis/event/gemma-4-12B")

MAX_MODEL_LEN = 32768
MAX_RETRIES = 3
SAFETY_MARGIN = 512
CHUNK_OVERLAP = 1

SEMANTIC_FIELDS = (
    "entities", "actions", "events", "spatial_relations",
    "state_changes", "temporal_relations", "causal_hypotheses",
)

# =============================================================================
# FUNZIONI DI UTILITÀ (I/O e Parsing)
# =============================================================================
def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Estrae e valida il JSON dall'output testuale dell'LLM."""
    if not text or not text.strip():
        raise ValueError("Risposta vuota dall'LLM.")

    cleaned = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start_idx = cleaned.find("{")
        if start_idx < 0:
            raise ValueError("La risposta non contiene l'apertura di un oggetto JSON '{'.")
        
        candidate = cleaned[start_idx:]
        try:
            from json_repair import repair_json
            repaired = repair_json(candidate)
            result = json.loads(repaired)
        except Exception as e:
            raise ValueError(f"JSON non recuperabile. Errore: {e}\nRAW:\n{cleaned[:2000]}")

    if not isinstance(result, dict):
        raise ValueError("Il JSON decodificato non è un dizionario/oggetto.")
    
    return result

def save_failed_raw(out_dir: Path, name: str, attempt: int, raw_text: str) -> Path:
    """Salva l'output fallato per debug."""
    path = out_dir / "_failed_raw" / f"{name}_attempt_{attempt}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_text or "<EMPTY RESPONSE>", encoding="utf-8")
    return path

# =============================================================================
# CLASSE INFERENCER (Gestione LLM)
# =============================================================================
class Inferencer:
    """Gestisce il caricamento del modello vLLM e la generazione dei prompt."""
    
    def __init__(self, model_id: str, max_new_tokens: int, gpu_utilization: float):
        self.max_input_tokens = MAX_MODEL_LEN - max_new_tokens - SAFETY_MARGIN
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        print(f"Caricamento {model_id} su 2 GPU (TP=2)...", flush=True)
        
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=2,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
            trust_remote_code=True,
            # PUNTO 2: enforce_eager è stato rimosso per evitare deadlock con TP>1
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def format_prompt(self, prompt_text: str) -> str:
        return self.processor.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def token_count(self, prompt_text: str) -> int:
        formatted = self.format_prompt(prompt_text)
        return len(self.processor.tokenizer.encode(formatted, add_special_tokens=False))

    def fits(self, prompt_text: str) -> bool:
        return self.token_count(prompt_text) <= self.max_input_tokens

    def generate(self, prompt_text: str) -> str:
        formatted = self.format_prompt(prompt_text)
        outputs = self.llm.generate([formatted], sampling_params=self.sampling, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

# =============================================================================
# LOGICA DI BUSINESS E PROMPT
# =============================================================================
def build_prompt(payload: Dict[str, Any]) -> str:
    return (
        "Consolida i segmenti dell'analisi semantica precedente in una rappresentazione cronologica degli eventi.\n\n"
        "Regole:\n"
        "- usa esclusivamente le informazioni presenti nell'analisi semantica;\n"
        "- unisci soltanto i duplicati causati dalla sovrapposizione delle finestre;\n"
        "- mantieni separati eventi realmente distinti o ripetuti;\n"
        "- ricava i tempi soltanto dai segmenti forniti;\n"
        "- non inventare intenzioni, cause, emozioni o azioni mancanti;\n"
        "- usa evidence_type=\"inferred\" soltanto se l'input contiene esplicitamente un'inferenza;\n"
        "- assegna gli ID E0001, E0002, ... in ordine temporale;\n"
        "- restituisci esclusivamente JSON valido.\n\n"
        "Schema:\n"
        "{\n"
        '  "events": [\n'
        "    {\n"
        '      "event_id": "E0001",\n'
        '      "description": "string",\n'
        '      "start_time": 0.0,\n'
        '      "end_time": 0.0,\n'
        '      "participants": ["string"],\n'
        '      "evidence_segments": ["segment_0000"],\n'
        '      "evidence_type": "observed|inferred|uncertain",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "temporal_relations": [\n'
        "    {\n"
        '      "first_event": "E0001",\n'
        '      "relation": "before|after|overlaps|during|simultaneous",\n'
        '      "second_event": "E0002",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )

def build_merge_prompt(video_id: str, partial_results: List[Dict[str, Any]]) -> str:
    return (
        "Fondi le analisi parziali degli eventi dello stesso video in una singola rappresentazione cronologica.\n\n"
        "Regole:\n"
        "- usa esclusivamente gli eventi presenti negli input;\n"
        "- elimina i duplicati dovuti alla suddivisione dell'input;\n"
        "- non fondere eventi realmente distinti o ripetuti;\n"
        "- conserva tempi, partecipanti ed evidenze;\n"
        "- non inventare nuovi eventi o relazioni;\n"
        "- ordina gli eventi temporalmente;\n"
        "- assegna nuovamente ID E0001, E0002, ...;\n"
        "- aggiorna le relazioni temporali con i nuovi ID;\n"
        "- restituisci esclusivamente JSON valido.\n\n"
        f"VIDEO:\n{video_id}\n\n"
        f"INPUT:\n{json.dumps(partial_results, ensure_ascii=False, separators=(',', ':'))}"
    )

def compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    segments = []
    for segment in payload.get("segments", []):
        item = {
            "segment_id": segment.get("segment_id"),
            "start_time": segment.get("start_time"),
            "end_time": segment.get("end_time"),
        }
        item.update({k: segment[k] for k in SEMANTIC_FIELDS if segment.get(k)})
        segments.append(item)
    return {"id_video": payload.get("id_video"), "segments": segments}

def create_payload_chunks(payload: Dict[str, Any], infer: Inferencer) -> List[Dict[str, Any]]:
    segments = payload.get("segments", [])
    if not segments:
        return [payload]

    chunks = []
    current_segments = []

    for seg in segments:
        candidate_payload = {"id_video": payload.get("id_video"), "segments": current_segments + [seg]}
        
        if infer.fits(build_prompt(candidate_payload)):
            current_segments.append(seg)
        else:
            if not current_segments:
                raise ValueError(f"Il segmento {seg.get('segment_id')} supera la context window.")
            
            chunks.append({"id_video": payload.get("id_video"), "segments": current_segments})
            overlap = current_segments[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
            current_segments = overlap + [seg]

            if not infer.fits(build_prompt({"id_video": payload.get("id_video"), "segments": current_segments})):
                current_segments = [seg]

            if not infer.fits(build_prompt({"id_video": payload.get("id_video"), "segments": current_segments})):
                raise ValueError(f"Il segmento {seg.get('segment_id')} supera la context window.")

    if current_segments:
        chunks.append({"id_video": payload.get("id_video"), "segments": current_segments})

    return chunks

def generate_with_retry(infer: Inferencer, prompt: str, out_dir: Path, name: str) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        raw = ""
        try:
            print(f"    Tentativo {attempt}/{MAX_RETRIES}", flush=True)
            raw = infer.generate(prompt)
            result = extract_json_from_text(raw)
            print(f"    JSON valido ({len(raw)} chars)", flush=True)
            return result
        except Exception as e:
            last_err = e
            print(f"    [WARN] {type(e).__name__}: {e}", flush=True)
            fail_path = save_failed_raw(out_dir, name, attempt, raw)
            print(f"    RAW salvato in: {fail_path}", flush=True)
            
    raise RuntimeError(f"Fallito dopo {MAX_RETRIES} tentativi. Errore: {last_err}")

def merge_hierarchical(video_id: str, results: List[Dict[str, Any]], infer: Inferencer, out_dir: Path) -> Dict[str, Any]:
    level = 1
    current_results = results

    while len(current_results) > 1:
        groups = []
        current_group = []
        
        for res in current_results:
            candidate = current_group + [res]
            if infer.fits(build_merge_prompt(video_id, candidate)):
                current_group = candidate
            else:
                if not current_group:
                    raise ValueError(f"Un parziale di {video_id} supera la context window.")
                groups.append(current_group)
                current_group = [res]
        
        if current_group:
            groups.append(current_group)
            
        if len(groups) == len(current_results) and all(len(g) == 1 for g in groups):
            raise ValueError(f"Impossibile ridurre il merge per {video_id} (limite token).")

        next_level_results = []
        for idx, group in enumerate(groups, start=1):
            if len(group) == 1:
                next_level_results.append(group[0])
                continue
                
            prompt = build_merge_prompt(video_id, group)
            print(f"  Merge livello {level}, gruppo {idx}/{len(groups)}", flush=True)
            name = f"{video_id}_merge_L{level}_{idx:02d}"
            next_level_results.append(generate_with_retry(infer, prompt, out_dir, name))
            
        current_results = next_level_results
        level += 1

    return current_results[0]

def normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    result.setdefault("events", [])
    result.setdefault("temporal_relations", [])
    if not isinstance(result["events"], list):
        raise ValueError("'events' non è una lista.")
    if not isinstance(result["temporal_relations"], list):
        raise ValueError("'temporal_relations' non è una lista.")
    return result

# =============================================================================
# ESECUZIONE PRINCIPALE
# =============================================================================
def process_video(path: Path, infer: Inferencer, args: argparse.Namespace, index: int, total: int) -> bool:
    payload = read_json(path)
    video_id = payload.get("id_video") or path.stem.removesuffix("_semantic")
    output_path = args.output_directory / f"{video_id}_events.json"

    if output_path.exists() and not args.overwrite:
        print(f"[{index}/{total}] SKIP {video_id}", flush=True)
        return False

    print(f"\n[{index}/{total}] {video_id}", flush=True)
    
    semantic = compact_payload(payload)
    chunks = create_payload_chunks(semantic, infer)
    
    print(f"  Segmenti: {len(semantic['segments'])} | Chunk elaborati: {len(chunks)}", flush=True)

    partial_results = []
    for c_idx, chunk in enumerate(chunks, start=1):
        prompt = build_prompt(chunk)
        print(f"  Chunk {c_idx}/{len(chunks)} ({len(chunk['segments'])} segs)", flush=True)
        name = f"{video_id}_chunk_{c_idx:02d}"
        partial_results.append(generate_with_retry(infer, prompt, args.output_directory, name))

    if len(partial_results) == 1:
        final_result = partial_results[0]
    else:
        final_result = merge_hierarchical(video_id, partial_results, infer, args.output_directory)

    final_result = normalize_result(final_result)
    final_result.update({
        "id_video": video_id,
        "model": args.model,
        "source_semantic_file": str(path),
    })

    write_json(output_path, final_result)
    print(f"  Salvato: {output_path.name} | Eventi: {len(final_result['events'])}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Event extraction Gemma")
    parser.add_argument("semantic_directory", nargs="?", type=Path, default=SEMANTIC_DIR)
    parser.add_argument("output_directory", nargs="?", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.semantic_directory.glob("*_semantic.json"))
    if args.limit_videos > 0:
        files = files[:args.limit_videos]

    if not files:
        parser.error(f"Nessun file *_semantic.json trovato in {args.semantic_directory}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    infer = Inferencer(args.model, args.max_new_tokens, args.gpu_utilization)

    stats = {"completed": 0, "skipped": 0, "failed": 0}

    for idx, path in enumerate(files, start=1):
        try:
            completed = process_video(path, infer, args, idx, len(files))
            if completed:
                stats["completed"] += 1
            else:
                stats["skipped"] += 1
        except Exception:
            stats["failed"] += 1
            print(f"\n[ERROR] Elaborazione fallita per: {path.name}", flush=True)
            traceback.print_exc()

    print(
        "\n" + "="*40 + "\n"
        "ELABORAZIONE COMPLETATA\n"
        + "="*40 + "\n"
        f"Video totali: {len(files)}\n"
        f"Completati:   {stats['completed']}\n"
        f"Saltati:      {stats['skipped']}\n"
        f"Falliti:      {stats['failed']}\n"
        + "="*40,
        flush=True,
    )

if __name__ == "__main__":
    main()