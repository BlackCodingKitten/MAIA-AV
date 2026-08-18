from __future__ import annotations

import os

# =============================================================================
# CONFIGURAZIONE AMBIENTE
#
# DEVE ESSERE ESEGUITA PRIMA DI IMPORTARE:
# - torch
# - transformers
# - vllm
# =============================================================================

# -----------------------------------------------------------------------------
# GPU
# -----------------------------------------------------------------------------
#
# Vengono utilizzate esclusivamente le GPU fisiche 0 e 7.
# Nel processo saranno visibili come:
#
#   GPU fisica 0 -> cuda:0
#   GPU fisica 7 -> cuda:1
#
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


# -----------------------------------------------------------------------------
# MULTIPROCESSING vLLM
# -----------------------------------------------------------------------------

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Comunicazione locale tra i worker.
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# Disabilita statistiche vLLM.
os.environ["VLLM_NO_USAGE_STATS"] = "1"


# -----------------------------------------------------------------------------
# FLASHINFER / ALL-REDUCE
# -----------------------------------------------------------------------------

# Non utilizzare FlashInfer per il sampling.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# Non utilizzare symmetric-memory all-reduce.
os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"

# Evita NCCL symmetric memory, se disponibile nella versione installata.
os.environ["VLLM_USE_NCCL_SYMM_MEM"] = "0"


# -----------------------------------------------------------------------------
# NCCL
# -----------------------------------------------------------------------------

# Niente comunicazione P2P diretta tra GPU.
os.environ["NCCL_P2P_DISABLE"] = "1"

# Niente shared memory come fallback NCCL.
os.environ["NCCL_SHM_DISABLE"] = "1"

# Niente InfiniBand.
os.environ["NCCL_IB_DISABLE"] = "1"

# Comunicazione NCCL attraverso socket locali.
os.environ["NCCL_SOCKET_IFNAME"] = "lo"


# -----------------------------------------------------------------------------
# CPU
# -----------------------------------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# =============================================================================
# IMPORT
# =============================================================================

import argparse
import json
import sys
import traceback

from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


# =============================================================================
# COSTANTI GLOBALI
# =============================================================================

MODEL_ID = "google/gemma-4-12B-it"

SEMANTIC_DIR = Path(
    "data/preliminar_analysis/entity/gemma-4-12B"
)

OUTPUT_DIR = Path(
    "data/preliminar_analysis/event/gemma-4-12B"
)

MAX_MODEL_LEN = 32768

# Numero massimo di nuove token generate per l'event extraction.
MAX_NEW_TOKENS = 1800

# Retry soltanto per errori recuperabili:
# output malformati, JSON non valido, ecc.
MAX_RETRIES = 3

# Margine per evitare di saturare completamente la context window.
SAFETY_MARGIN = 512

# Numero di segmenti sovrapposti fra chunk consecutivi.
CHUNK_OVERLAP = 1


SEMANTIC_FIELDS = (
    "entities",
    "actions",
    "events",
    "spatial_relations",
    "state_changes",
    "temporal_relations",
    "causal_hypotheses",
)


# =============================================================================
# ECCEZIONI
# =============================================================================


class EngineFatalError(RuntimeError):
    """
    Errore irreversibile dell'EngineCore vLLM.

    Una volta morto EngineCore, la stessa istanza LLM non può più essere
    riutilizzata. Non ha quindi senso effettuare retry sulla stessa istanza.
    """

    pass


# =============================================================================
# FUNZIONI DI UTILITÀ
# =============================================================================


def read_json(path: Path) -> Dict[str, Any]:
    """
    Legge un file JSON UTF-8.
    """

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    data: Dict[str, Any],
) -> None:
    """
    Scrive un dizionario come JSON UTF-8 indentato.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def is_engine_dead_exception(
    exc: BaseException,
) -> bool:
    """
    Determina se l'eccezione deriva dalla morte dell'EngineCore vLLM.

    Il controllo viene effettuato anche sulla catena __cause__/__context__,
    perché EngineDeadError può essere incapsulato da altre eccezioni.
    """

    current: BaseException | None = exc

    visited = set()

    while current is not None:

        current_id = id(current)

        if current_id in visited:
            break

        visited.add(current_id)

        class_name = current.__class__.__name__
        message = str(current)

        if class_name == "EngineDeadError":
            return True

        if "EngineCore encountered an issue" in message:
            return True

        if "EngineCore encountered a fatal error" in message:
            return True

        if "Engine core is dead" in message:
            return True

        if "EngineDeadError" in message:
            return True

        if current.__cause__ is not None:
            current = current.__cause__

        elif current.__context__ is not None:
            current = current.__context__

        else:
            current = None

    return False


# =============================================================================
# JSON PARSING
# =============================================================================


def extract_json_from_text(
    text: str,
) -> Dict[str, Any]:
    """
    Estrae un oggetto JSON dalla risposta generata dal modello.

    Strategia:
    1. rimuove eventuali markdown fence
    2. prova json.loads direttamente
    3. individua il primo oggetto JSON
    4. prova json_repair come fallback
    """

    if not text or not text.strip():
        raise ValueError(
            "Risposta vuota dall'LLM."
        )

    cleaned = (
        text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    # -------------------------------------------------------------------------
    # Primo tentativo: JSON già valido
    # -------------------------------------------------------------------------

    try:
        result = json.loads(cleaned)

        if not isinstance(result, dict):
            raise ValueError(
                "Il JSON decodificato non è un oggetto."
            )

        return result

    except json.JSONDecodeError:
        pass

    # -------------------------------------------------------------------------
    # Secondo tentativo: estrazione dell'oggetto JSON
    # -------------------------------------------------------------------------

    start_idx = cleaned.find("{")

    if start_idx < 0:
        raise ValueError(
            "La risposta non contiene l'apertura "
            "di un oggetto JSON '{'."
        )

    end_idx = cleaned.rfind("}")

    # Se troviamo sia apertura che chiusura, proviamo prima
    # il candidato delimitato.
    if end_idx >= start_idx:

        candidate = cleaned[
            start_idx:end_idx + 1
        ]

        try:
            result = json.loads(candidate)

            if not isinstance(result, dict):
                raise ValueError(
                    "Il JSON decodificato non è un oggetto."
                )

            return result

        except json.JSONDecodeError:
            pass

    # -------------------------------------------------------------------------
    # Terzo tentativo: json_repair
    # -------------------------------------------------------------------------

    candidate = cleaned[start_idx:]

    try:
        from json_repair import repair_json

    except ImportError as exc:
        raise ValueError(
            "Output JSON non valido e libreria "
            "'json_repair' non installata.\n"
            f"RAW:\n{cleaned[:2000]}"
        ) from exc

    try:

        repaired = repair_json(
            candidate
        )

        result = json.loads(
            repaired
        )

    except Exception as exc:

        raise ValueError(
            "JSON non recuperabile dopo json_repair.\n"
            f"Errore: {exc}\n"
            f"RAW:\n{cleaned[:2000]}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "Il JSON riparato non è un oggetto."
        )

    return result


# =============================================================================
# DEBUG OUTPUT
# =============================================================================


def save_failed_raw(
    out_dir: Path,
    name: str,
    attempt: int,
    raw_text: str,
) -> Path:
    """
    Salva la risposta RAW che non è stato possibile interpretare.
    """

    failed_dir = (
        out_dir
        / "_failed_raw"
    )

    failed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        failed_dir
        / f"{name}_attempt_{attempt}.txt"
    )

    path.write_text(
        raw_text
        if raw_text
        else "<EMPTY RESPONSE>",
        encoding="utf-8",
    )

    return path


# =============================================================================
# INFERENCER
# =============================================================================


class Inferencer:
    """
    Wrapper text-only per Gemma 4 12B utilizzato nella fase Event della
    Preliminary Analysis.

    Non riceve:
    - video
    - immagini
    - audio
    - trascrizioni

    Riceve esclusivamente l'output JSON prodotto dalla fase Semantic.
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        gpu_utilization: float,
    ):

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.gpu_utilization = gpu_utilization

        self.max_input_tokens = (
            MAX_MODEL_LEN
            - max_new_tokens
            - SAFETY_MARGIN
        )

        # ---------------------------------------------------------------------
        # PROCESSOR
        # ---------------------------------------------------------------------

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        print(
            "\n"
            "========================================\n"
            "CARICAMENTO MODELLO\n"
            "========================================\n"
            f"Model:            {model_id}\n"
            f"Tensor parallel:  2\n"
            f"Max model length: {MAX_MODEL_LEN}\n"
            f"Max new tokens:   {max_new_tokens}\n"
            f"GPU utilization:  {gpu_utilization}\n"
            "========================================",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # vLLM
        # ---------------------------------------------------------------------

        self.llm = LLM(
            model=model_id,

            dtype="bfloat16",

            # -----------------------------------------------------------------
            # DISTRIBUZIONE SU DUE GPU
            # -----------------------------------------------------------------

            tensor_parallel_size=2,

            # Usa multiprocessing locale.
            distributed_executor_backend="mp",

            # -----------------------------------------------------------------
            # MEMORIA
            # -----------------------------------------------------------------

            gpu_memory_utilization=gpu_utilization,

            max_model_len=MAX_MODEL_LEN,

            # Una sola richiesta per volta.
            max_num_seqs=1,

            # -----------------------------------------------------------------
            # STABILITÀ
            # -----------------------------------------------------------------

            # Disabilita il custom all-reduce vLLM.
            # Il tensor parallel utilizzerà NCCL con la configurazione
            # definita prima degli import.
            disable_custom_all_reduce=True,

            # Niente CUDA Graph.
            #
            # Questa Preliminary Analysis privilegia stabilità e
            # prevedibilità rispetto al throughput.
            enforce_eager=True,

            trust_remote_code=True,
        )

        # ---------------------------------------------------------------------
        # GENERATION PARAMETERS
        # ---------------------------------------------------------------------

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

    # =========================================================================
    # CHAT TEMPLATE
    # =========================================================================

    def format_prompt(
        self,
        prompt_text: str,
    ) -> str:

        messages = [
            {
                "role": "user",
                "content": prompt_text,
            }
        ]

        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # =========================================================================
    # TOKEN COUNT
    # =========================================================================

    def token_count(
        self,
        prompt_text: str,
    ) -> int:

        formatted = self.format_prompt(
            prompt_text
        )

        encoded = self.processor.tokenizer.encode(
            formatted,
            add_special_tokens=False,
        )

        return len(encoded)

    def fits(
        self,
        prompt_text: str,
    ) -> bool:

        return (
            self.token_count(prompt_text)
            <= self.max_input_tokens
        )

    # =========================================================================
    # GENERATION
    # =========================================================================

    def generate(
        self,
        prompt_text: str,
    ) -> str:

        formatted = self.format_prompt(
            prompt_text
        )

        outputs = self.llm.generate(
            [formatted],
            sampling_params=self.sampling,
            use_tqdm=False,
        )

        if not outputs:
            raise RuntimeError(
                "vLLM non ha restituito alcun risultato."
            )

        if not outputs[0].outputs:
            raise RuntimeError(
                "vLLM non ha restituito alcuna generazione."
            )

        return (
            outputs[0]
            .outputs[0]
            .text
            .strip()
        )


# =============================================================================
# PROMPT: EVENT EXTRACTION
# =============================================================================


def build_prompt(
    payload: Dict[str, Any],
) -> str:

    return (
        "Consolida i segmenti dell'analisi semantica precedente "
        "in una rappresentazione cronologica degli eventi.\n\n"

        "Regole:\n"

        "- usa esclusivamente le informazioni presenti "
        "nell'analisi semantica;\n"

        "- unisci soltanto i duplicati causati dalla "
        "sovrapposizione delle finestre;\n"

        "- mantieni separati eventi realmente distinti "
        "o ripetuti;\n"

        "- ricava i tempi soltanto dai segmenti forniti;\n"

        "- non inventare intenzioni, cause, emozioni "
        "o azioni mancanti;\n"

        "- usa evidence_type=\"inferred\" soltanto se "
        "l'input contiene esplicitamente un'inferenza;\n"

        "- assegna gli ID E0001, E0002, ... "
        "in ordine temporale;\n"

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

        '      "evidence_type": '
        '"observed|inferred|uncertain",\n'

        '      "confidence": 0.0\n'

        "    }\n"

        "  ],\n"

        '  "temporal_relations": [\n'

        "    {\n"

        '      "first_event": "E0001",\n'

        '      "relation": '
        '"before|after|overlaps|during|simultaneous",\n'

        '      "second_event": "E0002",\n'

        '      "confidence": 0.0\n'

        "    }\n"

        "  ]\n"

        "}\n\n"

        "INPUT:\n"

        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


# =============================================================================
# PROMPT: MERGE
# =============================================================================


def build_merge_prompt(
    video_id: str,
    partial_results: List[Dict[str, Any]],
) -> str:

    return (
        "Fondi le analisi parziali degli eventi dello stesso "
        "video in una singola rappresentazione cronologica.\n\n"

        "Regole:\n"

        "- usa esclusivamente gli eventi presenti negli input;\n"

        "- elimina i duplicati dovuti alla suddivisione "
        "dell'input;\n"

        "- non fondere eventi realmente distinti o ripetuti;\n"

        "- conserva tempi, partecipanti ed evidenze;\n"

        "- non inventare nuovi eventi o relazioni;\n"

        "- ordina gli eventi temporalmente;\n"

        "- assegna nuovamente ID E0001, E0002, ...;\n"

        "- aggiorna le relazioni temporali con i nuovi ID;\n"

        "- restituisci esclusivamente JSON valido.\n\n"

        f"VIDEO:\n{video_id}\n\n"

        "INPUT:\n"

        + json.dumps(
            partial_results,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


# =============================================================================
# COMPRESSIONE INPUT SEMANTICO
# =============================================================================


def compact_payload(
    payload: Dict[str, Any],
) -> Dict[str, Any]:

    compact_segments: List[Dict[str, Any]] = []

    for segment in payload.get(
        "segments",
        [],
    ):

        item: Dict[str, Any] = {
            "segment_id": segment.get(
                "segment_id"
            ),

            "start_time": segment.get(
                "start_time"
            ),

            "end_time": segment.get(
                "end_time"
            ),
        }

        for field in SEMANTIC_FIELDS:

            value = segment.get(field)

            if value:
                item[field] = value

        compact_segments.append(
            item
        )

    return {
        "id_video": payload.get(
            "id_video"
        ),

        "segments": compact_segments,
    }


# =============================================================================
# CHUNKING
# =============================================================================


def create_payload_chunks(
    payload: Dict[str, Any],
    infer: Inferencer,
) -> List[Dict[str, Any]]:

    segments = payload.get(
        "segments",
        [],
    )

    if not segments:
        return [payload]

    chunks: List[Dict[str, Any]] = []

    current_segments: List[
        Dict[str, Any]
    ] = []

    video_id = payload.get(
        "id_video"
    )

    for segment in segments:

        candidate_segments = (
            current_segments
            + [segment]
        )

        candidate_payload = {
            "id_video": video_id,
            "segments": candidate_segments,
        }

        # ---------------------------------------------------------------------
        # IL SEGMENTO ENTRA NEL CHUNK CORRENTE
        # ---------------------------------------------------------------------

        if infer.fits(
            build_prompt(
                candidate_payload
            )
        ):

            current_segments.append(
                segment
            )

            continue

        # ---------------------------------------------------------------------
        # IL CHUNK CORRENTE È PIENO
        # ---------------------------------------------------------------------

        if not current_segments:

            raise ValueError(
                f"Il segmento "
                f"{segment.get('segment_id')} "
                "supera da solo la context window."
            )

        chunks.append(
            {
                "id_video": video_id,
                "segments": current_segments,
            }
        )

        # ---------------------------------------------------------------------
        # OVERLAP
        # ---------------------------------------------------------------------

        if CHUNK_OVERLAP > 0:

            overlap = current_segments[
                -CHUNK_OVERLAP:
            ]

        else:

            overlap = []

        current_segments = (
            overlap
            + [segment]
        )

        overlap_payload = {
            "id_video": video_id,
            "segments": current_segments,
        }

        # Se overlap + nuovo segmento non entra,
        # eliminiamo l'overlap.
        if not infer.fits(
            build_prompt(
                overlap_payload
            )
        ):

            current_segments = [
                segment
            ]

        single_payload = {
            "id_video": video_id,
            "segments": current_segments,
        }

        if not infer.fits(
            build_prompt(
                single_payload
            )
        ):

            raise ValueError(
                f"Il segmento "
                f"{segment.get('segment_id')} "
                "supera da solo la context window."
            )

    # -------------------------------------------------------------------------
    # ULTIMO CHUNK
    # -------------------------------------------------------------------------

    if current_segments:

        chunks.append(
            {
                "id_video": video_id,
                "segments": current_segments,
            }
        )

    return chunks


# =============================================================================
# GENERAZIONE CON RETRY
# =============================================================================


def generate_with_retry(
    infer: Inferencer,
    prompt: str,
    out_dir: Path,
    name: str,
) -> Dict[str, Any]:

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        raw = ""

        print(
            f"    Tentativo "
            f"{attempt}/{MAX_RETRIES}",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # GENERAZIONE
        # ---------------------------------------------------------------------

        try:

            raw = infer.generate(
                prompt
            )

        except Exception as exc:

            # -----------------------------------------------------------------
            # ENGINECORE MORTO
            #
            # Non fare altri tentativi usando lo stesso Inferencer.
            # -----------------------------------------------------------------

            if is_engine_dead_exception(
                exc
            ):

                raise EngineFatalError(
                    "EngineCore vLLM terminato. "
                    "La stessa istanza non può essere "
                    "riutilizzata."
                ) from exc

            last_error = exc

            print(
                f"    [WARN] "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

            failed_path = save_failed_raw(
                out_dir,
                name,
                attempt,
                raw,
            )

            print(
                "    RAW salvato in: "
                f"{failed_path}",
                flush=True,
            )

            continue

        # ---------------------------------------------------------------------
        # PARSING JSON
        # ---------------------------------------------------------------------

        try:

            result = extract_json_from_text(
                raw
            )

            print(
                f"    JSON valido "
                f"({len(raw)} chars)",
                flush=True,
            )

            return result

        except Exception as exc:

            last_error = exc

            print(
                f"    [WARN] "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

            failed_path = save_failed_raw(
                out_dir,
                name,
                attempt,
                raw,
            )

            print(
                "    RAW salvato in: "
                f"{failed_path}",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # RETRY ESAURITI
    # -------------------------------------------------------------------------

    raise RuntimeError(
        f"Fallito dopo {MAX_RETRIES} tentativi. "
        f"Errore: {last_error}"
    )


# =============================================================================
# MERGE GERARCHICO
# =============================================================================


def merge_hierarchical(
    video_id: str,
    results: List[Dict[str, Any]],
    infer: Inferencer,
    out_dir: Path,
) -> Dict[str, Any]:

    if not results:
        return {
            "events": [],
            "temporal_relations": [],
        }

    if len(results) == 1:
        return results[0]

    level = 1

    current_results = results

    while len(current_results) > 1:

        groups: List[
            List[Dict[str, Any]]
        ] = []

        current_group: List[
            Dict[str, Any]
        ] = []

        # ---------------------------------------------------------------------
        # CREAZIONE GRUPPI CHE ENTRANO NELLA CONTEXT WINDOW
        # ---------------------------------------------------------------------

        for result in current_results:

            candidate = (
                current_group
                + [result]
            )

            prompt = build_merge_prompt(
                video_id,
                candidate,
            )

            if infer.fits(prompt):

                current_group.append(
                    result
                )

                continue

            if not current_group:

                raise ValueError(
                    f"Un risultato parziale "
                    f"di {video_id} supera "
                    "da solo la context window."
                )

            groups.append(
                current_group
            )

            current_group = [
                result
            ]

        if current_group:

            groups.append(
                current_group
            )

        # ---------------------------------------------------------------------
        # PROTEZIONE CONTRO LOOP INFINITO
        # ---------------------------------------------------------------------

        if (
            len(groups)
            == len(current_results)
            and all(
                len(group) == 1
                for group in groups
            )
        ):

            raise ValueError(
                f"Impossibile ridurre "
                f"il merge per {video_id}: "
                "ogni risultato occupa "
                "un gruppo indipendente."
            )

        # ---------------------------------------------------------------------
        # MERGE DEL LIVELLO CORRENTE
        # ---------------------------------------------------------------------

        next_level_results: List[
            Dict[str, Any]
        ] = []

        for group_idx, group in enumerate(
            groups,
            start=1,
        ):

            # Un singolo elemento non richiede
            # una nuova chiamata al modello.
            if len(group) == 1:

                next_level_results.append(
                    group[0]
                )

                continue

            prompt = build_merge_prompt(
                video_id,
                group,
            )

            print(
                f"  Merge livello {level}, "
                f"gruppo "
                f"{group_idx}/{len(groups)}",
                flush=True,
            )

            name = (
                f"{video_id}"
                f"_merge_L{level}"
                f"_{group_idx:02d}"
            )

            merged = generate_with_retry(
                infer,
                prompt,
                out_dir,
                name,
            )

            next_level_results.append(
                merged
            )

        current_results = (
            next_level_results
        )

        level += 1

    return current_results[0]


# =============================================================================
# NORMALIZZAZIONE RISULTATO
# =============================================================================


def normalize_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    result.setdefault(
        "events",
        [],
    )

    result.setdefault(
        "temporal_relations",
        [],
    )

    if not isinstance(
        result["events"],
        list,
    ):

        raise ValueError(
            "'events' non è una lista."
        )

    if not isinstance(
        result["temporal_relations"],
        list,
    ):

        raise ValueError(
            "'temporal_relations' "
            "non è una lista."
        )

    return result


# =============================================================================
# PROCESSAMENTO DI UN VIDEO
# =============================================================================


def process_video(
    path: Path,
    infer: Inferencer,
    args: argparse.Namespace,
    index: int,
    total: int,
) -> bool:

    # -------------------------------------------------------------------------
    # INPUT
    # -------------------------------------------------------------------------

    payload = read_json(
        path
    )

    video_id = (
        payload.get("id_video")
        or path.stem.removesuffix(
            "_semantic"
        )
    )

    output_path = (
        args.output_directory
        / f"{video_id}_events.json"
    )

    # -------------------------------------------------------------------------
    # SKIP FILE GIÀ COMPLETATI
    # -------------------------------------------------------------------------

    if (
        output_path.exists()
        and not args.overwrite
    ):

        print(
            f"[{index}/{total}] "
            f"SKIP {video_id}",
            flush=True,
        )

        return False

    print(
        f"\n[{index}/{total}] "
        f"{video_id}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # COMPRESSIONE INPUT
    # -------------------------------------------------------------------------

    semantic = compact_payload(
        payload
    )

    # -------------------------------------------------------------------------
    # CHUNKING
    # -------------------------------------------------------------------------

    chunks = create_payload_chunks(
        semantic,
        infer,
    )

    print(
        f"  Segmenti: "
        f"{len(semantic['segments'])}"
        f" | "
        f"Chunk elaborati: "
        f"{len(chunks)}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # EVENT EXTRACTION PER CHUNK
    # -------------------------------------------------------------------------

    partial_results: List[
        Dict[str, Any]
    ] = []

    for chunk_idx, chunk in enumerate(
        chunks,
        start=1,
    ):

        prompt = build_prompt(
            chunk
        )

        print(
            f"  Chunk "
            f"{chunk_idx}/{len(chunks)} "
            f"({len(chunk['segments'])} segs)",
            flush=True,
        )

        name = (
            f"{video_id}"
            f"_chunk_{chunk_idx:02d}"
        )

        partial = generate_with_retry(
            infer,
            prompt,
            args.output_directory,
            name,
        )

        partial_results.append(
            partial
        )

    # -------------------------------------------------------------------------
    # MERGE
    # -------------------------------------------------------------------------

    if len(partial_results) == 1:

        final_result = (
            partial_results[0]
        )

    else:

        final_result = merge_hierarchical(
            video_id,
            partial_results,
            infer,
            args.output_directory,
        )

    # -------------------------------------------------------------------------
    # NORMALIZZAZIONE
    # -------------------------------------------------------------------------

    final_result = normalize_result(
        final_result
    )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    final_result.update(
        {
            "id_video": video_id,

            "model": args.model,

            "source_semantic_file": str(
                path
            ),
        }
    )

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    write_json(
        output_path,
        final_result,
    )

    print(
        f"  Salvato: "
        f"{output_path.name}"
        f" | "
        f"Eventi: "
        f"{len(final_result['events'])}",
        flush=True,
    )

    return True


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Event extraction della Preliminary Analysis "
            "con Gemma 4 12B."
        )
    )

    parser.add_argument(
        "semantic_directory",
        nargs="?",
        type=Path,
        default=SEMANTIC_DIR,
        help=(
            "Directory contenente "
            "i file *_semantic.json."
        ),
    )

    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Directory di output "
            "per i file *_events.json."
        ),
    )

    parser.add_argument(
        "--model",
        default=MODEL_ID,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # RICERCA FILE
    # -------------------------------------------------------------------------

    files = sorted(
        args.semantic_directory.glob(
            "*_semantic.json"
        )
    )

    if args.limit_videos > 0:

        files = files[
            :args.limit_videos
        ]

    if not files:

        parser.error(
            "Nessun file *_semantic.json "
            "trovato in "
            f"{args.semantic_directory}"
        )

    # -------------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # -------------------------------------------------------------------------

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # MODELLO
    # -------------------------------------------------------------------------

    infer = Inferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        gpu_utilization=args.gpu_utilization,
    )

    # -------------------------------------------------------------------------
    # STATISTICHE
    # -------------------------------------------------------------------------

    stats = {
        "completed": 0,
        "skipped": 0,
        "failed": 0,
    }

    # -------------------------------------------------------------------------
    # PROCESSAMENTO
    # -------------------------------------------------------------------------

    for idx, path in enumerate(
        files,
        start=1,
    ):

        try:

            completed = process_video(
                path,
                infer,
                args,
                idx,
                len(files),
            )

            if completed:

                stats["completed"] += 1

            else:

                stats["skipped"] += 1

        # ---------------------------------------------------------------------
        # ENGINECORE MORTO
        # ---------------------------------------------------------------------

        except EngineFatalError:

            stats["failed"] += 1

            print(
                "\n"
                "========================================\n"
                "ENGINE vLLM TERMINATO\n"
                "========================================\n"
                f"File corrente: {path.name}\n\n"
                "EngineCore è morto e non può essere "
                "riutilizzato nello stesso processo.\n\n"
                "Gli output dei video completati sono "
                "già stati salvati.\n"
                "Rieseguendo lo script senza --overwrite, "
                "i file completati verranno saltati e "
                "l'elaborazione ripartirà dal primo "
                "mancante.\n"
                "========================================",
                flush=True,
            )

            traceback.print_exc()

            sys.exit(2)

        # ---------------------------------------------------------------------
        # ERRORE DEL SINGOLO VIDEO
        # ---------------------------------------------------------------------

        except Exception:

            stats["failed"] += 1

            print(
                "\n"
                "[ERROR] Elaborazione fallita per: "
                f"{path.name}",
                flush=True,
            )

            traceback.print_exc()

            # Gli errori normali di parsing/input non
            # compromettono EngineCore.
            # Il programma continua con il video successivo.

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    print(
        "\n"
        "========================================\n"
        "ELABORAZIONE COMPLETATA\n"
        "========================================\n"
        f"Video totali: {len(files)}\n"
        f"Completati:   {stats['completed']}\n"
        f"Saltati:      {stats['skipped']}\n"
        f"Falliti:      {stats['failed']}\n"
        "========================================",
        flush=True,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()