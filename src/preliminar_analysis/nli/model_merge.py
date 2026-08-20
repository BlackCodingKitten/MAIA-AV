#!/usr/bin/env python3

"""
analysis_finalization.py

Finalizzazione della preliminary analysis di MAIA-AV.

La preliminary analysis è composta da tre livelli:

    1. Semantic / Entity extraction
    2. Event extraction
    3. Causal inference

Questo script NON utilizza:
    - domande MAIA
    - caption
    - foil
    - NLI
    - embedding
    - similarity matching

Il suo unico compito è consolidare, per ogni video e per ogni modello,
i risultati dei tre livelli di analisi in una rappresentazione finale
question-independent.

Output attesi:

    data/preliminar_analysis/final_results/gemma.json
    data/preliminar_analysis/final_results/gemma-4-12B.json
    data/preliminar_analysis/final_results/qwen.json
    data/preliminar_analysis/final_results/qwen-30B.json

Questi file vengono successivamente utilizzati dallo script NLI.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

ROOT = Path("data/preliminar_analysis")

ENTITY_ROOT = ROOT / "entity"
EVENT_ROOT = ROOT / "event"
CAUSAL_ROOT = ROOT / "causal"

FINAL_DIR = ROOT / "final_results"

MODELS = (
    "gemma",
    "gemma-4-12B",
    "qwen",
    "qwen-30B",
)

DEFAULT_EXPECTED_VIDEOS = 100


# ============================================================================
# UTILITY
# ============================================================================


def normalize_video_id(value: str) -> str:
    """
    Converte qualunque stringa contenente l'identificativo numerico
    di un video nel formato canonico:

        video1
        video001
        video001_semantic
        video001_events.json

    -> video001
    """

    value = str(value)

    match = re.search(
        r"video[_\-\s]*0*(\d+)",
        value,
        flags=re.IGNORECASE,
    )

    if match is None:
        match = re.search(r"(\d+)", value)

    if match is None:
        raise ValueError(
            f"Impossibile determinare il video ID da: {value!r}"
        )

    number = int(match.group(1))

    return f"video{number:03d}"


def read_json(path: Path) -> Any:
    """
    Carica un file JSON e produce errori leggibili in caso
    di file assente, vuoto o JSON malformato.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}"
        )

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Impossibile leggere il file {path}: {exc}"
        ) from exc

    if not raw.strip():
        raise RuntimeError(
            f"Il file JSON è vuoto: {path}"
        )

    try:
        return json.loads(raw)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "\n".join(
                [
                    f"JSON non valido: {path}",
                    f"Linea: {exc.lineno}",
                    f"Colonna: {exc.colno}",
                    f"Errore: {exc.msg}",
                ]
            )
        ) from exc


def write_json(path: Path, data: Any) -> None:
    """
    Scrive un JSON UTF-8 leggibile.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    except OSError as exc:
        raise RuntimeError(
            f"Impossibile scrivere {path}: {exc}"
        ) from exc


def contains_explicit_error(data: Any) -> Optional[str]:
    """
    Cerca un errore esplicito nel livello principale del JSON.

    Non considera un array vuoto come errore.

    Per esempio:

        "causal_relations": []

    è un risultato perfettamente valido: significa semplicemente
    che non sono state individuate relazioni causali.
    """

    if not isinstance(data, dict):
        return None

    if "error" in data and data["error"] not in (
        None,
        "",
        False,
        [],
        {},
    ):
        return str(data["error"])

    status = str(
        data.get("status", "")
    ).strip().lower()

    if status in {
        "error",
        "failed",
        "failure",
    }:
        return str(
            data.get(
                "message",
                data.get(
                    "reason",
                    f"status={status}",
                ),
            )
        )

    return None


def validate_analysis(
    data: Any,
    path: Path,
    analysis_type: str,
) -> None:
    """
    Verifica che un risultato di analisi sia utilizzabile.

    La funzione NON impone la presenza di entità, eventi o
    relazioni causali: l'assenza di tali elementi può essere
    semanticamente legittima.
    """

    if data is None:
        raise RuntimeError(
            f"{analysis_type}: contenuto nullo in {path}"
        )

    if not isinstance(
        data,
        (dict, list),
    ):
        raise RuntimeError(
            f"{analysis_type}: formato inatteso in {path}. "
            f"Trovato {type(data).__name__}, atteso dict o list."
        )

    error = contains_explicit_error(data)

    if error is not None:
        raise RuntimeError(
            "\n".join(
                [
                    f"{analysis_type}: il file contiene un errore.",
                    f"File: {path}",
                    f"Errore: {error}",
                ]
            )
        )


# ============================================================================
# INDICIZZAZIONE DEI FILE
# ============================================================================


def index_analysis_files(
    directory: Path,
    suffix: str,
    analysis_name: str,
) -> Dict[str, Path]:
    """
    Indicizza i file di un determinato livello.

    Esempio:

        directory:
            data/preliminar_analysis/event/qwen-30B

        suffix:
            "_events"

    produce:

        {
            "video001": Path(.../video001_events.json),
            "video002": Path(.../video002_events.json),
            ...
        }
    """

    if not directory.exists():
        raise RuntimeError(
            f"Directory {analysis_name} inesistente: {directory}"
        )

    if not directory.is_dir():
        raise RuntimeError(
            f"Il percorso non è una directory: {directory}"
        )

    pattern = f"*{suffix}.json"

    paths = sorted(
        directory.glob(pattern)
    )

    if not paths:
        raise RuntimeError(
            "\n".join(
                [
                    f"Nessun file {analysis_name} trovato.",
                    f"Directory: {directory}",
                    f"Pattern cercato: {pattern}",
                ]
            )
        )

    result: Dict[str, Path] = {}

    for path in paths:

        video_id = normalize_video_id(
            path.stem
        )

        if video_id in result:
            raise RuntimeError(
                "\n".join(
                    [
                        f"Duplicato {analysis_name} per {video_id}:",
                        f"  {result[video_id]}",
                        f"  {path}",
                    ]
                )
            )

        result[video_id] = path

    return result


# ============================================================================
# VALIDAZIONE COMPLETEZZA
# ============================================================================


def format_video_list(
    videos: Set[str],
    limit: int = 20,
) -> str:
    """
    Formatta una lista di video per i messaggi di errore.
    """

    ordered = sorted(videos)

    if len(ordered) <= limit:
        return ", ".join(ordered)

    shown = ", ".join(
        ordered[:limit]
    )

    remaining = len(ordered) - limit

    return (
        f"{shown}, ... "
        f"(+{remaining} altri)"
    )


def validate_sets(
    model: str,
    semantic_files: Dict[str, Path],
    event_files: Dict[str, Path],
    causal_files: Dict[str, Path],
    expected_videos: Optional[int],
) -> List[str]:
    """
    Controlla che semantic, event e causal contengano lo stesso
    insieme di video.

    Restituisce la lista ordinata degli ID validi.
    """

    semantic_ids = set(
        semantic_files.keys()
    )

    event_ids = set(
        event_files.keys()
    )

    causal_ids = set(
        causal_files.keys()
    )

    union = (
        semantic_ids
        | event_ids
        | causal_ids
    )

    errors: List[str] = []

    missing_semantic = (
        union - semantic_ids
    )

    missing_event = (
        union - event_ids
    )

    missing_causal = (
        union - causal_ids
    )

    if missing_semantic:
        errors.append(
            "Semantic mancanti: "
            + format_video_list(
                missing_semantic
            )
        )

    if missing_event:
        errors.append(
            "Event mancanti: "
            + format_video_list(
                missing_event
            )
        )

    if missing_causal:
        errors.append(
            "Causal mancanti: "
            + format_video_list(
                missing_causal
            )
        )

    intersection = (
        semantic_ids
        & event_ids
        & causal_ids
    )

    if (
        expected_videos is not None
        and len(intersection) != expected_videos
    ):
        errors.append(
            f"Numero di video completi inatteso: "
            f"{len(intersection)} "
            f"(attesi {expected_videos})."
        )

    if errors:
        message = [
            "",
            "=" * 80,
            f"ANALISI INCOMPLETE PER IL MODELLO: {model}",
            "=" * 80,
        ]

        message.extend(
            f"- {error}"
            for error in errors
        )

        message.extend(
            [
                "",
                f"Semantic disponibili: {len(semantic_ids)}",
                f"Event disponibili:    {len(event_ids)}",
                f"Causal disponibili:   {len(causal_ids)}",
                f"Video completi:       {len(intersection)}",
            ]
        )

        raise RuntimeError(
            "\n".join(message)
        )

    return sorted(intersection)


# ============================================================================
# FINALIZZAZIONE DI UN SINGOLO VIDEO
# ============================================================================


def finalize_video(
    video_id: str,
    semantic_path: Path,
    event_path: Path,
    causal_path: Path,
) -> Dict[str, Any]:
    """
    Carica e consolida i tre livelli dell'analisi preliminare.

    È intenzionalmente una semplice composizione strutturale.
    Nessuna informazione viene selezionata in base alla domanda.
    """

    semantic_data = read_json(
        semantic_path
    )

    event_data = read_json(
        event_path
    )

    causal_data = read_json(
        causal_path
    )

    validate_analysis(
        semantic_data,
        semantic_path,
        "Semantic analysis",
    )

    validate_analysis(
        event_data,
        event_path,
        "Event analysis",
    )

    validate_analysis(
        causal_data,
        causal_path,
        "Causal analysis",
    )

    return {
        "video_id": video_id,

        "semantic_analysis": (
            semantic_data
        ),

        "event_analysis": (
            event_data
        ),

        "causal_analysis": (
            causal_data
        ),
    }


# ============================================================================
# FINALIZZAZIONE DI UN MODELLO
# ============================================================================


def finalize_model(
    model: str,
    expected_videos: Optional[int],
) -> Path:
    """
    Finalizza tutti i video per uno specifico modello.
    """

    print()
    print("=" * 80)
    print(
        f"FINALIZZAZIONE: {model}"
    )
    print("=" * 80)

    semantic_dir = (
        ENTITY_ROOT / model
    )

    event_dir = (
        EVENT_ROOT / model
    )

    causal_dir = (
        CAUSAL_ROOT / model
    )

    print(
        f"Semantic: {semantic_dir}"
    )

    print(
        f"Event:    {event_dir}"
    )

    print(
        f"Causal:   {causal_dir}"
    )

    print()

    semantic_files = (
        index_analysis_files(
            directory=semantic_dir,
            suffix="_semantic",
            analysis_name="semantic",
        )
    )

    event_files = (
        index_analysis_files(
            directory=event_dir,
            suffix="_events",
            analysis_name="event",
        )
    )

    causal_files = (
        index_analysis_files(
            directory=causal_dir,
            suffix="_causal",
            analysis_name="causal",
        )
    )

    video_ids = validate_sets(
        model=model,
        semantic_files=semantic_files,
        event_files=event_files,
        causal_files=causal_files,
        expected_videos=expected_videos,
    )

    print(
        f"Video completi trovati: "
        f"{len(video_ids)}"
    )

    videos: List[
        Dict[str, Any]
    ] = []

    failures: List[str] = []

    total = len(video_ids)

    for index, video_id in enumerate(
        video_ids,
        start=1,
    ):

        print(
            f"[{model}] "
            f"{index:03d}/{total:03d} "
            f"{video_id}"
        )

        try:
            result = finalize_video(
                video_id=video_id,
                semantic_path=semantic_files[
                    video_id
                ],
                event_path=event_files[
                    video_id
                ],
                causal_path=causal_files[
                    video_id
                ],
            )

            videos.append(
                result
            )

        except Exception as exc:
            failures.append(
                f"{video_id}: {exc}"
            )

    if failures:
        message = [
            "",
            "=" * 80,
            f"FINALIZZAZIONE FALLITA: {model}",
            "=" * 80,
            f"Video falliti: {len(failures)}",
            "",
        ]

        message.extend(
            f"- {failure}"
            for failure in failures
        )

        raise RuntimeError(
            "\n".join(message)
        )

    output = {
        "model": model,
        "num_videos": len(videos),
        "videos": videos,
    }

    output_path = (
        FINAL_DIR
        / f"{model}.json"
    )

    write_json(
        output_path,
        output,
    )

    print()
    print(
        f"[OK] {model}"
    )

    print(
        f"[OK] Video finalizzati: "
        f"{len(videos)}"
    )

    print(
        f"[OK] Output: "
        f"{output_path}"
    )

    return output_path


# ============================================================================
# CONTROLLO FINALE DEGLI OUTPUT
# ============================================================================


def validate_final_output(
    path: Path,
    expected_model: str,
    expected_videos: Optional[int],
) -> None:
    """
    Riapre il JSON appena prodotto e controlla che possa essere
    utilizzato dallo script NLI.
    """

    data = read_json(
        path
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Output finale non valido: {path}"
        )

    if (
        data.get("model")
        != expected_model
    ):
        raise RuntimeError(
            f"Modello errato in {path}: "
            f"{data.get('model')!r}"
        )

    videos = data.get(
        "videos"
    )

    if not isinstance(
        videos,
        list,
    ):
        raise RuntimeError(
            f"Campo 'videos' assente o invalido in {path}"
        )

    if (
        expected_videos is not None
        and len(videos)
        != expected_videos
    ):
        raise RuntimeError(
            f"{path}: contiene "
            f"{len(videos)} video, "
            f"attesi {expected_videos}."
        )

    seen: Set[str] = set()

    for item in videos:

        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                f"Elemento video non valido in {path}"
            )

        video_id = item.get(
            "video_id"
        )

        if not video_id:
            raise RuntimeError(
                f"Elemento senza video_id in {path}"
            )

        normalized = (
            normalize_video_id(
                video_id
            )
        )

        if normalized in seen:
            raise RuntimeError(
                f"video_id duplicato in {path}: "
                f"{normalized}"
            )

        seen.add(
            normalized
        )

        required_sections = (
            "semantic_analysis",
            "event_analysis",
            "causal_analysis",
        )

        for section in required_sections:

            if section not in item:
                raise RuntimeError(
                    f"{normalized}: sezione "
                    f"{section!r} mancante "
                    f"in {path}"
                )


# ============================================================================
# ARGOMENTI
# ============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Consolida semantic extraction, event extraction "
            "e causal inference nei file finali della "
            "preliminary analysis."
        )
    )

    parser.add_argument(
        "--model",
        choices=MODELS,
        default=None,
        help=(
            "Finalizza soltanto un modello. "
            "Se omesso vengono elaborati tutti."
        ),
    )

    parser.add_argument(
        "--expected-videos",
        type=int,
        default=DEFAULT_EXPECTED_VIDEOS,
        help=(
            "Numero atteso di video completi per modello. "
            "Default: 100."
        ),
    )

    parser.add_argument(
        "--no-count-check",
        action="store_true",
        help=(
            "Non richiede esattamente --expected-videos video. "
            "Semantic, event e causal devono comunque essere "
            "presenti per gli stessi video."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    args = parse_args()

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_videos = (
        None
        if args.no_count_check
        else args.expected_videos
    )

    models = (
        (args.model,)
        if args.model
        else MODELS
    )

    generated: List[
        Path
    ] = []

    print()
    print("=" * 80)
    print(
        "MAIA-AV PRELIMINARY ANALYSIS FINALIZATION"
    )
    print("=" * 80)

    print(
        f"Modelli: "
        f"{', '.join(models)}"
    )

    if expected_videos is None:
        print(
            "Controllo numero video: disabilitato"
        )
    else:
        print(
            f"Video attesi per modello: "
            f"{expected_videos}"
        )

    try:

        for model in models:

            path = finalize_model(
                model=model,
                expected_videos=expected_videos,
            )

            validate_final_output(
                path=path,
                expected_model=model,
                expected_videos=expected_videos,
            )

            generated.append(
                path
            )

    except Exception as exc:

        print(
            "",
            file=sys.stderr,
        )

        print(
            "=" * 80,
            file=sys.stderr,
        )

        print(
            "ERRORE DURANTE LA FINALIZZAZIONE",
            file=sys.stderr,
        )

        print(
            "=" * 80,
            file=sys.stderr,
        )

        print(
            str(exc),
            file=sys.stderr,
        )

        sys.exit(1)

    print()
    print("=" * 80)
    print(
        "FINALIZZAZIONE COMPLETATA"
    )
    print("=" * 80)

    for path in generated:
        print(
            f"[OK] {path}"
        )

    print()

    if set(models) == set(MODELS):

        expected_outputs = [
            FINAL_DIR
            / f"{model}.json"
            for model in MODELS
        ]

        if all(
            path.exists()
            for path in expected_outputs
        ):

            print(
                "Tutti i quattro file richiesti "
                "dallo script NLI sono presenti:"
            )

            for path in expected_outputs:
                print(
                    f"  - {path}"
                )

        else:

            print(
                "ATTENZIONE: non tutti i file "
                "richiesti dall'NLI sono presenti."
            )


if __name__ == "__main__":
    main()