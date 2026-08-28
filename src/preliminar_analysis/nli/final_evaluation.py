from pathlib import Path
from typing import Literal
import json

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field


MODEL = "gpt-4o-2024-08-06"
ANALYZED_MODEL = "qwen"

ROOT = Path("data/preliminar_analysis/final_results")
QUESTIONS = Path("data/vsv/question_classification.csv")
NLI_DIR = ROOT / "nli"

client = OpenAI()


PROMPT = """
Sei un valutatore diagnostico della comprensione visuale di un modello multimodale.

Non devi rispondere alla domanda.

Devi stabilire quanto delle informazioni NECESSARIE per poter rispondere alla
domanda sia stato effettivamente recuperato dal modello attraverso la sola
analisi dei frame del video.

Riceverai:

DOMANDA
La domanda originale.

REQUISITI DELLA DOMANDA
Una descrizione strutturata di ciò che sarebbe necessario recuperare dal video:
- entity: entità, referenti o attributi necessari
- event: eventi, stati o ancore temporali necessari
- inference: operazione inferenziale necessaria
- principal_label e auxiliary_labels: tipo di ragionamento richiesto
- objective_criterion: criterio operativo della domanda
- evidence_scope ed evidence_units: quantità e distribuzione dell'evidenza
  necessaria

ANALISI DEL MODELLO
La rappresentazione prodotta esclusivamente dai frame:
- semantic_analysis: entità, attributi, azioni, relazioni spaziali,
  cambiamenti di stato e osservazioni locali
- event_analysis: eventi consolidati e relazioni temporali
- causal_analysis: relazioni causali o motivazionali inferite

Il tuo obiettivo NON è verificare se nella rappresentazione compare
letteralmente il testo della domanda.

Devi effettuare un confronto semantico e inferenziale.

Valuta tre aspetti.

1. ENTITÀ
Quanto sono state recuperate le entità, i referenti, gli oggetti, gli attributi
o le localizzazioni richieste dal campo entity.

2. EVENTI
Quanto sono stati recuperati gli eventi, stati, azioni o ancore temporali
richiesti dal campo event.

3. INFERENZA
Quanto la rappresentazione disponibile permette di eseguire l'operazione
descritta dal campo inference.

Per l'inferenza puoi combinare coerentemente informazioni provenienti dai
diversi livelli dell'analisi.

Esempi:
- una relazione spaziale può derivare da entità + spatial_relations;
- una relazione prima/dopo può essere ricostruita da eventi, timestamp e
  temporal_relations;
- un cambiamento può essere ricostruito confrontando stati in intervalli diversi;
- una relazione causale può essere supportata da causal_analysis o da una
  catena di eventi sufficientemente esplicita.

REGOLE

- Accetta sinonimi, parafrasi e descrizioni semanticamente equivalenti.
- Non richiedere corrispondenza lessicale esatta.
- Gli ID delle entità possono cambiare tra segmenti: ragiona sul loro significato.
- Informazioni distribuite in segmenti differenti possono essere integrate.
- Usa timestamp e intervalli quando sono rilevanti.
- La semplice successione temporale non implica causalità.
- Non usare conoscenza esterna per inventare informazioni mancanti.
- Non attribuire dialoghi, intenzioni, cause o stati mentali che non siano
  supportati dall'analisi.
- Un elemento con confidence 0 non costituisce evidenza utile.
- Una confidence alta aumenta l'affidabilità dell'elemento ma non sostituisce
  la coerenza semantica.
- Informazioni irrilevanti non aumentano il punteggio.
- Non penalizzare una domanda perché è strutturalmente complessa:
  valuta soltanto quanto dei suoi requisiti specifici è stato recuperato.
- Non cercare di indovinare la risposta corretta alla domanda.

SCORE

Assegna score_entita, score_eventi e score_inferenza da 0 a 100.

0 significa che il requisito non è stato recuperato.
100 significa che il requisito è completamente e chiaramente disponibile.

Assegna poi score_comprensione da 0 a 100.

Lo score_comprensione rappresenta quanto il modello abbia complessivamente
costruito una rappresentazione visuale sufficiente per affrontare la domanda.

Usa come riferimento:

0-19:
quasi nessun requisito utile recuperato.

20-39:
riconosciuto il contesto generale o alcune entità, ma mancano elementi
fondamentali.

40-59:
copertura parziale sostanziale, ma manca almeno un prerequisito decisivo.

60-79:
quasi tutti i prerequisiti sono disponibili, con alcune lacune o incertezze.

80-94:
la rappresentazione contiene informazioni sufficienti per affrontare
seriamente la domanda.

95-100:
tutti i prerequisiti rilevanti sono chiaramente recuperati e coerenti.

Lo score complessivo NON deve essere una semplice media meccanica.
Un requisito decisivo mancante deve ridurre lo score anche se gli altri
elementi sono presenti.

Classifica infine la domanda come:

- insufficiente
- parziale
- sufficiente
- completa

Fornisci una motivazione breve basata esclusivamente sui requisiti recuperati
e mancanti.
"""


class Valutazione(BaseModel):
    score_entita: int = Field(ge=0, le=100)
    score_eventi: int = Field(ge=0, le=100)
    score_inferenza: int = Field(ge=0, le=100)
    score_comprensione: int = Field(ge=0, le=100)

    livello: Literal[
        "insufficiente",
        "parziale",
        "sufficiente",
        "completa",
    ]

    evidenza_recuperata: list[str]
    evidenza_mancante: list[str]
    motivazione: str


def clean(value):
    if isinstance(value, dict):
        if float(value.get("confidence", 1) or 0) == 0:
            return None

        return {
            key: cleaned
            for key, item in value.items()
            if key not in {
                "input_frames",
                "evidence_frames",
                "configuration",
                "errors",
                "source_semantic_file",
                "source_event_file",
            }
            and (cleaned := clean(item)) not in (None, [], {})
        }

    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := clean(item)) not in (None, [], {})
        ]

    return value


def load_analysis() -> dict[str, dict]:
    path = ROOT / f"{ANALYZED_MODEL}.json"

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return {
        video["video_id"]: clean(video)
        for video in data["videos"]
    }


def evaluate(
    question: pd.Series,
    analysis: dict,
) -> Valutazione:

    fields = [
        "question_text",
        "principal_dimension",
        "diagnostic_label",
        "complexity_level",
        "principal_label",
        "auxiliary_labels",
        "entity",
        "event",
        "inference",
        "objective_criterion",
        "evidence_scope",
        "evidence_units",
        "information_explicitness",
        "temporal_dependency",
        "dimension_combination",
    ]

    requirements = {
        field: question[field]
        for field in fields
        if field in question
        and pd.notna(question[field])
    }

    response = client.chat.completions.parse(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "requisiti_domanda": requirements,
                        "analisi_modello": analysis,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_format=Valutazione,
    )

    return response.choices[0].message.parsed


def main():
    NLI_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = pd.read_csv(
        QUESTIONS,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )

    analyses = load_analysis()

    output = NLI_DIR / f"{ANALYZED_MODEL}.csv"

    results = (
        pd.read_csv(output)
        if output.exists()
        else pd.DataFrame()
    )

    completed = (
        set(
            zip(
                results["video_id"].astype(str),
                results["question_order"],
            )
        )
        if not results.empty
        else set()
    )

    for index, question in questions.iterrows():

        video_id = Path(
            question["video_name"]
        ).stem

        question_order = question[
            "question_order"
        ]

        key = (
            video_id,
            question_order,
        )

        if key in completed:
            continue

        if video_id not in analyses:
            print(
                f"[SKIP] "
                f"{video_id}: "
                f"analisi non trovata"
            )
            continue

        print(
            f"[{ANALYZED_MODEL}] "
            f"{index + 1}/{len(questions)} "
            f"{video_id} "
            f"Q{question_order}"
        )

        result = evaluate(
            question,
            analyses[video_id],
        )

        row = {
            "model": ANALYZED_MODEL,
            "video_id": video_id,
            "question_order": question_order,
            "question": question["question_text"],
            "principal_dimension": question["principal_dimension"],
            "principal_label": question["principal_label"],
            "diagnostic_label": question["diagnostic_label"],
            "complexity_level": question["complexity_level"],
            "entity_required": question["entity"],
            "event_required": question["event"],
            "inference_required": question["inference"],
            "objective_criterion": question["objective_criterion"],
            "evidence_scope": question["evidence_scope"],
            "evidence_units": question["evidence_units"],
            "information_explicitness": question["information_explicitness"],
            "temporal_dependency": question["temporal_dependency"],
            "dimension_combination": question["dimension_combination"],
            **result.model_dump(),
        }

        pd.DataFrame(
            [row]
        ).to_csv(
            output,
            mode="a",
            header=not output.exists(),
            index=False,
            encoding="utf-8-sig",
        )

        completed.add(key)

    print(
        f"[OK] {output}"
    )


if __name__ == "__main__":
    main()