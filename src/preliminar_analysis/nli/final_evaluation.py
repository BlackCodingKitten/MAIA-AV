from pathlib import Path
from typing import Literal
import json
import re

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel


MODEL = "gpt-4o-2024-08-06"

FINAL_DIR = Path("data/preliminar_analysis/final_results")
NLI_DIR = FINAL_DIR / "nli"

QUESTIONS_FILE = Path("data/vsv/question_classification.csv")
CAPTION_FOIL_FILE = Path("data/vsv/caption-foil/caption_foil.csv")

OUTPUT_RESULTS = FINAL_DIR / "nli_risultati.csv"
OUTPUT_PROFICIENCY = FINAL_DIR / "question_proficiency.csv"

client = OpenAI()


PROMPT = """Sei un valutatore di Natural Language Inference.

Riceverai:
- una DOMANDA;
- una PREMESSA, cioè la rappresentazione strutturata estratta da un video;
- una IPOTESI.

Devi classificare la relazione tra PREMESSA e IPOTESI usando esclusivamente le informazioni contenute nella PREMESSA.

Usa soltanto una delle seguenti etichette:

- implicazione: la PREMESSA contiene informazioni sufficienti per sostenere chiaramente l'IPOTESI;
- contraddizione: la PREMESSA contiene informazioni chiaramente incompatibili con l'IPOTESI;
- neutro: la PREMESSA non contiene informazioni sufficienti per sostenere o contraddire chiaramente l'IPOTESI.

Regole:
- usa esclusivamente la PREMESSA come fonte di evidenza;
- la DOMANDA serve soltanto a indicare quale informazione è rilevante;
- non usare conoscenza esterna;
- non completare informazioni mancanti;
- non considerare la plausibilità dell'IPOTESI come evidenza;
- se l'evidenza è incompleta, debole, ambigua o richiede un'inferenza non chiaramente supportata, scegli neutro;
- in caso di dubbio scegli sempre neutro;
- non produrre spiegazioni."""


REGOLE = {
    "spaziale": "Per le relazioni spaziali, non inferire una relazione dalla sola presenza simultanea o vicinanza generica di due entità. La relazione deve essere chiaramente supportata dalla PREMESSA.",
    "temporale": "Per le relazioni temporali, non inferire prima o dopo dal semplice ordine con cui gli eventi sono scritti. Usa soltanto timestamp, intervalli o relazioni temporali presenti nella PREMESSA.",
    "causale": "Per le relazioni causali, la successione temporale non implica causalità. Non inventare cause, effetti, intenzioni, motivazioni o condizioni abilitanti non chiaramente supportate dalla PREMESSA.",
}


class RispostaNLI(BaseModel):
    etichetta: Literal["implicazione", "contraddizione", "neutro"]


def normalizza_video(value):
    numero = re.search(r"\d+", str(value))
    if not numero:
        raise ValueError(f"ID video non riconosciuto: {value}")
    return f"video{int(numero.group()):03d}"


def normalizza_categoria(value):
    categoria = str(value).strip().lower()
    return {"spatial": "spaziale", "temporal": "temporale", "causal": "causale"}.get(categoria, categoria)


def carica_domande():
    df = pd.read_csv(QUESTIONS_FILE, sep=None, engine="python", encoding="utf-8-sig")
    df["video_id"] = df["video_name"].map(normalizza_video)
    df["categoria"] = df["principal_dimension"].map(normalizza_categoria)
    return df[["video_id", "categoria", "question_text"]].rename(columns={"question_text": "domanda"})


def carica_coppie(domande):
    df = pd.read_csv(CAPTION_FOIL_FILE, sep=None, engine="python", encoding="utf-8-sig")
    df["video_id"] = df["video_id"].map(normalizza_video)
    df["categoria"] = df["question_category"].map(normalizza_categoria)
    return df[["id", "video_id", "categoria", "caption", "foil"]].merge(domande, on=["video_id", "categoria"], how="inner", validate="many_to_one")


def carica_finalizzazioni(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return {normalizza_video(item.get("video_id") or item.get("id_video")): item for item in data}

    if isinstance(data.get("videos"), list):
        return {normalizza_video(item.get("video_id") or item.get("id_video")): item for item in data["videos"]}

    if isinstance(data.get("results"), list):
        return {normalizza_video(item.get("video_id") or item.get("id_video")): item for item in data["results"]}

    if "video_id" in data or "id_video" in data:
        return {normalizza_video(data.get("video_id") or data.get("id_video")): data}

    return {normalizza_video(video_id): risultato for video_id, risultato in data.items() if isinstance(risultato, dict)}


def classifica(premessa, domanda, categoria, ipotesi):
    contenuto = json.dumps({"domanda": domanda, "premessa": premessa, "ipotesi": ipotesi}, ensure_ascii=False)

    risposta = client.chat.completions.parse(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": f"{PROMPT}\n\n{REGOLE[categoria]}"},
            {"role": "user", "content": contenuto},
        ],
        response_format=RispostaNLI,
    )

    risultato = risposta.choices[0].message.parsed

    if risultato is None:
        raise RuntimeError("Il modello non ha restituito una classificazione NLI valida.")

    return risultato.etichetta


def valuta_coppia(caption, foil):
    return "PASS" if caption == "implicazione" and foil == "contraddizione" else "NOT PASS"


def valuta_modello(path, coppie):
    modello = path.stem
    finalizzazioni = carica_finalizzazioni(path)
    checkpoint = NLI_DIR / f"{modello}_nli.csv"

    risultati = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    completati = set(risultati["id"]) if not risultati.empty else set()

    for i, row in coppie.iterrows():
        if row["id"] in completati:
            continue

        if row["video_id"] not in finalizzazioni:
            raise ValueError(f"Finalizzazione mancante per {row['video_id']} nel modello {modello}.")

        print(f"[{modello}] {i + 1}/{len(coppie)} - {row['id']}")

        premessa = finalizzazioni[row["video_id"]]
        nli_caption = classifica(premessa, row["domanda"], row["categoria"], row["caption"])
        nli_foil = classifica(premessa, row["domanda"], row["categoria"], row["foil"])

        risultato = {
            "modello": modello,
            "id": row["id"],
            "video_id": row["video_id"],
            "categoria": row["categoria"],
            "domanda": row["domanda"],
            "caption": row["caption"],
            "foil": row["foil"],
            "nli_caption": nli_caption,
            "nli_foil": nli_foil,
            "esito": valuta_coppia(nli_caption, nli_foil),
        }

        pd.DataFrame([risultato]).to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
        completati.add(row["id"])

    return pd.read_csv(checkpoint)


def crea_proficiency(risultati):
    risultati["pass"] = (risultati["esito"] == "PASS").astype(int)

    df = risultati.groupby(["modello", "video_id", "categoria", "domanda"]).agg(
        totale_coppie=("esito", "size"),
        coppie_pass=("pass", "sum"),
    ).reset_index()

    df["tasso_pass"] = df["coppie_pass"] / df["totale_coppie"]
    df["proficiency"] = ["PASS" if passate == totale else "NOT PASS" for passate, totale in zip(df["coppie_pass"], df["totale_coppie"])]

    return df


def main():
    NLI_DIR.mkdir(parents=True, exist_ok=True)

    final_files = sorted(FINAL_DIR.glob("*.json"))

    if len(final_files) != 4:
        raise RuntimeError(f"Mi aspettavo 4 file JSON in {FINAL_DIR}, ma ne ho trovati {len(final_files)}.")

    domande = carica_domande()
    coppie = carica_coppie(domande)

    print(f"Domande: {len(domande)}")
    print(f"Coppie caption-foil: {len(coppie)}")
    print(f"Modelli: {len(final_files)}")

    risultati = pd.concat([valuta_modello(path, coppie) for path in final_files], ignore_index=True)
    risultati.to_csv(OUTPUT_RESULTS, index=False)

    proficiency = crea_proficiency(risultati)
    proficiency.to_csv(OUTPUT_PROFICIENCY, index=False)

    print(f"Risultati NLI salvati in: {OUTPUT_RESULTS}")
    print(f"Proficiency salvata in: {OUTPUT_PROFICIENCY}")
    print(f"File intermedi salvati in: {NLI_DIR}")


if __name__ == "__main__":
    main()