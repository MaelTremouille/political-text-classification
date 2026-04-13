"""
Build an aggressively cleaned text column `text_clean_v2` alongside the
existing `text_clean`. The goal is to strip the lexical leakage that
dominates the TF-IDF top features in the baseline (stopwords, election
dates, French place names, politicians' surnames, coalition banners,
Alsatian bilingual fragments).

The transformer pipeline keeps using `text_clean` because attention models
need the natural syntactic structure.

Input:  data/corpus_labeled.parquet, data/{train,val,test}.parquet
Output: same files, with an added `text_clean_v2` column.
"""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")


# --- French stopwords (comprehensive, no external dependency) ---
# Based on nltk/spacy French lists, extended with OCR variants frequent
# in 1970s-90s manifestos.
FRENCH_STOPWORDS = {
    "a", "à", "abord", "absolument", "afin", "ah", "ai", "aie", "ailleurs", "ainsi",
    "allaient", "allo", "allô", "allons", "après", "assez", "attendu", "au", "aucun",
    "aucune", "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "auraient", "aurait",
    "auront", "aussi", "autre", "autrefois", "autrement", "autres", "autrui", "aux",
    "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avoir",
    "avons", "ayant", "b", "bah", "bas", "basee", "bat", "beaucoup", "bien", "bigre",
    "boum", "bravo", "brrr", "c", "ça", "car", "ce", "ceci", "cela", "celle",
    "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci",
    "celui-là", "cent", "cependant", "certain", "certaine", "certaines", "certains",
    "certes", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune",
    "chaque", "cher", "chers", "chez", "chiche", "chut", "ci", "cinq", "cinquantaine",
    "cinquante", "cinquantième", "cinquième", "clac", "clic", "combien", "comme",
    "comment", "comparable", "comparables", "compris", "concernant", "contre",
    "couic", "crac", "d", "da", "dans", "de", "debout", "dedans", "dehors",
    "delà", "depuis", "derrière", "des", "dès", "désormais", "desquelles",
    "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement", "devant",
    "devers", "devra", "différent", "différente", "différentes", "différents",
    "dire", "divers", "diverse", "diverses", "dix", "dix-huit", "dixième",
    "dix-neuf", "dix-sept", "doit", "doivent", "donc", "dont", "douze", "douzième",
    "dring", "du", "duquel", "durant", "e", "effet", "eh", "elle", "elle-même",
    "elles", "elles-mêmes", "en", "encore", "entre", "envers", "environ", "es",
    "ès", "est", "et", "etant", "étaient", "étais", "était", "étant", "etc",
    "été", "etre", "être", "eu", "euh", "eux", "eux-mêmes", "excepté", "f",
    "façon", "fais", "faisaient", "faisant", "fait", "faut", "feront", "fi",
    "flac", "floc", "font", "g", "gens", "h", "ha", "hé", "hein", "hélas",
    "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors", "hou", "houp",
    "hue", "hum", "hurrah", "i", "il", "ils", "importe", "j", "je", "jusqu",
    "jusque", "juste", "k", "l", "la", "là", "laquelle", "las", "le", "lequel",
    "les", "lès", "lesquelles", "lesquels", "leur", "leurs", "longtemps", "lorsque",
    "lui", "lui-même", "m", "ma", "maint", "maintenant", "mais", "malgré", "me",
    "même", "mêmes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille",
    "mince", "moi", "moi-même", "moins", "mon", "moyennant", "n", "na", "ne",
    "néanmoins", "neuf", "neuvième", "ni", "nombreuses", "nombreux", "non", "nos",
    "notre", "nôtre", "nôtres", "nous", "nous-mêmes", "nul", "o", "o|", "ô", "oh",
    "ohé", "olé", "ollé", "on", "ont", "onze", "onzième", "ore", "ou", "où",
    "ouf", "ouias", "oust", "ouste", "outre", "p", "paf", "pan", "par", "parce",
    "parmi", "parole", "partant", "particulier", "particulière", "particulièrement",
    "pas", "passé", "pendant", "personne", "peu", "peut", "peuvent", "peux", "pff",
    "pfft", "pfut", "pif", "plein", "plouf", "plus", "plusieurs", "plutôt", "pouah",
    "pour", "pourquoi", "premier", "première", "premièrement", "près", "proche",
    "psitt", "puisque", "q", "qu", "quand", "quant", "quanta", "quant-à-soi", "quarante",
    "quatorze", "quatre", "quatre-vingt", "quatrième", "quatrièmement", "que", "quel",
    "quelconque", "quelle", "quelles", "quelque", "quelques", "quelqu'un", "quels",
    "qui", "quiconque", "quinze", "quoi", "quoique", "r", "revoici", "revoilà",
    "rien", "s", "sa", "sacrebleu", "sans", "sapristi", "sauf", "se", "seize",
    "selon", "sept", "septième", "sera", "seraient", "serait", "seront", "ses",
    "seulement", "si", "sien", "sienne", "siennes", "siens", "sinon", "six", "sixième",
    "soi", "soi-même", "soit", "soixante", "son", "sont", "sous", "stop", "suis",
    "suivant", "sur", "surtout", "t", "ta", "tac", "tant", "te", "té", "tel",
    "telle", "tellement", "telles", "tels", "tenant", "tes", "tic", "tien", "tienne",
    "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant", "toujours",
    "tous", "tout", "toute", "toutes", "treize", "trente", "très", "trois",
    "troisième", "troisièmement", "trop", "tsoin", "tsouin", "tu", "u", "un",
    "une", "unes", "uns", "v", "va", "vais", "vas", "vé", "vers", "via", "vif",
    "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voilà", "vont",
    "vos", "votre", "vôtre", "vôtres", "vous", "vous-mêmes", "vu", "w", "x", "y",
    "z", "zut", "ans", "an", "années", "année", "aux", "celui-ci", "puisque",
    # German/bilingual Alsatian artefacts detected in the current top features
    "die", "und", "nicht", "der", "das", "ist", "sind", "eine", "ein", "mit",
    "von", "für", "wir", "sie", "es", "den", "dem", "zu", "zur", "zum",
}


# --- French months (for date masking) ---
MONTHS_FR = [
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre", "décembre", "decembre",
]


# --- Toponyms: French departments (names + common OCR variants) ---
# 95 metropolitan + Corsica (2A/2B) + overseas departments + historical regions
DEPARTMENT_NAMES = [
    "ain", "aisne", "allier", "alpes-de-haute-provence", "alpes-maritimes", "ardèche",
    "ardeche", "ardennes", "ariège", "ariege", "aube", "aude", "aveyron",
    "bouches-du-rhône", "bouches-du-rhone", "calvados", "cantal", "charente",
    "charente-maritime", "cher", "corrèze", "correze", "corse", "haute-corse",
    "corse-du-sud", "côte-d'or", "cote-d'or", "côtes-d'armor", "cotes-d'armor",
    "côtes-du-nord", "cotes-du-nord", "creuse", "dordogne", "doubs", "drôme",
    "drome", "eure", "eure-et-loir", "finistère", "finistere", "gard",
    "haute-garonne", "garonne", "gers", "gironde", "hérault", "herault",
    "ille-et-vilaine", "indre", "indre-et-loire", "isère", "isere", "jura",
    "landes", "loir-et-cher", "loire", "haute-loire", "loire-atlantique",
    "loiret", "lot", "lot-et-garonne", "lozère", "lozere", "maine-et-loire",
    "manche", "marne", "haute-marne", "mayenne", "meurthe-et-moselle", "meuse",
    "morbihan", "moselle", "nièvre", "nievre", "nord", "oise", "orne",
    "pas-de-calais", "puy-de-dôme", "puy-de-dome", "pyrénées-atlantiques",
    "pyrenees-atlantiques", "hautes-pyrénées", "hautes-pyrenees",
    "pyrénées-orientales", "pyrenees-orientales", "bas-rhin", "haut-rhin",
    "rhône", "rhone", "haute-saône", "haute-saone", "saône-et-loire", "saone-et-loire",
    "sarthe", "savoie", "haute-savoie", "paris", "seine", "seine-maritime",
    "seine-et-marne", "yvelines", "deux-sèvres", "deux-sevres", "somme",
    "tarn", "tarn-et-garonne", "var", "vaucluse", "vendée", "vendee", "vienne",
    "haute-vienne", "vosges", "yonne", "territoire-de-belfort", "essonne",
    "hauts-de-seine", "seine-saint-denis", "val-de-marne", "val-d'oise",
    "guadeloupe", "martinique", "guyane", "réunion", "reunion", "mayotte",
    "nouvelle-calédonie", "nouvelle-caledonie",
]

# Historical French regions frequently mentioned in the manifestos
REGIONS = [
    "alsace", "aquitaine", "auvergne", "bourgogne", "bretagne", "champagne",
    "corse", "dauphiné", "dauphine", "flandre", "franche-comté", "franche-comte",
    "gascogne", "languedoc", "limousin", "lorraine", "midi-pyrénées", "midi-pyrenees",
    "nord-pas-de-calais", "normandie", "pays-de-la-loire", "picardie",
    "poitou-charentes", "provence", "roussillon", "savoie", "touraine",
]

# Top French cities (large cities + frequent administrative centres).
# We cannot cover the ~36k INSEE communes without an external gazetteer,
# but the most frequently cited ones provide most of the coverage.
MAJOR_CITIES = [
    "paris", "marseille", "lyon", "toulouse", "nice", "nantes", "strasbourg",
    "montpellier", "bordeaux", "lille", "rennes", "reims", "le havre", "havre",
    "saint-étienne", "saint-etienne", "toulon", "grenoble", "dijon", "angers",
    "nîmes", "nimes", "villeurbanne", "saint-denis", "aix-en-provence",
    "le mans", "mans", "clermont-ferrand", "brest", "limoges", "tours", "amiens",
    "perpignan", "metz", "besançon", "besancon", "orléans", "orleans", "mulhouse",
    "rouen", "caen", "nancy", "argenteuil", "montreuil", "roubaix", "tourcoing",
    "dunkerque", "avignon", "poitiers", "nanterre", "créteil", "creteil",
    "versailles", "courbevoie", "asnières", "asnieres", "colombes", "la rochelle",
    "rochelle", "aulnay", "vitry", "champigny", "saint-maur", "drancy", "calais",
    "béziers", "beziers", "fort-de-france", "mérignac", "merignac", "ajaccio",
    "antibes", "noisy-le-grand", "saint-nazaire", "nazaire", "cannes", "bayonne",
    "valence", "chambéry", "chambery", "quimper", "issy-les-moulineaux", "issy",
    "levallois", "clichy", "troyes", "villeneuve", "ivry", "neuilly", "sarcelles",
    "pessac", "vénissieux", "venissieux", "cergy", "clamart", "montauban",
    "boulogne", "laval", "lorient", "narbonne", "saint-quentin", "quentin",
    "fontenay", "annecy", "belfort", "bastia", "bobigny", "vincennes", "corbeil",
    "charleville", "thionville", "chartres", "évreux", "evreux", "niort",
    "meaux", "cherbourg", "vannes", "tarbes", "compiègne", "compiegne",
    "châlon", "chalon", "macon", "mâcon", "arras", "albi", "blois", "belfort",
    # Overseas territories
    "pointe-à-pitre", "pointe-a-pitre", "basse-terre", "saint-pierre",
    "kourou", "papeete", "nouméa", "noumea", "mamoudzou",
]

ALL_TOPONYMS = set(DEPARTMENT_NAMES) | set(REGIONS) | set(MAJOR_CITIES)


# --- Extended leakage list: politicians, coalition banners, FN-era vocab ---
# These persisted in text_clean despite label_extraction's party removal
# because they are person names or sub-coalition labels not in PARTY_PATTERNS.
EXTENDED_LEAKAGE_PATTERNS = [
    # Politicians 1973-1993
    r"\bLE\s+PEN\b", r"\bMARIE\s+LE\s+PEN\b", r"\bJEAN[- ]MARIE\s+LE\s+PEN\b",
    r"\bMARCHAIS\b", r"\bCHIRAC\b", r"\bMITTERRAND\b", r"\bROCARD\b",
    r"\bBARRE\b", r"\bGISCARD\b", r"\bGISCARD\s+D'ESTAING\b",
    r"\bFABIUS\b", r"\bJOSPIN\b", r"\bBALLADUR\b", r"\bMAUROY\b",
    r"\bCRESSON\b", r"\bPASQUA\b", r"\bMÉGRET\b", r"\bMEGRET\b",
    r"\bDELORS\b", r"\bCHEVÈNEMENT\b", r"\bCHEVENEMENT\b",
    r"\bWAECHTER\b", r"\bLAJOINIE\b", r"\bJUPPÉ\b", r"\bJUPPE\b",
    r"\bSÉGUIN\b", r"\bSEGUIN\b", r"\bVEIL\b",
    # Coalition / organizational names leaked through
    r"\bCERCLE\s+NATIONAL\b", r"\bFRONT\s+NATIONAL\b",
    r"\bNOUVELLE\s+SOCIÉTÉ\b", r"\bNOUVELLE\s+SOCIETE\b",
    r"\bUNION\s+DE\s+LA\s+GAUCHE\b", r"\bUNION\s+POUR\s+LA\s+FRANCE\b",
    r"\bGAUCHE\s+UNIE\b", r"\bFRANCE\s+UNIE\b",
    r"\bMAJORITÉ\s+PRÉSIDENTIELLE\b", r"\bMAJORITE\s+PRESIDENTIELLE\b",
    r"\bRASSEMBLEMENT\s+NATIONAL\b",
    # Direct ideological marker words that leak the label (PCF/PS)
    r"\bCOMMUNISTE[S]?\b", r"\bCOMMUNISME\b",
    r"\bSOCIALISTE[S]?\b", r"\bSOCIALISME\b",
    r"\bR[ÉE]PUBLICAIN[S]?\b",
    # Recurring FN slogans
    r"\bFRAN[ÇC]AIS\s+D'ABORD\b", r"\bPR[ÉE]F[ÉE]RENCE\s+NATIONALE\b",
    # Candidate-specific template phrases
    r"\bVOTRE\s+CANDIDAT\b", r"\bCANDIDAT\s+OFFICIEL\b",
]


def _build_regexes():
    """Pre-compile all regex patterns once."""
    # Dates: day + month + year, month + year, day + month, isolated 19XX years
    months_alt = "|".join(MONTHS_FR)
    date_patterns = [
        re.compile(rf"\b\d{{1,2}}\s+({months_alt})\s+\d{{4}}\b", re.IGNORECASE),
        re.compile(rf"\b\d{{1,2}}\s+({months_alt})\b", re.IGNORECASE),
        re.compile(rf"\b({months_alt})\s+\d{{4}}\b", re.IGNORECASE),
        re.compile(rf"\b({months_alt})\b", re.IGNORECASE),
        re.compile(r"\b(19[7-9]\d)\b"),
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
        re.compile(r"\b\d{1,2}\s*(?:er|e|ème|ere)\s+tour\b", re.IGNORECASE),
    ]

    # Toponyms: one big alternation, word-boundary-anchored, case-insensitive
    toponym_sorted = sorted(ALL_TOPONYMS, key=len, reverse=True)
    toponym_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in toponym_sorted) + r")\b",
        re.IGNORECASE,
    )

    # Extended list of name-leakage patterns
    leak_patterns = [re.compile(p, re.IGNORECASE) for p in EXTENDED_LEAKAGE_PATTERNS]

    return date_patterns, toponym_pattern, leak_patterns


DATE_PATTERNS, TOPONYM_PATTERN, LEAK_PATTERNS = _build_regexes()


def _mask_stopwords(text):
    """Remove French stopwords and German/Alsatian bilingual fragments."""
    tokens = re.split(r"(\W+)", text)
    kept = [tok if (not tok.strip() or tok.strip().lower() not in FRENCH_STOPWORDS) else " " for tok in tokens]
    return "".join(kept)


def clean_text_v2(text):
    """Aggressive cleaning on top of text_clean: dates, toponyms, names, stopwords.

    Order matters: toponyms/names/dates first (they use original casing and
    word boundaries), stopwords last (since they operate on the tokenized form).
    """
    cleaned = text
    for pat in LEAK_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    for pat in DATE_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = TOPONYM_PATTERN.sub(" ", cleaned)
    cleaned = _mask_stopwords(cleaned)
    # Strip isolated single characters and digits that remain
    cleaned = re.sub(r"\b[a-zA-Z]\b", " ", cleaned)
    cleaned = re.sub(r"\b\d+\b", " ", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def process_file(path: Path):
    df = pd.read_parquet(path)
    if "text_clean" not in df.columns:
        raise RuntimeError(f"text_clean missing in {path}")
    print(f"[{path.name}] {len(df)} rows — building text_clean_v2")
    df["text_clean_v2"] = df["text_clean"].apply(clean_text_v2)
    df.to_parquet(path, index=False)
    return df


def main():
    files = [
        DATA_DIR / "corpus_labeled.parquet",
        DATA_DIR / "train.parquet",
        DATA_DIR / "val.parquet",
        DATA_DIR / "test.parquet",
    ]
    for p in files:
        if not p.exists():
            print(f"Skip missing: {p}")
            continue
        df = process_file(p)

    # Sanity check on the labeled parquet
    df = pd.read_parquet(DATA_DIR / "corpus_labeled.parquet")
    print("\n=== Sanity check: 3 random docs before/after ===")
    sample = df.sample(3, random_state=0)
    for _, row in sample.iterrows():
        print(f"\n--- {row['doc_name']} ({row['party_family']}) ---")
        print("CLEAN    :", row["text_clean"][:250].replace("\n", " "))
        print("CLEAN V2 :", row["text_clean_v2"][:250])


if __name__ == "__main__":
    main()
