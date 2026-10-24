"""Conservative extraction of explicit patient observations, without case-specific diagnoses."""
from __future__ import annotations
import re
from typing import Any

_DURATION = re.compile(r'\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|days?|d|weeks?|w)\b', re.I)
_BP = re.compile(r'\b(?:BP\s*[:=]?\s*)?(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?\b', re.I)
_TEMP = re.compile(r'\b(?:temp(?:erature)?\s*[:=]?\s*)?(\d{2}(?:\.\d+)?)\s*\°?\s*(?:c|celsius)\b', re.I)


def extract_patient_features(patient: dict[str, Any]) -> dict[str, Any]:
    """No absent symptom is inferred false; numeric features preserve units."""
    result: dict[str, Any] = {}
    if isinstance(patient.get('age'), (int, float)):
        result['age_years'] = patient['age']
    elif str(patient.get('age', '')).strip().isdigit():
        result['age_years'] = int(str(patient['age']).strip())
    sex = str(patient.get('sex', '')).strip().lower()
    if sex in {'female', 'male'}:
        result['sex'] = sex
    duration = str(patient.get('duration', ''))
    m = _DURATION.search(duration)
    if m:
        number, unit = float(m.group(1)), m.group(2).lower()
        hours = number * (24 if unit.startswith('d') else 168 if unit.startswith('w') else 1)
        result['duration_hours'] = hours
        result['days_since_illness_onset'] = hours / 24
    observations = ' '.join(str(patient.get(k, '')) for k in ('measurements', 'clinical_findings', 'symptoms'))
    m = _BP.search(observations)
    if m:
        result['sbp_mmhg'], result['dbp_mmhg'] = int(m.group(1)), int(m.group(2))
    m = _TEMP.search(observations)
    if m:
        result['temperature_c'] = float(m.group(1))
    # Only explicit findings are promoted. A missing phrase remains unknown.
    text = ' '.join(str(patient.get(k, '')) for k in ('symptoms', 'clinical_findings')).lower()
    patterns = {
        'acute_watery_diarrhoea': r'\b(?:watery|rice.water)\s+diarrh(?:oe|e)a\b',
        'sunken_eyes': r'\bsunken eyes\b',
        'absent_or_weak_pulse': r'\b(?:weak|absent)\s+(?:rapid\s+)?pulse\b',
        'vomiting_present': r'\b(?:vomit(?:ing)?|emesis)\b',
        'arthralgia_present': r'\b(?:joint pain|arthralgia|polyarthralgia)\b',
        'rash_present': r'\b(?:rash|skin eruption)\b',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match and not re.search(r'\b(?:no|denies|without)\s+' + pattern, text):
            result[key] = True
    return result


def diagnostic_query(patient: dict[str, Any]) -> str:
    """Keep clinical narrative for retrieval; unit-bearing measurements for rules."""
    return '. '.join(str(patient[k]).strip() for k in ('symptoms', 'clinical_findings', 'history')
                     if patient.get(k) not in (None, '', [], {}))


# Clinical matching is intentionally evidence-led: diagnostic criterion cards, not
# severity/differential cards, introduce diseases into the candidate set.
import math
from collections import Counter

_STOP = set("the and this that with without from into for are was has have patient patients person people within onset clinical criteria symptoms signs history assessment suspected confirmed disease infection considered based after before such condition when than their other may or including severe mild moderate high low above below possible present absent further test laboratory classification".split())
_SYNONYMS = {
    "diarrhea": "diarrhoea", "diarrhoeal": "diarrhoea",
    "polyarthralgia": "arthralgia", "arthritis": "arthralgia",
    "febrile": "fever", "febrility": "fever",
    "vomit": "vomiting", "emesis": "vomiting",
    "hypertensive": "hypertension",
}
def _clinical_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {_SYNONYMS.get(w, w) for w in words if w not in _STOP}


def rank_diseases(patient: dict[str, Any], cards: dict[str, Any]) -> dict[str, Any]:
    """Rank diseases from case-definition evidence only; no case-specific answer rules.

    Scores are retrieval heuristics, not calibrated medical probabilities.
    For weak/close evidence, leave the result ambiguous or unmatched.
    """
    from card_roles import assign_role
    features = extract_patient_features(patient)
    query = diagnostic_query(patient)
    terms = _clinical_terms(query)
    # Numerical observations are represented by typed rule fields, never bare terms.
    if features.get("temperature_c") is not None and features["temperature_c"] >= 38:
        terms.add("fever")
    grouped: dict[str, list[tuple[str, set[str]]]] = {}
    for card in cards.values():
        if card.clinical_category != "Diagnosis" or assign_role(card) != "case_definition":
            continue
        disease = str(card.disease_protocol)
        if "," in disease or " and " in disease.lower():
            continue
        criteria = str(getattr(card, "relevant_symptoms_or_prerequisites", "") or "")
        grouped.setdefault(disease, []).append((card.card_id, _clinical_terms(criteria)))
    frequency = Counter(term for definitions in grouped.values()
                        for term in set().union(*(x[1] for x in definitions)))
    scores = []
    for disease, definitions in grouped.items():
        best_score = 0.0
        best_card = None
        best_terms: list[str] = []
        for card_id, criteria in definitions:
            matching = terms & criteria
            value = sum(math.log(1 + (len(grouped) + 1) / (1 + frequency[t]))
                        for t in matching)
            if value > best_score:
                best_score, best_card, best_terms = value, card_id, sorted(matching)
        if best_score:
            scores.append({"disease": disease, "score": round(best_score, 4),
                           "card_id": best_card, "matched_features": best_terms})
    scores.sort(key=lambda item: (-item["score"], item["disease"]))
    if not scores:
        return {"status": "no_match", "candidates": [], "features": features}
    # Relative margin separates strongly supported evidence from overlapping cases.
    # This is a conservative heuristic pending calibration on diverse cases.
    top = scores[0]["score"]
    selected = [item for item in scores if item["score"] >= top * 0.8][:2]
    status = "clear" if len(selected) == 1 else "ambiguous"
    return {"status": status, "candidates": selected, "features": features}
