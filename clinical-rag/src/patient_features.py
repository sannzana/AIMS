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
