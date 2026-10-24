"""Offline regression tests for single-pass improved diagnosis candidate selection."""
from types import SimpleNamespace
from patient_features import extract_patient_features, rank_diseases

def card(cid, disease, criteria):
    return SimpleNamespace(card_id=cid, disease_protocol=disease,
        clinical_category="Diagnosis", card_title="Clinical Case Definition",
        clinical_classification_presentation=None,
        relevant_symptoms_or_prerequisites=criteria,
        model_dump=lambda: {"clinical_category":"Diagnosis",
          "card_title":"Clinical Case Definition",
          "clinical_classification_presentation": None})

def test_units_are_not_interchanged():
    assert extract_patient_features({"duration":"12 hours", "age":38})["days_since_illness_onset"] == 0.5
    assert extract_patient_features({"duration":"12 days", "age":38})["days_since_illness_onset"] == 12

def test_strong_watery_diarrhoea_does_not_become_joint_fever_case():
    cards = {"A": card("A","Waterborne disease","acute watery diarrhoea, intense thirst and dehydration"),
             "B": card("B","Febrile joint disease","sudden fever, severe joint pain and swollen joints")}
    patient = {"symptoms":"Sudden acute watery diarrhoea, intense thirst and dehydration",
               "duration":"12 hours", "age":38}
    outcome = rank_diseases(patient,cards)
    assert outcome["status"]=="clear"
    assert [v["disease"] for v in outcome["candidates"]]==["Waterborne disease"]

def test_weak_unrelated_findings_produce_no_match():
    cards={"A":card("A","Disease A","fever, headache, arthralgia")}
    assert rank_diseases({"symptoms":"isolated toothache"},cards)["status"]=="no_match"

def test_both_symptom_sets_can_remain_ambiguous():
    cards={"A":card("A","Disease A","fever, rash, headache"),
           "B":card("B","Disease B","fever, rash, joint pain")}
    result=rank_diseases({"symptoms":"fever, rash, headache and joint pain"},cards)
    assert result["status"] == "ambiguous"

def test_raised_blood_pressure_remains_clinically_searchable():
    cards={"H":card("H","Pressure disorder","raised blood pressure assessment"),
           "F":card("F","Fever disorder","fever and rash")}
    patient={"measurements":"BP 162/98 mmHg", "symptoms":"morning dizziness"}
    result=rank_diseases(patient,cards)
    assert result["candidates"][0]["disease"] == "Pressure disorder"

def test_single_generic_overlap_cannot_create_another_diagnosis():
    cards={"A":card("A","Disease A","acute watery diarrhoea intense thirst"),
           "B":card("B","Disease B","sudden fever arthralgia rash")}
    patient={"symptoms":"Sudden acute watery diarrhoea intense thirst"}
    result=rank_diseases(patient,cards)
    assert result["status"] == "clear"
    assert len(result["candidates"]) == 1

def test_duration_and_age_cannot_create_disease_match():
    cards={"A":card("A","Disease A","12 days fever age 38 years")}
    result=rank_diseases({"age":38,"duration":"12 hours","symptoms":"isolated toothache"},cards)
    assert result["status"] == "no_match"
