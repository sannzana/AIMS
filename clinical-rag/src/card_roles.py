"""Classify source cards for retrieval."""

def assign_role(card):
    data = card.model_dump() if hasattr(card, 'model_dump') else dict(card)
    category = str(data.get('clinical_category') or '').casefold()
    if category != 'diagnosis':
        return category or 'other'
    text = str(data.get('card_title') or '').casefold() + ' ' + str(data.get('clinical_classification_presentation') or '').casefold()
    if any(k in text for k in ('differential', 'versus', 'mimic')):
        return 'differential'
    if any(k in text for k in ('severity', 'classification')):
        return 'severity'
    if any(k in text for k in ('history', 'physical examination', 'risk factor', 'hmod')):
        return 'history_taking'
    return 'case_definition'
