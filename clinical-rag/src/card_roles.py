"""Classify source cards for retrieval."""

def assign_role(card):
    data = card.model_dump() if hasattr(card, 'model_dump') else (dict(card) if isinstance(card, dict) else vars(card))
    category = str(data.get('clinical_category') or '').casefold()
    if category != 'diagnosis':
        return category or 'other'
    text = str(data.get('card_title') or '').casefold() + ' ' + str(data.get('clinical_classification_presentation') or '').casefold()
    if any(k in text for k in ('differential', 'versus', 'mimic')):
        return 'differential'
    if any(k in text for k in ('history', 'physical examination', 'risk factor', 'hmod')):
        return 'history_taking'
    # Assessment/classification cards describe a branch or severity, not the
    # disease case definition. Include common adjective forms such as "severe".
    if any(k in text for k in (
        'severity', 'severe', 'classification', 'assessment',
        'some and no dehydration', 'some dehydration', 'no dehydration',
    )):
        return 'severity'
    return 'case_definition'
