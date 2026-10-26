"""Escaped, theme-aware display helpers; no patient or rule evaluation."""
from html import escape


def readable(value):
    return str(value).replace("_", " ")


def muted(st, text):
    st.markdown('<div style="font-size:.78rem;line-height:1.55;opacity:.72;margin:.18rem 0 .45rem">'
                + escape(str(text)) + '</div>', unsafe_allow_html=True)


def detail(st, label, value, secondary=False):
    size, opacity = ('.8rem', '.78') if secondary else ('.9rem', '1')
    st.markdown(f'<div style="font-size:{size};line-height:1.65;opacity:{opacity};margin:.2rem 0">'
                f'<strong>{escape(readable(label))}:</strong> {escape(readable(value))}</div>', unsafe_allow_html=True)


def _condition_html(tree):
    if isinstance(tree, dict):
        for key, label in (("all", "ALL of these (AND)"), ("any", "ANY of these (OR)")):
            if isinstance(tree.get(key), list):
                children = ''.join('<li>' + _condition_html(child) + '</li>' for child in tree[key])
                return '<strong>' + label + '</strong><ul style="margin:.2rem 0 .3rem;padding-left:1.3rem">' + children + '</ul>'
    from management_evidence import condition_text
    return escape(readable(condition_text(tree)))


def condition(st, tree, secondary=False):
    size, opacity = ('.8rem', '.78') if secondary else ('.88rem', '1')
    st.markdown(f'<div style="font-size:{size};line-height:1.65;opacity:{opacity};margin:.25rem 0 .65rem">'
                + _condition_html(tree) + '</div>', unsafe_allow_html=True)


def requirements(st, gates, secondary=False):
    for gate in gates:
        value = str(gate['required_value'])
        if gate.get('status'):
            value += ' (' + str(gate['status']) + ')'
        detail(st, 'Requires ' + readable(gate['field']), value, secondary)


def render_conditional_options(st, entries):
    """Group repeated shared gates; preserve every condition and restriction."""
    from management_evidence import condition_text
    st.caption('These requirements come from the source, not patient findings or instructions to start treatment.')
    for entry in entries:
        with st.container(border=True):
            st.markdown('**' + entry['title'] + '**')
            muted(st, 'Source: ' + entry['card_id'])
            if entry.get('missing_inputs'):
                detail(st, 'Information needed', ', '.join(entry['missing_inputs']))
            rules = entry['conditional_rules']
            def gate_key(g):
                return (g['field'], g['required_value'], g.get('status'))
            common = [g for g in rules[0].get('prerequisites', []) if all(
                any(gate_key(other) == gate_key(g) for other in r.get('prerequisites', [])) for r in rules)] if rules else []
            requirements(st, common)
            common_keys = {gate_key(g) for g in common}
            shown_restrictions = set()
            for index, rule in enumerate(rules, 1):
                with st.container(border=True):
                    muted(st, f"Condition set {index} · {rule.get('rule_id') or 'source rule'}")
                    if rule['status'] == 'restricted':
                        st.warning('A source restriction applies; this instruction is not approved for the supplied facts.')
                    selection = rule.get('age_clause_selection')
                    if selection and selection['status'] != 'matched':
                        detail(st, 'Age-specific instruction requires review', selection['status'])
                        if rule.get('source_instruction'):
                            st.write('Unselected source alternatives (not patient-specific instructions): ' + rule['source_instruction'])
                    if rule.get('unsupported_rule'):
                        st.warning('This source rule/table cannot yet be evaluated; do not assume a matching row.')
                    if rule.get('condition'):
                        detail(st, 'Applies if', '')
                        condition(st, rule['condition'])
                    requirements(st, [g for g in rule.get('prerequisites', []) if gate_key(g) not in common_keys])
                    for restriction in rule.get('restrictions', []):
                        text = (restriction.get('interpretation') or restriction.get('action') or condition_text(restriction.get('condition'))
                                if isinstance(restriction, dict) else str(restriction))
                        if text not in shown_restrictions:
                            detail(st, 'Restriction', text)
                            shown_restrictions.add(text)
