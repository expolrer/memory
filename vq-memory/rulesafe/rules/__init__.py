from .base import Rule, SafeState
from .generated_rules import *
from .paper_fidelity import PAPER_RULE_CLASSES, PAPER_RULE_DESCRIPTIONS, PAPER_RULES

RULES = {cls.rule_id: cls for cls in ALL_RULE_CLASSES}
RULE_PROFILES = {
    "historical": RULES,
    "paper_fidelity": PAPER_RULES,
}
RULE_DESCRIPTION_PROFILES = {
    "historical": {rule_id: cls.description for rule_id, cls in RULES.items()},
    "paper_fidelity": PAPER_RULE_DESCRIPTIONS,
}

__all__ = [
    "Rule",
    "SafeState",
    "RULES",
    "ALL_RULE_CLASSES",
    "PAPER_RULES",
    "PAPER_RULE_CLASSES",
    "PAPER_RULE_DESCRIPTIONS",
    "RULE_PROFILES",
    "RULE_DESCRIPTION_PROFILES",
] + [cls.__name__ for cls in ALL_RULE_CLASSES]
