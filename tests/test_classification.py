import pytest
from src.classification.rules import RuleBasedClassifier

def test_rule_based_classifier_superhuman():
    classifier = RuleBasedClassifier()
    # Provide perfect metrics (at or better than ideal)
    metrics = {
        "smoothness_ldlj": 0.0,
        "smoothness_sparc": 0.0,
        "symmetry": 0.0,
        "periodicity": 1.0,
        "rom_utilisation": 2.0
    }
    score, tier = classifier.classify(metrics)
    assert score == 1.0
    assert tier == "Superhuman/Industrial"

def test_rule_based_classifier_research():
    classifier = RuleBasedClassifier()
    # Provide metrics in the middle range (between ideal and worst)
    metrics_research = {
        "smoothness_ldlj": -7.5,
        "smoothness_sparc": -15.0,
        "symmetry": 17.5,
        "periodicity": 0.45,
        "rom_utilisation": 0.35
    }
    score, tier = classifier.classify(metrics_research)
    assert 0.40 <= score <= 0.85

def test_rule_based_classifier_experimental():
    classifier = RuleBasedClassifier()
    # Provide terrible metrics (worse than Class 3 acceptable)
    metrics = {
        "smoothness_ldlj": -30.0,
        "smoothness_sparc": -50.0,
        "symmetry": 100.0,
        "periodicity": 0.0,
        "rom_utilisation": 0.01
    }
    score, tier = classifier.classify(metrics)
    assert score == 0.0
    assert tier == "Experimental"

def test_missing_metrics():
    classifier = RuleBasedClassifier()
    # Missing some metrics should still classify based on provided ones
    metrics = {
        "smoothness_ldlj": 0.0,
        "periodicity": 1.0,
    }
    score, tier = classifier.classify(metrics)
    assert score == 1.0
    assert tier == "Superhuman/Industrial"
