import pytest
from src.classification.rules import RuleBasedClassifier

def test_rule_based_classifier_superhuman():
    classifier = RuleBasedClassifier()
    # Provide perfect metrics (at or better than ideal)
    metrics = {
        "control_precision": 0.0,
        "dynamic_stability": 0.0,
        "cost_of_transport": 0.0,
        "system_latency": 0.0,
    }
    score, tier = classifier.classify(metrics)
    assert score == 1.0
    assert tier == "Superhuman/Industrial"

def test_rule_based_classifier_research():
    classifier = RuleBasedClassifier()
    # Provide metrics in the middle range (between ideal and worst)
    metrics_research = {
        "control_precision": 0.13,
        "dynamic_stability": 0.04025,
        "cost_of_transport": 5.15,
        "system_latency": 0.0775,
    }
    score, tier = classifier.classify(metrics_research)
    assert 0.40 <= score <= 0.85

def test_rule_based_classifier_experimental():
    classifier = RuleBasedClassifier()
    # Provide terrible metrics (worse than Class 3 acceptable)
    metrics = {
        "control_precision": 0.5,
        "dynamic_stability": 0.5,
        "cost_of_transport": 20.0,
        "system_latency": 0.5,
    }
    score, tier = classifier.classify(metrics)
    assert score == 0.0
    assert tier == "Experimental"

def test_missing_metrics():
    classifier = RuleBasedClassifier()
    # Missing some metrics should still classify based on provided ones
    metrics = {
        "control_precision": 0.0,
        "system_latency": 0.0,
    }
    score, tier = classifier.classify(metrics)
    assert score == 1.0
    assert tier == "Superhuman/Industrial"
