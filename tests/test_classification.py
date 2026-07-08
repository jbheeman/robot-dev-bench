<<<<<<< HEAD
# Tests for src/classification — clustering and rule-based motion labellers.
=======
import pytest
from src.classification.rules import RuleBasedClassifier

def test_rule_based_classifier_superhuman():
    classifier = RuleBasedClassifier()
    # Provide perfect metrics (at or better than ideal)
    metrics = {
        "rmse": 0.0,
        "cot": 0.05,
        "latency_ms": 0.5,
        "stress": 0.0,
        "imu_variance": 0.0
    }
    score, tier = classifier.classify(metrics)
    assert score == 1.0
    assert tier == "Superhuman/Industrial"

def test_rule_based_classifier_research():
    classifier = RuleBasedClassifier()
    # Provide metrics in the middle range (between ideal and worst)
    metrics_research = {
        "rmse": 0.04,
        "cot": 0.45,
        "latency_ms": 10.0,
        "stress": 0.2,
        "imu_variance": 0.02
    }
    score, tier = classifier.classify(metrics_research)
    assert 0.60 <= score < 0.85
    assert tier == "Research"

def test_rule_based_classifier_experimental():
    classifier = RuleBasedClassifier()
    # Provide terrible metrics (worse than Class 3 acceptable)
    metrics = {
        "rmse": 0.5,
        "cot": 2.0,
        "latency_ms": 50.0,
        "stress": 2.0,
        "imu_variance": 0.5
    }
    score, tier = classifier.classify(metrics)
    assert score == 0.0
    assert tier == "Experimental"

def test_missing_metrics():
    classifier = RuleBasedClassifier()
    # Missing some metrics should still classify based on provided ones
    metrics = {
        "rmse": 0.0,
        "cot": 0.1,
    }
    score, tier = classifier.classify(metrics)
    assert score == 1.0
    assert tier == "Superhuman/Industrial"
>>>>>>> d8d255ef7cce25e829b0eef8d4032f0ebc4ac185
