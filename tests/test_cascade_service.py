"""Integration tests for cascade training, evaluation, and the CLI (Milestone 4)."""

from pathlib import Path

from routellm.application.classifier_service import ClassifierService
from routellm.classification.cascade_model import CascadeModel, load_cascade_model
from routellm.classification.dataset import Dataset, write_dataset
from routellm.cli import main as cli_main
from routellm.evaluation.cascade_report import CascadeEvaluationReport

_TEMPLATES = {
    "coding": (
        "write a python script",
        "fix the java bug",
        "build a rest api",
        "debug the docker container",
        "refactor this typescript class",
        "write a golang function",
        "query the sql database",
        "create a kubernetes deployment",
        "write a rust compiler",
        "implement binary search",
    ),
    "math": (
        "calculate the percentage",
        "what is the sum",
        "solve this equation",
        "compute the derivative",
        "divide the numbers",
        "multiply two values",
        "find the square root",
        "solve the algebra problem",
        "calculate the integral",
        "subtract the numbers",
    ),
    "translation": (
        "translate into tamil",
        "translate into hindi",
        "translate into telugu",
        "translate into spanish",
        "translate into french",
        "translate into german",
        "translate into japanese",
        "translate into chinese",
        "translate into korean",
        "translate this paragraph",
    ),
}


def _write_keyword_dataset(path) -> Path:
    texts = []
    categories = []
    for category, phrases in _TEMPLATES.items():
        for phrase in phrases:
            for index in range(2):
                texts.append(f"{phrase} {index}")
                categories.append(category)
    write_dataset(Dataset(texts=tuple(texts), categories=tuple(categories)), path)
    return path


def test_train_cascade_writes_model_and_settings(tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "models" / "cascade.joblib"
    report = ClassifierService().train_cascade(str(dataset_path), str(model_path))
    assert model_path.exists()
    assert Path(report.report_path).exists()
    assert report.n_train + report.n_validation + report.n_test == 60
    assert 0.0 <= report.threshold <= 1.0
    assert 0.0 <= report.validation_macro_f1 <= 1.0
    assert report.override_categories <= frozenset({"coding", "math", "translation"})
    assert report.rule_precision["coding"] == 1.0
    model = load_cascade_model(str(model_path))
    assert isinstance(model, CascadeModel)
    assert model.threshold == report.threshold
    assert model.override_categories == report.override_categories


def test_evaluate_cascade_reports_sources_and_rates(tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "cascade.joblib"
    ClassifierService().train_cascade(str(dataset_path), str(model_path))
    evaluation = ClassifierService().evaluate(str(model_path))
    assert isinstance(evaluation, CascadeEvaluationReport)
    total = evaluation.n_rules + evaluation.n_classifier + evaluation.n_fallback
    assert evaluation.n_prompts == total
    assert 0.0 <= evaluation.fallback_rate <= 1.0
    assert 0.0 <= evaluation.override_rate <= 1.0
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert 0.0 <= evaluation.threshold <= 1.0


def test_evaluate_cascade_rejects_changed_dataset(tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "cascade.joblib"
    ClassifierService().train_cascade(str(dataset_path), str(model_path))
    with dataset_path.open("a", encoding="utf-8") as handle:
        handle.write('"extra row","coding"\n')
    try:
        ClassifierService().evaluate(str(model_path))
    except ValueError as error:
        assert "differs from the one used" in str(error)
    else:
        raise AssertionError("expected evaluate to reject a changed dataset")


def test_route_service_uses_cascade_rules(tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "cascade.joblib"
    ClassifierService().train_cascade(str(dataset_path), str(model_path))
    model = load_cascade_model(str(model_path))
    from routellm.application.route_service import RouteService

    decision = RouteService(model=model).route("write a python script")
    assert decision.category == "coding"
    assert decision.route == "coding-local"
    assert decision.source == "rules"
    assert decision.confidence is None


def test_cli_cascade_subcommand_writes_model(capsys, tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "cascade.joblib"
    exit_code = cli_main.main(["cascade", "--dataset", str(dataset_path), "--out", str(model_path)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Cascade model trained." in output
    assert "Selected threshold" in output
    assert "Rule override categories" in output
    assert model_path.exists()


def test_cli_route_with_cascade_model(capsys, tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "cascade.joblib"
    assert cli_main.main(["cascade", "--dataset", str(dataset_path), "--out", str(model_path)]) == 0
    capsys.readouterr()
    exit_code = cli_main.main(
        ["route", "--model", str(model_path), "write", "a", "python", "script"]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Category      : coding" in output
    assert "Route         : coding-local" in output
    assert "Source        : rules" in output


def test_cli_route_model_requires_cascade_model(capsys, tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "classifier.joblib"
    ClassifierService().train(
        str(dataset_path), str(model_path), splits_dir=str(tmp_path / "splits")
    )
    exit_code = cli_main.main(
        ["route", "--model", str(model_path), "write", "a", "python", "script"]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "route failed:" in captured.err


def test_cli_evaluate_cascade_prints_cascade_report(capsys, tmp_path):
    dataset_path = _write_keyword_dataset(tmp_path / "prompts.csv")
    model_path = tmp_path / "cascade.joblib"
    assert cli_main.main(["cascade", "--dataset", str(dataset_path), "--out", str(model_path)]) == 0
    capsys.readouterr()
    exit_code = cli_main.main(["evaluate", "--model", str(model_path)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Evaluation of the cascade" in output
    assert "Fallback decisions" in output
    assert "Rule decisions" in output
