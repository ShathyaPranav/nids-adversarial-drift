"""Clean-performance metrics matching the synopsis's evaluation protocol:
accuracy, precision/recall, macro-F1 -- with per-class F1 broken out separately so a
single macro-F1 number can't hide collapse on rare classes (SQL Injection, XSS, etc.).
"""
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

RARE_CLASS_SAMPLE_THRESHOLD = 1000


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    per_class = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for name, p, r, f, s in zip(class_names, precision, recall, f1, support)
    }
    rare_classes = {name: v for name, v in per_class.items() if v["support"] < RARE_CLASS_SAMPLE_THRESHOLD}

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "rare_classes": rare_classes,
    }


def format_report(metrics: dict, class_names: list[str]) -> str:
    lines = [
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro-F1: {metrics['macro_f1']:.4f}",
        "",
        "Per-class:",
    ]
    for name in class_names:
        v = metrics["per_class"][name]
        flag = "  <- rare class" if name in metrics["rare_classes"] else ""
        lines.append(
            f"  {name:20s} P={v['precision']:.3f} R={v['recall']:.3f} "
            f"F1={v['f1']:.3f} n={v['support']}{flag}"
        )
    return "\n".join(lines)
