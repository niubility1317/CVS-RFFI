import json
from pathlib import Path


LABELS = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]


def pct(value):
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def pct_list(values):
    return [None if value is None else round(100.0 * float(value), 2) for value in values]


def main():
    root = Path(__file__).resolve().parent / "remote_artifacts"
    files = sorted(root.glob("*14-7_to_3-19*.json"))
    print(f"files={len(files)}")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        row = data["rows"][0]
        history = row.get("history", [])
        eval_history = row.get("target_eval_history", [])
        print("\n###", path.name)
        print(
            "selected",
            pct(row.get("target_accuracy")),
            "best_loss_epoch",
            row.get("best_target_loss_epoch"),
            "best_loss",
            row.get("best_target_loss"),
        )
        print("selected_by_class", pct_list(row.get("target_accuracy_by_class", [])))
        print("selected_pred_hist", row.get("target_pred_hist"))
        if eval_history:
            best_acc = max(eval_history, key=lambda item: item.get("target_accuracy", -1.0))
            min_loss = min(
                eval_history,
                key=lambda item: item.get("target_loss", float("inf"))
                if item.get("target_loss") is not None
                else float("inf"),
            )
            print(
                "eval_best_acc",
                best_acc.get("epoch"),
                pct(best_acc.get("target_accuracy")),
                "loss",
                best_acc.get("target_loss"),
            )
            print(
                "eval_min_loss",
                min_loss.get("epoch"),
                pct(min_loss.get("target_accuracy")),
                "loss",
                min_loss.get("target_loss"),
            )
        if not history:
            continue
        best = max(history, key=lambda item: item.get("target_pred_acc", -1.0))
        print(
            "hist_best",
            best.get("epoch"),
            pct(best.get("target_pred_acc")),
            "selected",
            best.get("target_selected"),
            "pseudo_acc",
            pct(best.get("target_pseudo_selected_acc")),
        )
        print("hist_best_by_class", pct_list(best.get("target_pred_acc_by_true_class", [])))
        print("hist_best_pred_hist", best.get("target_pred_hist"))
        print("hist_best_pseudo_pred_hist", best.get("target_pseudo_selected_hist"))
        print("hist_best_pseudo_true_hist", best.get("target_pseudo_selected_true_hist"))
        print(
            "hist_best_pseudo_acc_true",
            pct_list(best.get("target_pseudo_selected_acc_by_true_class", [])),
        )
        print(
            "hist_best_pseudo_acc_pred",
            pct_list(best.get("target_pseudo_selected_acc_by_pred_class", [])),
        )
        print(
            "hist_best_class_weight_last",
            [round(float(value), 3) for value in best.get("class_weight_last_by_class", [])],
        )
        print(
            "hist_best_threshold_last",
            [round(float(value), 3) for value in best.get("pseudo_threshold_last_by_class", [])],
        )
        print("epoch,acc,worst,selected,pseudo_acc,pred_hist,pseudo_pred_hist")
        for item in history:
            per_class = item.get("target_pred_acc_by_true_class", [])
            worst = min(per_class) if per_class else None
            print(
                ",".join(
                    [
                        str(item.get("epoch")),
                        pct(item.get("target_pred_acc")),
                        pct(worst),
                        str(item.get("target_selected")),
                        pct(item.get("target_pseudo_selected_acc")),
                        repr(item.get("target_pred_hist")),
                        repr(item.get("target_pseudo_selected_hist")),
                    ]
                )
            )


if __name__ == "__main__":
    main()
