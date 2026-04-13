"""
Сравнительная оценка трёх движков:
  1. TF-IDF (базовый)
  2. TF-IDF + синонимы
  3. Sentence-Transformers

Запуск: python eval_compare.py
Результат: таблица метрик + графики в папке results/
"""

import os
import json
import time
import numpy as np
from nlp_utils import load_knowledge_base
from engines import TfidfEngine, TfidfSynonymEngine, AdvancedTfidfEngine

# ── Тестовая выборка ─────────────────────────────────────────
# (перефразированный вопрос, ожидаемый id | None для мусора)
TEST_CASES = [
    # Дедлайны
    ("когда крайний срок сдачи курсовой", 1),
    ("срок сдачи работы", 1),
    ("можно ли перенести дедлайн", 2),
    ("дайте больше времени на работу", 2),
    ("когда пересдавать экзамен", 3),
    ("до которого часа принимаете работы", 4),
    ("когда выложат оценки за экзамен", 5),
    # Материалы
    ("где найти методичку", 6),
    ("дайте слайды с лекций", 7),
    ("какую литературу читать к экзамену", 8),
    ("есть ли видеозаписи ваших лекций", 10),
    # Оценки
    ("сколько у меня баллов сейчас", 11),
    ("за что снизили оценку за дз", 12),
    ("можно переписать тест", 13),
    ("какой балл нужен чтобы не сдавать экзамен", 14),
    # Организация
    ("когда можно подойти на консультацию", 16),
    ("в каком кабинете будет пара", 18),
    ("как вам написать", 20),
    # Формат
    ("как правильно оформить лабу", 21),
    ("принимаете рукописные работы", 22),
    ("в каком виде отправлять программу", 23),
    ("обязателен ли титульник", 24),
    ("можно ли использовать chatgpt", 25),
    # Экзамен
    ("что спросят на экзамене", 26),
    ("разрешены ли шпоры", 27),
    ("сколько длится экзамен", 28),
    ("можно сдать экзамен раньше", 30),
    # Техника
    ("не получается загрузить файл на сайт", 31),
    ("компьютер сломался что делать", 32),
    ("не работает ссылка на moodle", 34),
    # Мусор (ожидаем отказ)
    ("какая погода сегодня", None),
    ("посоветуй фильм на вечер", None),
    ("сколько будет 2+2", None),
]


def evaluate_engine(engine, test_cases):
    """Прогоняет тесты через движок, возвращает метрики."""
    correct = 0
    tp, fp, fn = 0, 0, 0
    scores = []
    details = []

    t0 = time.time()
    for question, expected_id in test_cases:
        result = engine.find_answer(question)
        actual_id = result.get("matched_id")
        score = result["score"]
        scores.append(score)

        if expected_id is None:
            # Мусорный вопрос
            if actual_id is None:
                correct += 1
                status = "TN"
            else:
                fp += 1
                status = "FP"
        else:
            if actual_id is None:
                fn += 1
                status = "FN"
            elif actual_id == expected_id:
                correct += 1
                tp += 1
                status = "TP"
            else:
                status = f"WRONG(exp={expected_id},got={actual_id})"

        details.append({
            "question": question,
            "expected": expected_id,
            "actual": actual_id,
            "score": round(score, 4),
            "status": status,
        })

    elapsed = time.time() - t0
    total = len(test_cases)
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_score": float(np.mean(scores)),
        "time_sec": elapsed,
        "details": details,
    }


def print_table(results: dict):
    """Печатает сводную таблицу."""
    header = f"{'Движок':<28} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Avg Score':>9} {'Время':>8}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, r in results.items():
        print(
            f"{name:<28} {r['accuracy']:>7.1%} {r['precision']:>9.1%} "
            f"{r['recall']:>7.1%} {r['f1']:>7.3f} {r['avg_score']:>9.3f} "
            f"{r['time_sec']:>7.2f}s"
        )
    print(sep)


def print_details(results: dict):
    """Печатает подробности по каждому вопросу."""
    engines = list(results.keys())
    print(f"\n{'Вопрос':<45}", end="")
    for e in engines:
        short = e[:12]
        print(f" {short:>12}", end="")
    print()
    print("─" * (45 + 13 * len(engines)))

    for i, (question, expected_id) in enumerate(TEST_CASES):
        label = question[:43]
        print(f"{label:<45}", end="")
        for name in engines:
            d = results[name]["details"][i]
            st = d["status"]
            if st in ("TP", "TN"):
                mark = f"✅ {d['score']:.2f}"
            elif st == "FN":
                mark = f"❌ FN"
            elif st == "FP":
                mark = f"❌ FP"
            else:
                mark = f"❌ {d['score']:.2f}"
            print(f" {mark:>12}", end="")
        print()


def generate_charts(results: dict):
    """Генерирует PNG-графики сравнения (matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib не установлен, графики не сгенерированы")
        return

    os.makedirs("results", exist_ok=True)

    names = list(results.keys())
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    # ── 1. Барчарт метрик ────────────────────────────────────
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Значение")
    ax.set_title("Сравнение метрик трёх движков")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/metrics_comparison.png", dpi=150)
    plt.close()
    print("📊 Сохранено: results/metrics_comparison.png")

    # ── 2. Распределение scores ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, name in enumerate(names):
        scores = [d["score"] for d in results[name]["details"]]
        axes[i].hist(scores, bins=15, color=colors[i], alpha=0.8, edgecolor="white")
        axes[i].axvline(x=0.15, color="red", linestyle="--", linewidth=1, label="Порог 0.15")
        axes[i].set_title(name)
        axes[i].set_xlabel("Cosine Similarity")
        axes[i].legend(fontsize=8)
    axes[0].set_ylabel("Количество вопросов")
    fig.suptitle("Распределение уверенности (scores) по движкам", fontsize=13)
    fig.tight_layout()
    fig.savefig("results/score_distribution.png", dpi=150)
    plt.close()
    print("📊 Сохранено: results/score_distribution.png")

    # ── 3. Heatmap: score каждого вопроса по движкам ─────────
    fig, ax = plt.subplots(figsize=(8, 12))
    data = []
    labels = []
    for q, _ in TEST_CASES:
        labels.append(q[:40])
    for name in names:
        col = [d["score"] for d in results[name]["details"]]
        data.append(col)
    data = np.array(data).T  # questions x engines
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Уверенность (score) по каждому вопросу")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Cosine Similarity")
    fig.tight_layout()
    fig.savefig("results/score_heatmap.png", dpi=150)
    plt.close()
    print("📊 Сохранено: results/score_heatmap.png")


def save_json_report(results: dict):
    """Сохраняет полный JSON-отчёт."""
    os.makedirs("results", exist_ok=True)
    report = {}
    for name, r in results.items():
        report[name] = {
            "accuracy": round(r["accuracy"], 4),
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "avg_score": round(r["avg_score"], 4),
            "time_sec": round(r["time_sec"], 4),
            "details": r["details"],
        }
    with open("results/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("📄 Сохранено: results/eval_report.json")


def main():
    print("Загрузка базы знаний...")
    qa_data = load_knowledge_base()

    print("Инициализация движков...")
    engines = [
        TfidfEngine(qa_data),
        TfidfSynonymEngine(qa_data),
        AdvancedTfidfEngine(qa_data),
    ]

    results = {}
    for eng in engines:
        print(f"  Тестирование: {eng.name}...")
        results[eng.name] = evaluate_engine(eng, TEST_CASES)

    print_table(results)
    print_details(results)
    generate_charts(results)
    save_json_report(results)

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
