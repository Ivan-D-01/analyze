from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def objective_raw(x: np.ndarray) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (x1**2 + x2 - 5) ** 2 + (x1 + x2**2 - 5) ** 2


def gradient_raw(x: np.ndarray) -> np.ndarray:
    x1, x2 = float(x[0]), float(x[1])
    g1 = 4 * x1 * (x1**2 + x2 - 5) + 2 * (x1 + x2**2 - 5)
    g2 = 2 * (x1**2 + x2 - 5) + 4 * x2 * (x1 + x2**2 - 5)
    return np.array([g1, g2], dtype=float)


@dataclass
class EvaluationCounter:
    func_calls: int = 0

    def f(self, x: np.ndarray) -> float:
        self.func_calls += 1
        return objective_raw(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return gradient_raw(x)


@dataclass
class RunResult:
    method: str
    run_id: str
    start: np.ndarray
    x_best: np.ndarray
    f_best: float
    iterations: int
    func_calls: int
    elapsed_time: float
    converged: bool
    stop_reason: str
    history: list[float]
    point_history: list[np.ndarray]


def adam_optimize(
    start: np.ndarray,
    eps_f: float = 1e-8,
    eps_x: float = 1e-8,
    max_iter: int = 5000,
    learning_rate: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> RunResult:
    counter = EvaluationCounter()
    x = np.array(start, dtype=float)
    m = np.zeros(2, dtype=float)
    v = np.zeros(2, dtype=float)
    history: list[float] = []
    point_history: list[np.ndarray] = [x.copy()]

    start_time = time.perf_counter()
    converged = False
    stop_reason = "max_iter"
    f_value = counter.f(x)
    history.append(f_value)

    for iteration in range(1, max_iter + 1):
        grad = counter.grad(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**iteration)
        v_hat = v / (1 - beta2**iteration)
        x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        step_norm = float(np.linalg.norm(x_new - x))
        x = x_new
        f_value = counter.f(x)
        history.append(f_value)
        point_history.append(x.copy())

        if f_value <= eps_f:
            converged = True
            stop_reason = "Значение функции <= eps_f"
            break
        if step_norm <= eps_x:
            converged = True
            stop_reason = "Значение не меняется step_norm <= eps_x"
            break
    else:
        iteration = max_iter

    elapsed_time = time.perf_counter() - start_time
    return RunResult(
        method="Adam",
        run_id=f"start={np.array2string(start, precision=2, separator=', ')}",
        start=np.array(start, dtype=float),
        x_best=x,
        f_best=f_value,
        iterations=iteration,
        func_calls=counter.func_calls,
        elapsed_time=elapsed_time,
        converged=converged,
        stop_reason=stop_reason,
        history=history,
        point_history=point_history,
    )


def build_initial_simplex(start: np.ndarray, step: float) -> np.ndarray:
    x0 = np.array(start, dtype=float)
    simplex = np.array(
        [
            x0,
            x0 + np.array([step, 0.0]),
            x0 + np.array([0.0, step]),
        ],
        dtype=float,
    )
    return simplex


def nelder_mead_optimize(
    start: np.ndarray,
    eps_f: float = 1e-8,
    eps_x: float = 1e-8,
    max_iter: int = 2000,
    simplex_step: float = 0.75,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> RunResult:
    counter = EvaluationCounter()
    # В двумерном случае симплекс состоит из трёх вершин:
    # стартовой точки и двух соседних точек со сдвигом.
    simplex = build_initial_simplex(start, simplex_step)
    values = np.array([counter.f(vertex) for vertex in simplex], dtype=float)
    history: list[float] = [float(values.min())]
    point_history: list[np.ndarray] = [simplex[np.argmin(values)].copy()]

    start_time = time.perf_counter()
    converged = False
    stop_reason = "Достигнуто максимальное число итераций"

    for iteration in range(1, max_iter + 1):
        # Упорядочиваем вершины симплекса:
        # в начале лучшая точка, в конце худшая.
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        best = simplex[0]
        worst = simplex[-1]
        second_worst_value = values[-2]
        # Центроид считаем по всем вершинам, кроме худшей.
        centroid = simplex[:-1].mean(axis=0)

        # Останавливаемся, если лучшее значение функции уже достаточно мало.
        if values[0] <= eps_f:
            converged = True
            stop_reason = "Значение функции <= eps_f"
            break

        # Останавливаемся, если симплекс стал очень маленьким,
        # а значения функции в его вершинах почти совпали.
        simplex_diameter = max(
            float(np.linalg.norm(simplex[i] - best)) for i in range(1, len(simplex))
        )
        if simplex_diameter <= eps_x and float(np.max(np.abs(values - values[0]))) <= eps_f:
            converged = True
            stop_reason = "simplex_tolerance"
            break

        # Отражение: отражаем худшую вершину относительно центроида.
        reflected = centroid + alpha * (centroid - worst)
        reflected_value = counter.f(reflected)

        if values[0] <= reflected_value < second_worst_value:
            # Отражённая точка лучше худшей, значит заменяем ею худшую вершину.
            simplex[-1] = reflected
            values[-1] = reflected_value
        elif reflected_value < values[0]:
            # Отражение оказалось лучше текущей лучшей точки,
            # поэтому пробуем растяжение.
            expanded = centroid + gamma * (reflected - centroid)
            expanded_value = counter.f(expanded)
            if expanded_value < reflected_value:
                simplex[-1] = expanded
                values[-1] = expanded_value
            else:
                simplex[-1] = reflected
                values[-1] = reflected_value
        else:
            # Отражение не дало хорошего улучшения,
            # поэтому пробуем сжатие к центроиду.
            if reflected_value < values[-1]:
                contracted = centroid + rho * (reflected - centroid)
            else:
                contracted = centroid + rho * (worst - centroid)
            contracted_value = counter.f(contracted)

            if contracted_value < values[-1]:
                simplex[-1] = contracted
                values[-1] = contracted_value
            else:
                # Если сжатие тоже не помогло,
                # стягиваем весь симплекс к лучшей вершине.
                for i in range(1, len(simplex)):
                    simplex[i] = best + sigma * (simplex[i] - best)
                    values[i] = counter.f(simplex[i])

        # Сохраняем лучшую точку и лучшее значение после итерации
        # для графиков и итогового отчёта.
        history.append(float(values.min()))
        point_history.append(simplex[np.argmin(values)].copy())
    else:
        iteration = max_iter

    order = np.argsort(values)
    simplex = simplex[order]
    values = values[order]
    elapsed_time = time.perf_counter() - start_time

    return RunResult(
        method="Nelder-Mead",
        run_id=f"start={np.array2string(start, precision=2, separator=', ')}",
        start=np.array(start, dtype=float),
        x_best=simplex[0].copy(),
        f_best=float(values[0]),
        iterations=iteration,
        func_calls=counter.func_calls,
        elapsed_time=elapsed_time,
        converged=converged,
        stop_reason=stop_reason,
        history=history,
        point_history=point_history,
    )


def pso_optimize(
    start: np.ndarray,
    eps_f: float = 1e-8,
    eps_x: float = 1e-8,
    max_iter: int = 800,
    swarm_size: int = 40,
    inertia: float = 0.70,
    c1: float = 1.55,
    c2: float = 1.55,
    bounds: tuple[float, float] = (-6.0, 6.0),
    seed: int = 0,
) -> RunResult:
    counter = EvaluationCounter()
    rng = np.random.default_rng(seed)
    low, high = bounds

    # Начальные позиции частиц генерируем около стартовой точки.
    positions = start + rng.normal(loc=0.0, scale=2.0, size=(swarm_size, 2))
    positions = np.clip(positions, low, high)
    # Первую частицу оставляем ровно в стартовой точке.
    positions[0] = np.array(start, dtype=float)
    # Начальные скорости задаём случайно, но не слишком большими.
    velocities = rng.uniform(-(high - low) * 0.1, (high - low) * 0.1, size=(swarm_size, 2))

    # В начале текущая позиция каждой частицы считается её личным лучшим положением.
    personal_best_positions = positions.copy()
    personal_best_values = np.array([counter.f(position) for position in positions], dtype=float)
    # Глобально лучшую точку выбираем как лучшую среди всех частиц.
    best_index = int(np.argmin(personal_best_values))
    global_best_position = personal_best_positions[best_index].copy()
    global_best_value = float(personal_best_values[best_index])
    # История нужна для графиков: лучшее значение функции и лучшая точка на каждой итерации.
    history: list[float] = [global_best_value]
    point_history: list[np.ndarray] = [global_best_position.copy()]

    start_time = time.perf_counter()
    converged = False
    stop_reason = "max_iter"
    # Эти переменные нужны, чтобы отслеживать застой алгоритма.
    previous_global_best = global_best_position.copy()
    stall_iterations = 0

    for iteration in range(1, max_iter + 1):
        # Случайные коэффициенты делают движение роя стохастическим.
        r1 = rng.random((swarm_size, 2))
        r2 = rng.random((swarm_size, 2))
        # Скорость частицы складывается из инерции,
        # притяжения к своему лучшему положению и к лучшему положению всего роя.
        velocities = (
            inertia * velocities
            + c1 * r1 * (personal_best_positions - positions)
            + c2 * r2 * (global_best_position - positions)
        )
        # Обновляем позиции и не даём частицам выйти за область поиска.
        positions = np.clip(positions + velocities, low, high)

        # Обновляем личный лучший результат каждой частицы.
        for i in range(swarm_size):
            value = counter.f(positions[i])
            if value < personal_best_values[i]:
                personal_best_values[i] = value
                personal_best_positions[i] = positions[i].copy()

        # После обновления личных рекордов проверяем глобальный лучший результат роя.
        best_index = int(np.argmin(personal_best_values))
        if personal_best_values[best_index] < global_best_value:
            global_best_value = float(personal_best_values[best_index])
            global_best_position = personal_best_positions[best_index].copy()

        # Сохраняем историю лучшего найденного значения и траекторию лучшей точки.
        history.append(global_best_value)
        point_history.append(global_best_position.copy())
        # best_step показывает, насколько сдвинулась лучшая точка,
        # best_value_change — насколько изменилось лучшее значение функции.
        best_step = float(np.linalg.norm(global_best_position - previous_global_best))
        best_value_change = abs(history[-1] - history[-2])
        previous_global_best = global_best_position.copy()

        # Останавливаемся, если значение функции уже достаточно мало.
        if global_best_value <= eps_f:
            converged = True
            stop_reason = "Значение функции <= eps_f"
            break

        # Если и точка, и значение функции почти не меняются,
        # считаем это одной итерацией застоя.
        if best_step <= eps_x and best_value_change <= eps_f:
            stall_iterations += 1
        else:
            stall_iterations = 0

        # Если застой длится долго, считаем, что алгоритм фактически сошёлся.
        if stall_iterations >= 40:
            converged = True
            stop_reason = "Значение не меняется уже 40 итераций"
            break
    else:
        iteration = max_iter

    elapsed_time = time.perf_counter() - start_time
    return RunResult(
        method="PSO",
        run_id=f"start={np.array2string(start, precision=2, separator=', ')}, seed={seed}",
        start=np.array(start, dtype=float),
        x_best=global_best_position,
        f_best=global_best_value,
        iterations=iteration,
        func_calls=counter.func_calls,
        elapsed_time=elapsed_time,
        converged=converged,
        stop_reason=stop_reason,
        history=history,
        point_history=point_history,
    )


def run_experiments() -> list[RunResult]:
    starts = [
        np.array([10.0, 15.0]),
        np.array([3.0, 3.0]),
        np.array([-3.0, -3.0]),
        np.array([2.0, -2.0]),
        np.array([-2.0, 2.0]),
    ]
    pso_seeds = [11, 23, 37]
    results: list[RunResult] = []

    for start in starts:
        results.append(adam_optimize(start))
        results.append(nelder_mead_optimize(start))
        for seed in pso_seeds:
            results.append(pso_optimize(start, seed=seed))

    return results


def summarize_by_method(results: list[RunResult]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    methods = sorted({result.method for result in results})
    for method in methods:
        method_results = [result for result in results if result.method == method]
        success_count = sum(1 for result in method_results if result.converged)
        summary.append(
            {
                "method": method,
                "runs": len(method_results),
                "successes": success_count,
                "best_f": min(result.f_best for result in method_results),
                "avg_time": sum(result.elapsed_time for result in method_results) / len(method_results),
                "avg_func_calls": sum(result.func_calls for result in method_results)
                / len(method_results),
            }
        )
    return summary


def detect_distinct_minima(results: list[RunResult], tolerance: float = 1e-3) -> list[np.ndarray]:
    minima: list[np.ndarray] = []
    successful_points = [
        result.x_best
        for result in results
        if result.f_best <= 1e-6
    ]
    for point in successful_points:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in minima):
            minima.append(point.copy())
    return minima


def best_result_per_method(results: list[RunResult]) -> dict[str, RunResult]:
    best: dict[str, RunResult] = {}
    for result in results:
        if result.method not in best or result.f_best < best[result.method].f_best:
            best[result.method] = result
    return best


def longest_result_per_method(results: list[RunResult]) -> dict[str, RunResult]:
    longest: dict[str, RunResult] = {}
    for result in results:
        if result.method not in longest or result.iterations > longest[result.method].iterations:
            longest[result.method] = result
    return longest


def format_point(point: np.ndarray) -> str:
    return f"({point[0]:.12f}, {point[1]:.12f})"


def format_point_pdf(point: np.ndarray) -> str:
    return f"({point[0]:.2f}, {point[1]:.2f})"


def build_conclusion(results: list[RunResult]) -> list[str]:
    summary = summarize_by_method(results)
    minima = detect_distinct_minima(results)
    best_by_method = best_result_per_method(results)

    fastest = min(summary, key=lambda item: item["avg_time"])
    cheapest = min(summary, key=lambda item: item["avg_func_calls"])
    most_reliable = max(summary, key=lambda item: item["successes"] / item["runs"])

    lines = [
        "Краткий вывод:",
        f"1. Найдено различных минимумов с f(x) ~= 0: {len(minima)}.",
        f"2. Наиболее быстрый по среднему времени: {fastest['method']}.",
        f"3. Наиболее экономный по вызовам функции: {cheapest['method']}.",
        f"4. Наиболее стабильный по числу успешных запусков: {most_reliable['method']}.",
    ]

    for method, result in best_by_method.items():
        lines.append(
            f"{method}: лучшая точка {format_point(result.x_best)}, "
            f"f = {result.f_best:.3e}, итерации = {result.iterations}."
        )

    lines.append(
        "Практически для этой задачи Adam обычно выигрывает по скорости при хорошем старте, "
        "Nelder-Mead удобен без производных, а PSO полезен как более глобальный метод, "
        "но платит за это большим числом вычислений."
    )
    return lines


def build_conclusion_pdf(results: list[RunResult]) -> list[str]:
    summary = summarize_by_method(results)
    minima = detect_distinct_minima(results)
    best_by_method = best_result_per_method(results)

    fastest = min(summary, key=lambda item: item["avg_time"])
    cheapest = min(summary, key=lambda item: item["avg_func_calls"])
    most_reliable = max(summary, key=lambda item: item["successes"] / item["runs"])

    lines = [
        "Краткий вывод:",
        f"1. Найдено различных минимумов с f(x) ~= 0: {len(minima)}.",
        f"2. Наиболее быстрый по среднему времени: {fastest['method']}.",
        f"3. Наиболее экономный по вызовам функции: {cheapest['method']}.",
        f"4. Наиболее стабильный по числу успешных запусков: {most_reliable['method']}.",
    ]

    for method, result in best_by_method.items():
        lines.append(
            f"{method}: лучшая точка {format_point_pdf(result.x_best)}, "
            f"f = {result.f_best:.2e}, iterations = {result.iterations}."
        )

    lines.append(
        "Практически для этой задачи Adam обычно выигрывает по скорости при хорошем старте, "
        "Nelder-Mead удобен без производных, а PSO полезен как более глобальный метод, "
        "но платит за это большим числом вычислений."
    )
    return lines


def add_table_page(
    pdf: PdfPages,
    title: str,
    columns: list[str],
    rows: list[list[str]],
    font_size: int = 9,
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=14)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9e8fb")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f6f8")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_plot_page(
    pdf: PdfPages,
    results: list[RunResult],
    selected_runs: dict[str, RunResult],
    title_suffix: str,
    convergence_title: str,
    run_label: str,
    show_iterations: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    contour_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    convergence_ax = axes[1, 1]
    x = np.linspace(-20.5, 20.5, 250)
    y = np.linspace(-20.5, 20.5, 250)
    xx, yy = np.meshgrid(x, y)
    zz = (xx**2 + yy - 5) ** 2 + (xx + yy**2 - 5) ** 2

    colors = {"Adam": "#d1495b", "Nelder-Mead": "#00798c", "PSO": "#edae49"}
    methods = ["Adam", "Nelder-Mead", "PSO"]
    for ax, method in zip(contour_axes, methods):
        method_results = [result for result in results if result.method == method]
        #contour = ax.contourf(xx, yy, np.log10(zz + 1e-6), levels=30, cmap="viridis")
        contour = ax.contourf(xx, yy, zz, levels=30, cmap="viridis")
        ax.scatter(
            [result.x_best[0] for result in method_results],
            [result.x_best[1] for result in method_results],
            s=40,
            color=colors[method],
            edgecolors="black",
            linewidths=0.3,
            alpha=0.85,
        )
        ax.set_title(f"{method}: найденные точки")
        trajectory = np.array(selected_runs[method].point_history, dtype=float)
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="white",
            linewidth=1.8,
            alpha=0.95,
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=run_label,
        )
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            marker="s",
            s=55,
            color="white",
            edgecolors="black",
            linewidths=0.5,
            label="старт",
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            marker="*",
            s=120,
            color="white",
            edgecolors="black",
            linewidths=0.5,
            label="финиш",
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if show_iterations:
            ax.text(
                0.03,
                0.03,
                f"iterations = {selected_runs[method].iterations}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 4},
            )
        ax.legend(loc="upper right")
    #fig.colorbar(contour, ax=contour_axes, shrink=0.82, pad=0.04, label="log10(f(x) + 1e-6)")
    fig.colorbar(contour, ax=contour_axes, shrink=0.82, pad=0.04, label="f(x)")

    for method, result in selected_runs.items():
        values = np.maximum(np.array(result.history, dtype=float), 1e-16)
        convergence_ax.semilogy(values, label=method, color=colors[method], linewidth=2)

    convergence_ax.set_title("Сходимость лучших запусков")
    convergence_ax.set_xlabel("Итерация")
    convergence_ax.set_ylabel("f(x)")
    convergence_ax.grid(True, which="both", alpha=0.3)
    convergence_ax.legend()

    pdf.savefig(fig)
    plt.close(fig)


def add_conclusion_page(pdf: PdfPages, results: list[RunResult]) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("Краткий отчет", fontsize=16, pad=18)

    conclusion_lines = build_conclusion_pdf(results)
    text = "\n".join(conclusion_lines)
    ax.text(0.03, 0.95, text, va="top", ha="left", fontsize=12, wrap=True)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def create_pdf_report(results: list[RunResult], output_path: Path) -> None:
    detailed_columns = [
        "Метод",
        "Запуск",
        "Старт",
        "Лучшая точка",
        "f_best",
        "Итерации",
        "f calls",
        "Время, миллисекунды"
    ]
    detailed_results = sorted(results, key=lambda result: result.f_best, reverse=True)
    detailed_rows = [
        [
            result.method,
            result.run_id,
            format_point_pdf(result.start),
            format_point_pdf(result.x_best),
            f"{result.f_best:.2e}",
            str(result.iterations),
            str(result.func_calls),
            f"{result.elapsed_time * 1000:.2f}"
        ]
        for result in detailed_results
    ]

    summary_columns = [
        "Метод",
        "Запусков",
        "Успешно",
        "Лучшая f",
        "Среднее время, c",
        "Средние f calls",
    ]
    summary_rows = [
        [
            str(item["method"]),
            str(item["runs"]),
            str(item["successes"]),
            f"{float(item['best_f']):.2e}",
            f"{float(item['avg_time']) * 1000:.2f}",
            f"{float(item['avg_func_calls']):.2f}",
        ]
        for item in summarize_by_method(results)
    ]

    with PdfPages(output_path) as pdf:
        add_conclusion_page(pdf, results)
        add_table_page(pdf, "Сводная таблица по методам", summary_columns, summary_rows, font_size=10)
        add_table_page(pdf, "Подробные результаты запусков (сортировка по f_best по убыванию)", detailed_columns, detailed_rows, font_size=7)
        add_plot_page(
            pdf,
            results,
            best_result_per_method(results),
            "best convergence trajectory",
            "Best-run convergence",
            "траектория лучшего запуска",
        )
        add_plot_page(
            pdf,
            results,
            longest_result_per_method(results),
            "longest trajectory",
            "Longest-run convergence",
            "Самый длинный запуск по кол-ву итераций",
            show_iterations=True,
        )


def print_console_summary(results: list[RunResult]) -> None:
    print("Summary by method:")
    for item in summarize_by_method(results):
        print(
            f"{item['method']}: runs={item['runs']}, successes={item['successes']}, "
            f"best_f={item['best_f']:.3e}, avg_time={item['avg_time']:.4f}s, "
            f"avg_f_calls={item['avg_func_calls']:.1f}"
        )

    print("\nDetailed runs:")
    for result in results:
        print(
            f"{result.method:12} | {result.run_id:34} | "
            f"x={format_point(result.x_best):20} | f={result.f_best:.3e} | "
            f"iter={result.iterations:4d} | f_calls={result.func_calls:4d} | "
            f"time={result.elapsed_time:.4f}s | "
            f"stop={result.stop_reason}"
        )


def main() -> None:
    results = run_experiments()
    output_path = Path(__file__).with_name("optimization_report.pdf")
    create_pdf_report(results, output_path)
    print_console_summary(results)
    print(f"\nPDF report saved to: {output_path}")


if __name__ == "__main__":
    main()
