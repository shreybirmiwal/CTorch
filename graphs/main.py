from pathlib import Path
import re


LOSS_LINE_RE = re.compile(r"Iteration\s+(\d+)\s+Loss:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def load_loss_points(loss_path: Path) -> tuple[list[int], list[float]]:
    iterations: list[int] = []
    losses: list[float] = []

    with loss_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LOSS_LINE_RE.search(line)
            if match is None:
                continue
            iterations.append(int(match.group(1)))
            losses.append(float(match.group(2)))

    if not iterations:
        raise ValueError(f"No valid loss lines found in {loss_path}")

    return iterations, losses


def main() -> None:
    here = Path(__file__).resolve().parent
    loss_path = here / "loss.txt"
    iterations, losses = load_loss_points(loss_path)
    max_iteration_to_plot = 99
    filtered = [(it, loss) for it, loss in zip(iterations, losses) if it <= max_iteration_to_plot]
    if filtered:
        iterations, losses = map(list, zip(*filtered))
    try:
        import matplotlib.pyplot as plt  # type: ignore

        output_path = here / "loss_plot.png"
        plt.figure(figsize=(10, 5))
        plt.plot(iterations, losses, linewidth=1.5)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        print(f"Saved plot to {output_path}")
        return
    except ModuleNotFoundError:
        pass

    output_path = here / "loss_plot.svg"
    width = 1000
    height = 500
    margin = 50
    inner_w = width - 2 * margin
    inner_h = height - 2 * margin

    min_loss = min(losses)
    max_loss = max(losses)
    loss_span = max(max_loss - min_loss, 1e-12)
    max_iter = max(iterations)
    iter_span = max(max_iter - min(iterations), 1)

    points: list[str] = []
    for it, loss in zip(iterations, losses):
        x = margin + ((it - iterations[0]) / iter_span) * inner_w
        y = margin + (1.0 - ((loss - min_loss) / loss_span)) * inner_h
        points.append(f"{x:.2f},{y:.2f}")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black" stroke-width="1"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black" stroke-width="1"/>
  <polyline fill="none" stroke="steelblue" stroke-width="2" points="{' '.join(points)}"/>
  <text x="{width/2}" y="25" text-anchor="middle" font-size="20">Training Loss</text>
  <text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="14">Iteration</text>
  <text x="18" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {height/2})">Loss</text>
  <text x="{margin}" y="{height-margin+20}" font-size="12">{iterations[0]}</text>
  <text x="{width-margin-40}" y="{height-margin+20}" font-size="12">{max_iter}</text>
  <text x="{margin-40}" y="{margin+5}" font-size="12">{max_loss:.3f}</text>
  <text x="{margin-40}" y="{height-margin+5}" font-size="12">{min_loss:.3f}</text>
</svg>
"""

    output_path.write_text(svg, encoding="utf-8")
    print(f"matplotlib not installed; saved SVG plot to {output_path}")


if __name__ == "__main__":
    main()
