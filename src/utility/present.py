"""
Terminal Presentation — Loan Default Risk Prediction Hackathon
Navigate: [→] / [Space] / [Enter] = Next slide   [←] / [Backspace] = Previous   [Q] = Quit
"""
from __future__ import annotations
import sys
import os

# Windows keyboard input
if sys.platform == "win32":
    import msvcrt
else:
    import tty, termios

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich.align import Align
from rich import box

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# SLIDE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

def slide_title():
    console.print()
    console.print(Align.center(
        Text("LOAN DEFAULT RISK PREDICTION", style="bold white on dark_blue", justify="center")
    ))
    console.print(Align.center(Text("AI-Driven Multi-Model Ensemble", style="bold cyan")))
    console.print()
    console.print(Rule(style="blue"))
    console.print()
    grid = Table.grid(expand=True)
    grid.add_column(justify="center")
    grid.add_row(Text("🏦  Home Credit Risk Dataset", style="bold yellow"))
    grid.add_row(Text("246,008 applicants  ·  190 features  ·  5-fold CV", style="white"))
    grid.add_row(Text("LightGBM · XGBoost · CatBoost · Random Forest · TabNet · Stacking", style="dim cyan"))
    console.print(Panel(grid, border_style="blue", padding=(1, 4)))
    console.print()
    console.print(Align.center(Text("Best CV AUC: 0.764  (CatBoost)", style="bold green")))


def slide_problem():
    console.print(Panel(Text("1.  PROBLEM STATEMENT", style="bold white"), style="on dark_blue", padding=(0,2)))
    console.print()
    left = Panel(
        "\n[bold]What Are We Solving?[/bold]\n\n"
        "Financial institutions struggle to\nidentify borrowers who will default.\n\n"
        "Traditional methods fail to:\n"
        "  • Handle 246k+ applicant records\n"
        "  • Capture hidden risk patterns\n"
        "  • Adapt to complex signals\n\n"
        "[cyan]→ We predict default probability\n   using AI-driven ensemble[/cyan]",
        title="Problem", border_style="red", padding=(0,1)
    )
    right = Panel(
        "\n[bold]Why Does It Matter?[/bold]\n\n"
        "Poor risk assessment causes:\n"
        "  ✗ Financial losses for lenders\n"
        "  ✗ Higher interest rates\n"
        "  ✗ Credit denied to good borrowers\n\n"
        "[bold green]Our solution helps:[/bold green]\n"
        "  ✓ Reduce loan defaults\n"
        "  ✓ Improve financial inclusion\n"
        "  ✓ Enable smarter lending",
        title="Impact", border_style="green", padding=(0,1)
    )
    console.print(Columns([left, right], equal=True))
    console.print()
    t = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
    t.add_column(style="bold yellow"); t.add_column(style="dim")
    t.add_row("Users", "Banks · NBFCs · Fintech lenders · Credit risk teams")
    t.add_row("Existing Gap", "Rule-based ✗ Static   Credit scores ✗ Not adaptive   Manual ✗ No scale")
    console.print(Panel(t, border_style="yellow", title="Stakeholders & Market Gap"))


def slide_dataset():
    console.print(Panel(Text("2.  DATASET & PREPROCESSING", style="bold white"), style="on dark_blue", padding=(0,2)))
    console.print()

    stats = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    stats.add_column(style="bold yellow", width=22)
    stats.add_column(style="white")
    stats.add_row("Training samples", "246,008 applications")
    stats.add_row("Test samples", "48,744 applications")
    stats.add_row("Engineered features", "190 (started at ~120 raw)")
    stats.add_row("Class imbalance", "~8% defaults  →  scale_pos_weight=11")
    stats.add_row("Anomalies fixed", "DAYS_EMPLOYED=365243 | PHONE_CHANGE=0")

    feat = Table(title="14 Feature Groups", box=box.SIMPLE_HEAD, border_style="green", header_style="bold green")
    feat.add_column("Group", style="cyan", width=22)
    feat.add_column("Key Features", width=38)
    feat.add_column("N", justify="right", style="bold yellow")
    rows = [
        ("Amount Ratios", "CREDIT_TERM, PAYMENT_TO_INCOME", "6"),
        ("EXT_SOURCE Stats", "Mean, Harmonic, Geometric, Range", "11"),
        ("EXT_SOURCE Cross", "EXT_1_TO_BIRTH, EXT_2_TO_CREDIT", "16"),
        ("Family Burden", "CREDIT_PER_FAM, INCOME_PER_CHILD", "4"),
        ("Building Info", "BUILDING_AVG_MEAN, LIVING_TO_TOTAL", "5"),
        ("Nonlinear Stress", "PAYMENT_SQ, CREDIT_SQ, TERM_SQ", "4"),
        ("Region × Risk", "RATING×EXT2, CITY×CREDIT", "3"),
        ("Target Encoding", "OCCUPATION_TE, ORGANIZATION_TE", "2"),
        ("Temporal", "AGE_YEARS, EMPLOYED_YEARS", "8"),
        ("+ 5 more groups", "Missingness, Docs, Bins, Income, Social", "31"),
    ]
    for r in rows:
        feat.add_row(*r)

    console.print(Columns([stats, feat]))


def slide_architecture():
    console.print(Panel(Text("3.  MODEL ARCHITECTURE", style="bold white"), style="on dark_blue", padding=(0,2)))
    console.print()

    t = Table(title="6 Independent Pipelines", box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    t.add_column("Model", style="bold yellow", width=18)
    t.add_column("File", style="dim", width=22)
    t.add_column("Strength")
    t.add_row("LightGBM",        "src/models/lgbm.py",    "Speed · NaN-native · categoricals")
    t.add_row("XGBoost",         "src/models/xgb.py",     "Regularisation · reliable")
    t.add_row("CatBoost",        "src/models/catboost.py","Best on categoricals · robust")
    t.add_row("Random Forest",   "src/models/rf.py",      "Diverse errors · interpretable")
    t.add_row("TabNet  🔥",      "src/models/tabnet.py",  "Attention DL · GTX 1650 GPU (CUDA 12.6)")
    t.add_row("Stacking",        "src/models/stacking.py","Bayesian Ridge meta-model on OOF")
    console.print(t)
    console.print()

    strategy = Table.grid(padding=(0,2))
    strategy.add_column(style="bold cyan"); strategy.add_column(style="white")
    strategy.add_row("CV",       "5-fold Stratified K-Fold (preserves 8% ratio per fold)")
    strategy.add_row("Tuning",   "Optuna Bayesian TPE · 50 trials · data loaded once (4-10× faster)")
    strategy.add_row("Stopping", "LGBM: 50 rounds · CatBoost: od_wait=50 · TabNet: patience=20")
    strategy.add_row("Ablation", "Leave-one-group-out across 14 feature groups → pruned log_transforms")
    console.print(Panel(strategy, title="Training Strategy", border_style="green"))


def slide_performance():
    console.print(Panel(Text("4.  MODEL PERFORMANCE", style="bold white"), style="on dark_blue", padding=(0,2)))
    console.print()
    console.print(Align.center(Text(
        "Metric: ROC-AUC  (accuracy is misleading at 92:8 imbalance)",
        style="bold yellow"
    )))
    console.print()

    results = Table(title="5-Fold Stratified CV Results", box=box.ROUNDED,
                    border_style="green", header_style="bold green")
    results.add_column("Model", style="bold white", width=28)
    results.add_column("CV AUC", justify="center", style="bold", width=10)
    results.add_column("Notes")
    results.add_row("Raw baseline (no engineering)", "0.745", "105 features, no transforms", style="dim")
    results.add_row("XGBoost",                       "0.744", "190 features")
    results.add_row("Stacking Ensemble",              "0.752", "LGBM+XGB+CatBoost OOF blend")
    results.add_row("LightGBM",                       "0.755", "190 features")
    results.add_row("[bold green]CatBoost  ★[/bold green]",
                                                     "[bold green]0.764[/bold green]",
                                                     "[bold green]190 features — BEST single model[/bold green]")
    console.print(results)
    console.print()

    prog = Table(title="AUC Progression from Feature Engineering", box=box.SIMPLE,
                 header_style="bold cyan")
    prog.add_column("Stage", style="cyan")
    prog.add_column("Features", justify="right")
    prog.add_column("AUC", justify="right", style="bold")
    prog.add_column("Bar")
    data = [
        ("Raw baseline",       "105", "0.745", "█████████████████████████░░░░░"),
        ("+ Amount ratios",    "111", "0.748", "███████████████████████████░░░"),
        ("+ EXT_SOURCE stats", "122", "0.753", "████████████████████████████░░"),
        ("+ All new groups",   "190", "0.764", "██████████████████████████████"),
    ]
    colors = ["dim", "dim", "white", "bold green"]
    for (row, col) in zip(data, colors):
        prog.add_row(*row, style=col)
    console.print(prog)


def slide_innovation():
    console.print(Panel(Text("5.  INNOVATION & OPTIMIZATION", style="bold white"), style="on dark_blue", padding=(0,2)))
    console.print()

    innov = Table(box=box.SIMPLE, show_header=False, padding=(0,1))
    innov.add_column(style="bold yellow", width=4)
    innov.add_column(style="bold cyan", width=28)
    innov.add_column(style="white")
    innov.add_row("①", "Ablation-Driven FE",    "Measured each of 14 groups → removed log_transforms (hurt AUC)")
    innov.add_row("②", "14 Custom Groups",      "Harmonic/geometric EXT scores · nonlinear stress · target encoding")
    innov.add_row("③", "Sentinel Fix",          "DAYS_EMPLOYED=365243 & PHONE_CHANGE=0 → would silently produce inf")
    innov.add_row("④", "TabNet GPU",            "Attention DL on GTX 1650 · custom −1 embedding fix · different errors")
    console.print(Panel(innov, title="What Makes Us Different", border_style="magenta"))
    console.print()

    ablation = Table(title="Ablation: AUC Drop When Group Removed", box=box.SIMPLE_HEAD,
                     header_style="bold red")
    ablation.add_column("Feature Group", width=26)
    ablation.add_column("AUC Drop", justify="right", width=10)
    ablation.add_column("Importance Bar", width=30)
    abl_data = [
        ("flag_aggregations",    "−0.00180", "[red]████████████████████[/red]"),
        ("ext_source_aggs",      "−0.00178", "[red]████████████████████[/red]"),
        ("ext_source_cross",     "−0.00136", "[red]███████████████[/red]"),
        ("temporal",             "−0.00074", "[yellow]████████[/yellow]"),
        ("social_circle",        "−0.00073", "[yellow]████████[/yellow]"),
        ("amount_ratios",        "−0.00036", "[yellow]████[/yellow]"),
        ("bins (removed)",       "+0.00001", "[green]▪ neutral[/green]"),
    ]
    for row in abl_data:
        ablation.add_row(*row)
    console.print(ablation)


def slide_viability():
    console.print(Panel(Text("6.  REAL-WORLD VIABILITY", style="bold white"), style="on dark_blue", padding=(0,2)))
    console.print()

    left = Panel(
        "\n[bold cyan]Deployment Options[/bold cyan]\n\n"
        "  • REST API scoring endpoint\n"
        "  • Cloud scoring service\n"
        "  • Internal bank decision engine\n\n"
        "[bold yellow]Works with:[/bold yellow]\n"
        "  ✓ Batch processing\n"
        "  ✓ Real-time scoring\n"
        "  ✓ API integration\n\n"
        "[dim]LGBM/CatBoost: <1ms per applicant\nTabNet GPU: optional[/dim]",
        title="Deployment", border_style="cyan", padding=(0,1)
    )
    right = Panel(
        "\n[bold green]Business Impact[/bold green]\n\n"
        "  ✓ Direct reduction in NPA exposure\n"
        "  ✓ Risk-based pricing (not flat rates)\n"
        "  ✓ Portfolio-level risk visibility\n"
        "  ✓ Automated underwriting support\n\n"
        "[bold magenta]Social Impact[/bold magenta]\n\n"
        "  ✓ Credit expansion without risk\n"
        "  ✓ Fairer, bias-reduced decisions\n"
        "  ✓ Financial system stability",
        title="Impact", border_style="green", padding=(0,1)
    )
    console.print(Columns([left, right], equal=True))


def slide_demo():
    console.print(Panel(Text("7.  DEMO / PROTOTYPE", style="bold white"), style="on dark_blue", padding=(0,2)))
    console.print()
    layers = [
        ("📂 load_data.py",   "cyan",    "Loads train (246k) + test (48k) · schema validation"),
        ("🔧 preprocess.py",  "yellow",  "Fixes DAYS_EMPLOYED=365243 · PHONE_CHANGE=0 · label encodes"),
        ("⚙  features.py",   "green",   "Builds 190 features across 14 groups · target encoding"),
        ("🤖 models/*",       "magenta", "LGBM · XGB · CatBoost · RF · TabNet(GPU) — each 5-fold CV"),
        ("📊 validation/",    "blue",    "Stratified K-Fold · ROC-AUC per fold · ablation study"),
        ("🔗 stacking.py",    "red",     "Bayesian Ridge meta-model on OOF predictions"),
        ("🚀 inference.py",   "white",   "Generates submission.csv  →  SK_ID_CURR + TARGET probability"),
    ]
    for (layer, color, desc) in layers:
        console.print(f"  [{color}]{layer:22s}[/{color}]  {desc}")
        if layer != layers[-1][0]:
            console.print(f"  [dim]{'':22s}  ↓[/dim]")
    console.print()
    console.print(Panel(
        "  [bold]Run:[/bold]  python -m src.train --models lgbm xgb cat rf tabnet stack\n"
        "  [bold]Then:[/bold] python -m src.inference  →  submission.csv",
        border_style="dim", title="One-Command Pipeline"
    ))


def slide_conclusion():
    console.print(Panel(Text("8.  CONCLUSION", style="bold white"), style="on dark_blue", padding=(0,2)))
    console.print()
    points = [
        ("Transforms",  "raw applicant data into actionable default risk insights"),
        ("Enables",     "lenders to grow approvals without proportionally increasing risk"),
        ("Moves",       "credit decisions from rule-based judgment to predictive intelligence"),
        ("Creates",     "a scalable foundation for automated underwriting"),
        ("Ablation",    "discipline ensures every feature earns its place — no bloat"),
    ]
    for i, (bold, rest) in enumerate(points):
        console.print(f"  [bold green]✓[/bold green]  [bold white]{bold}[/bold white] {rest}")
        console.print()
    console.print(Rule(style="green"))
    console.print()
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0,3))
    summary.add_column(style="bold cyan"); summary.add_column(style="bold yellow")
    summary.add_row("Dataset",  "246,008 applications · 190 features")
    summary.add_row("Best AUC", "0.764 (CatBoost, 5-fold CV)")
    summary.add_row("Models",   "LGBM · XGB · CatBoost · RF · TabNet · Stack")
    summary.add_row("Pipeline", "Fully automated — one command end-to-end")
    console.print(Align.center(summary))


# ──────────────────────────────────────────────────────────────────────────────
SLIDES = [
    ("Title",              slide_title),
    ("Problem Statement",  slide_problem),
    ("Dataset & Prep",     slide_dataset),
    ("Model Architecture", slide_architecture),
    ("Performance",        slide_performance),
    ("Innovation",         slide_innovation),
    ("Real-World",         slide_viability),
    ("Demo / Prototype",   slide_demo),
    ("Conclusion",         slide_conclusion),
]


def get_key() -> str:
    if sys.platform == "win32":
        ch = msvcrt.getch()
        if ch in (b'\xe0', b'\x00'):      # special key prefix on Windows
            ch2 = msvcrt.getch()
            if ch2 == b'M': return "right"
            if ch2 == b'K': return "left"
        if ch in (b' ', b'\r', b'\n'): return "next"
        if ch in (b'\x08',):            return "prev"
        if ch.lower() in (b'q',):       return "quit"
        return "other"
    else:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch in (' ', '\r', '\n'):        return "next"
        if ch in ('\x7f', '\x08'):         return "prev"
        if ch.lower() == 'q':             return "quit"
        if ch == '\x1b':
            ch2 = sys.stdin.read(2)
            if ch2 == '[C': return "right"
            if ch2 == '[D': return "left"
        return "other"


def render_slide(idx: int):
    os.system("cls" if sys.platform == "win32" else "clear")
    name, fn = SLIDES[idx]
    fn()
    console.print()
    console.print(Rule(style="dim"))
    nav = (
        f"[dim]  ← Prev   Space/→ Next   Q Quit   "
        f"[bold white]{idx+1}/{len(SLIDES)}[/bold white]  —  {name}[/dim]"
    )
    console.print(nav)


def main():
    idx = 0
    render_slide(idx)
    while True:
        key = get_key()
        if key == "quit":
            os.system("cls" if sys.platform == "win32" else "clear")
            console.print("\n[bold green]Presentation ended. Good luck! 🚀[/bold green]\n")
            break
        elif key in ("next", "right") and idx < len(SLIDES) - 1:
            idx += 1
            render_slide(idx)
        elif key in ("prev", "left") and idx > 0:
            idx -= 1
            render_slide(idx)


if __name__ == "__main__":
    main()
