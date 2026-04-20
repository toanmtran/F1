"""
================================================================================
PROBLEM 2: How Fast Is the Fastest? Estimating True Driver Pace
================================================================================
Course Topic: Confidence Intervals & Parameter Estimation

Research Question:
    Can we estimate a driver's "true pace" at a circuit with a confidence
    interval? How does sample size (number of laps) affect the precision
    of this estimate?

Methodology:
    1. Compute parametric (t-based) CIs for mean lap time per driver
    2. Compare CIs across top finishers via a forest plot
    3. Demonstrate effect of sample size on CI width (CLT in action)
    4. Compare parametric CIs vs. bootstrap CIs
    5. MLE vs. Method of Moments parameter estimation
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs" / "problem2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette — premium F1-inspired (shared across all problems)
COLORS = {
    'bg_dark':      '#0F1117',
    'bg_card':      '#1A1D29',
    'bg_card_alt':  '#222639',
    'accent_red':   '#E10600',
    'accent_blue':  '#3671C6',
    'accent_cyan':  '#00D2BE',
    'accent_gold':  '#FFD700',
    'accent_orange':'#FF8700',
    'accent_purple':'#9B59B6',
    'accent_green': '#2ECC71',
    'accent_pink':  '#E91E63',
    'text_primary': '#EAEAEA',
    'text_muted':   '#8892A0',
    'grid_line':    '#2A2E3A',
}

# Driver-specific colors (for the 2023 Italian GP top finishers)
DRIVER_COLORS = [
    '#E10600',   # Red
    '#3671C6',   # Blue
    '#00D2BE',   # Cyan/teal
    '#FF8700',   # Orange
    '#9B59B6',   # Purple
    '#FFD700',   # Gold
    '#2ECC71',   # Green
    '#E91E63',   # Pink
    '#1ABC9C',   # Turquoise
    '#F39C12',   # Amber
]

plt.rcParams.update({
    'figure.facecolor':   COLORS['bg_dark'],
    'axes.facecolor':     COLORS['bg_card'],
    'axes.edgecolor':     COLORS['grid_line'],
    'axes.labelcolor':    COLORS['text_primary'],
    'axes.grid':          True,
    'grid.color':         COLORS['grid_line'],
    'grid.alpha':         0.3,
    'text.color':         COLORS['text_primary'],
    'xtick.color':        COLORS['text_muted'],
    'ytick.color':        COLORS['text_muted'],
    'legend.facecolor':   COLORS['bg_card_alt'],
    'legend.edgecolor':   COLORS['grid_line'],
    'font.family':        'sans-serif',
    'font.size':          11,
    'axes.titlesize':     14,
    'axes.titleweight':   'bold',
    'figure.titlesize':   18,
    'figure.titleweight': 'bold',
    'savefig.dpi':        200,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  COLORS['bg_dark'],
})


def add_watermark(fig, text="F1 Statistical Analysis"):
    """Add a subtle watermark to figures."""
    fig.text(0.99, 0.01, text, ha='right', va='bottom',
             fontsize=8, color=COLORS['text_muted'], alpha=0.4, style='italic')


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load and merge lap times with driver and race info."""
    print("=" * 70)
    print("  PROBLEM 2: How Fast Is the Fastest?")
    print("  Confidence Intervals & Parameter Estimation")
    print("=" * 70)
    print("\n[*] Loading data...")

    lap_times = pd.read_csv(DATA_DIR / "lap_times.csv", na_values="\\N")
    races     = pd.read_csv(DATA_DIR / "races.csv", na_values="\\N")
    circuits  = pd.read_csv(DATA_DIR / "circuits.csv", na_values="\\N")
    results   = pd.read_csv(DATA_DIR / "results.csv", na_values="\\N")
    drivers   = pd.read_csv(DATA_DIR / "drivers.csv", na_values="\\N")

    print(f"   [OK] Loaded {len(lap_times):,} lap time records")

    return lap_times, races, circuits, results, drivers


def find_race_id(races, circuits, circuit_ref, year):
    """Find a raceId given a circuit reference name and year."""
    circuit_row = circuits[circuits['circuitRef'] == circuit_ref]
    if circuit_row.empty:
        return None
    circuit_id = circuit_row.iloc[0]['circuitId']
    race_row = races[(races['circuitId'] == circuit_id) & (races['year'] == year)]
    if race_row.empty:
        return None
    return race_row.iloc[0]['raceId']


def clean_driver_laps(lap_times, race_id, driver_id):
    """
    Clean lap times for one driver in one race.
    Removes lap 1 and outliers (IQR method).
    """
    mask = (lap_times['raceId'] == race_id) & (lap_times['driverId'] == driver_id)
    driver_laps = lap_times[mask].copy()

    # Remove lap 1
    driver_laps = driver_laps[driver_laps['lap'] > 1]

    if len(driver_laps) == 0:
        return driver_laps, 0, 0

    driver_laps['time_seconds'] = driver_laps['milliseconds'] / 1000.0

    # IQR outlier removal (pit in/out laps, safety car)
    Q1 = driver_laps['time_seconds'].quantile(0.25)
    Q3 = driver_laps['time_seconds'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 2.0 * IQR
    upper = Q3 + 2.0 * IQR

    n_before = len(driver_laps)
    driver_laps = driver_laps[
        (driver_laps['time_seconds'] >= lower) &
        (driver_laps['time_seconds'] <= upper)
    ]
    n_after = len(driver_laps)

    return driver_laps, n_before, n_after


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_t_ci(data, confidence=0.95):
    """Compute parametric CI using t-distribution."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=se)
    return mean, ci[0], ci[1], se


def compute_bootstrap_ci(data, n_boot=10000, confidence=0.95, seed=42):
    """Compute non-parametric bootstrap CI."""
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))
    return np.mean(data), ci_low, ci_high, boot_means


def mle_lognormal(data):
    """Fit log-normal via MLE (scipy default)."""
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    mu_mle = np.log(scale)
    sigma_mle = shape
    return mu_mle, sigma_mle


def mom_lognormal(data):
    """Fit log-normal via Method of Moments."""
    m = np.mean(data)
    v = np.var(data)
    sigma2_mom = np.log(1 + v / m**2)
    mu_mom = np.log(m) - sigma2_mom / 2
    sigma_mom = np.sqrt(sigma2_mom)
    return mu_mom, sigma_mom


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_forest_plot(driver_data, race_label, filename):
    """
    FIGURE 1: Forest plot comparing CIs across top finishers.
    Shows point estimate and 95% CI for each driver's mean lap time.
    """
    n_drivers = len(driver_data)
    fig, ax = plt.subplots(figsize=(14, max(6, n_drivers * 0.7 + 2)))

    fig.suptitle(f"95% Confidence Intervals for Mean Lap Time",
                 fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94,
             f"{race_label} -- Top {n_drivers} Finishers  |  "
             f"Overlapping CIs = no statistically significant pace difference",
             ha='center', fontsize=11, color=COLORS['accent_cyan'])

    y_positions = range(n_drivers)

    for i, (name, mean, ci_low, ci_high, n_laps, color) in enumerate(driver_data):
        ci_width = ci_high - ci_low

        # Error bar
        ax.errorbar(mean, i, xerr=[[mean - ci_low], [ci_high - mean]],
                    fmt='o', markersize=10, color=color,
                    ecolor=color, elinewidth=2.5, capsize=8, capthick=2,
                    zorder=3)

        # Label
        ax.text(ci_high + 0.03, i, f"  {mean:.3f}s  [{ci_low:.3f}, {ci_high:.3f}]",
                va='center', fontsize=10, color=COLORS['text_primary'])
        ax.text(ci_low - 0.03, i, f"n={n_laps}  ",
                va='center', ha='right', fontsize=9, color=COLORS['text_muted'])

    ax.set_yticks(y_positions)
    ax.set_yticklabels([d[0] for d in driver_data], fontsize=12, fontweight='bold')
    ax.set_xlabel("Mean Lap Time (seconds)", fontsize=13)
    ax.invert_yaxis()

    # Add vertical line at overall mean
    all_means = [d[1] for d in driver_data]
    overall_mean = np.mean(all_means)
    ax.axvline(x=overall_mean, color=COLORS['accent_gold'], linestyle='--',
               linewidth=1.5, alpha=0.6, label=f'Overall mean = {overall_mean:.3f}s')
    ax.legend(fontsize=10, loc='lower right')

    ax.set_title("", pad=0)  # clear default title
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


def plot_sample_size_effect(driver_laps, driver_name, race_label, filename):
    """
    FIGURE 2: How CI width shrinks with more laps (CLT demonstration).
    Shows CI width vs. sample size, with the theoretical 1/sqrt(n) curve.
    """
    times = driver_laps['time_seconds'].values
    total_laps = len(times)

    sample_sizes = list(range(5, total_laps + 1, 1))
    ci_widths = []
    ci_lows = []
    ci_highs = []
    means = []

    for n in sample_sizes:
        subset = times[:n]
        mean, low, high, _ = compute_t_ci(subset, confidence=0.95)
        ci_widths.append(high - low)
        ci_lows.append(low)
        ci_highs.append(high)
        means.append(mean)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Effect of Sample Size on Confidence Interval Precision",
                 fontsize=20, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             f"{driver_name} -- {race_label}  |  Demonstrating the Central Limit Theorem",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28,
                           left=0.08, right=0.95, top=0.9, bottom=0.07)

    # ── Panel 1: CI Width vs Sample Size ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(sample_sizes, ci_widths, color=COLORS['accent_cyan'],
             linewidth=2.5, label='Observed CI width', zorder=3)

    # Theoretical curve: width proportional to 1/sqrt(n)
    s = np.std(times)
    theoretical = [2 * stats.t.ppf(0.975, df=n-1) * s / np.sqrt(n)
                   for n in sample_sizes]
    ax1.plot(sample_sizes, theoretical, '--', color=COLORS['accent_orange'],
             linewidth=2, label=r'Theoretical: $\propto 1/\sqrt{n}$', alpha=0.8, zorder=2)

    ax1.set_xlabel("Number of Laps (sample size)", fontsize=12)
    ax1.set_ylabel("95% CI Width (seconds)", fontsize=12)
    ax1.set_title("CI Width Shrinks with More Data", fontsize=14, pad=10)
    ax1.legend(fontsize=10, framealpha=0.9)

    # Annotate key points
    for n_mark in [10, 20, 40]:
        if n_mark <= total_laps:
            idx = n_mark - 5  # offset for our range starting at 5
            if 0 <= idx < len(ci_widths):
                ax1.annotate(f"n={n_mark}\nwidth={ci_widths[idx]:.3f}s",
                             xy=(n_mark, ci_widths[idx]),
                             xytext=(n_mark + 3, ci_widths[idx] + 0.1),
                             fontsize=9, color=COLORS['accent_gold'],
                             arrowprops=dict(arrowstyle='->', color=COLORS['accent_gold'],
                                             lw=1.2))

    # ── Panel 2: Evolving CI band ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.fill_between(sample_sizes, ci_lows, ci_highs,
                     alpha=0.25, color=COLORS['accent_blue'], label='95% CI band')
    ax2.plot(sample_sizes, means, color=COLORS['accent_cyan'],
             linewidth=2, label='Running mean', zorder=3)

    # Final estimate
    final_mean = means[-1]
    ax2.axhline(y=final_mean, color=COLORS['accent_gold'], linestyle='--',
                linewidth=1.5, alpha=0.5, label=f'Final estimate: {final_mean:.3f}s')

    ax2.set_xlabel("Number of Laps Used", fontsize=12)
    ax2.set_ylabel("Mean Lap Time (seconds)", fontsize=12)
    ax2.set_title("Confidence Interval Converges with More Laps", fontsize=14, pad=10)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)

    # ── Panel 3: Histogram of all laps with CI ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.hist(times, bins=30, density=True, alpha=0.4, color=COLORS['accent_blue'],
             edgecolor=COLORS['bg_dark'], linewidth=0.5, zorder=2)

    # Overlay normal curve from mean and std
    x_range = np.linspace(times.min() - 0.5, times.max() + 0.5, 300)
    pdf = stats.norm.pdf(x_range, loc=np.mean(times), scale=np.std(times))
    ax3.plot(x_range, pdf, color=COLORS['accent_red'], linewidth=2, label='Normal fit', zorder=3)

    # Show CI on the x-axis
    _, ci_low_full, ci_high_full, _ = compute_t_ci(times)
    ax3.axvspan(ci_low_full, ci_high_full, alpha=0.15, color=COLORS['accent_gold'],
                label=f'95% CI: [{ci_low_full:.3f}, {ci_high_full:.3f}]', zorder=1)
    ax3.axvline(x=np.mean(times), color=COLORS['accent_gold'], linewidth=2,
                linestyle='--', alpha=0.8, zorder=4)

    ax3.set_xlabel("Lap Time (seconds)", fontsize=12)
    ax3.set_ylabel("Probability Density", fontsize=12)
    ax3.set_title(f"Distribution of {driver_name}'s Lap Times (n={total_laps})", fontsize=14, pad=10)
    ax3.legend(fontsize=9, framealpha=0.9)

    # ── Panel 4: Formula illustration ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_axis_off()

    ax4.set_title("The Mathematics Behind Confidence Intervals", fontsize=14,
                  fontweight='bold', pad=15)

    formulas = [
        ("Point Estimate", f"x-bar = {np.mean(times):.3f} seconds"),
        ("Standard Error", f"SE = s / sqrt(n) = {np.std(times, ddof=1):.3f} / sqrt({total_laps}) = {stats.sem(times):.4f}"),
        ("95% CI Formula", "x-bar +/- t_(0.025, n-1) * SE"),
        ("Result", f"[{ci_low_full:.3f}, {ci_high_full:.3f}] seconds"),
        ("CI Width", f"{ci_high_full - ci_low_full:.4f}s  (proportional to 1/sqrt(n))"),
        ("Interpretation", f"We are 95% confident the true mean\nlap time of {driver_name} lies in this interval"),
    ]

    for i, (label, value) in enumerate(formulas):
        y = 0.92 - i * 0.155
        ax4.text(0.03, y, label + ":", fontsize=12, fontweight='bold',
                 color=COLORS['accent_cyan'], transform=ax4.transAxes, va='top')
        ax4.text(0.03, y - 0.06, value, fontsize=10,
                 color=COLORS['text_primary'], transform=ax4.transAxes, va='top',
                 linespacing=1.5, family='monospace')

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


def plot_bootstrap_vs_parametric(driver_laps, driver_name, race_label, filename):
    """
    FIGURE 3: Compare parametric (t-based) CI vs. bootstrap CI.
    Also shows the bootstrap sampling distribution of the mean.
    """
    times = driver_laps['time_seconds'].values

    # Parametric CI
    mean_t, ci_low_t, ci_high_t, se_t = compute_t_ci(times)

    # Bootstrap CI
    mean_b, ci_low_b, ci_high_b, boot_means = compute_bootstrap_ci(times, n_boot=10000)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Parametric vs. Bootstrap Confidence Intervals",
                 fontsize=20, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             f"{driver_name} -- {race_label}  |  "
             f"Do we need to assume normality?",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28,
                           left=0.08, right=0.95, top=0.9, bottom=0.07)

    # ── Panel 1: Bootstrap distribution of means ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    ax1.hist(boot_means, bins=80, density=True, alpha=0.5,
             color=COLORS['accent_blue'], edgecolor=COLORS['bg_dark'],
             linewidth=0.3, label='Bootstrap distribution', zorder=2)

    # Overlay normal curve
    x_range = np.linspace(boot_means.min(), boot_means.max(), 300)
    pdf = stats.norm.pdf(x_range, loc=np.mean(boot_means), scale=np.std(boot_means))
    ax1.plot(x_range, pdf, color=COLORS['accent_red'], linewidth=2,
             label='Normal approximation', zorder=3)

    # Mark CIs
    ax1.axvspan(ci_low_t, ci_high_t, alpha=0.15, color=COLORS['accent_red'],
                label=f't-based CI: [{ci_low_t:.4f}, {ci_high_t:.4f}]', zorder=1)
    ax1.axvspan(ci_low_b, ci_high_b, alpha=0.15, color=COLORS['accent_cyan'],
                label=f'Bootstrap CI: [{ci_low_b:.4f}, {ci_high_b:.4f}]', zorder=1)
    ax1.axvline(x=mean_t, color=COLORS['accent_gold'], linewidth=2.5,
                linestyle='--', alpha=0.8, label=f'Sample mean: {mean_t:.4f}s', zorder=4)

    ax1.set_xlabel("Bootstrap Sample Mean (seconds)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Bootstrap Sampling Distribution of the Mean (B = 10,000)",
                  fontsize=14, pad=10)
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)

    # ── Panel 2: Side-by-side CI comparison ───────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    methods = ['t-distribution\n(Parametric)', 'Bootstrap\n(Non-parametric)']
    means_list = [mean_t, mean_b]
    ci_lows = [ci_low_t, ci_low_b]
    ci_highs = [ci_high_t, ci_high_b]
    colors_list = [COLORS['accent_red'], COLORS['accent_cyan']]

    for i, (method, m, cl, ch, c) in enumerate(
            zip(methods, means_list, ci_lows, ci_highs, colors_list)):
        ax2.errorbar(m, i, xerr=[[m - cl], [ch - m]],
                     fmt='s', markersize=12, color=c,
                     ecolor=c, elinewidth=3, capsize=10, capthick=2.5,
                     zorder=3)
        ax2.text(ch + 0.01, i,
                 f"  Width: {(ch-cl):.4f}s",
                 va='center', fontsize=11, color=COLORS['text_primary'])

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(methods, fontsize=12, fontweight='bold')
    ax2.set_xlabel("Mean Lap Time (seconds)", fontsize=12)
    ax2.set_title("CI Comparison: Parametric vs. Bootstrap", fontsize=14, pad=10)
    ax2.invert_yaxis()

    # ── Panel 3: Discussion box ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_axis_off()

    ax3.set_title("When Do They Differ?", fontsize=14, fontweight='bold', pad=15)

    diff = abs((ci_high_t - ci_low_t) - (ci_high_b - ci_low_b))
    pct_diff = diff / (ci_high_t - ci_low_t) * 100

    skewness = stats.skew(times)

    discussion_items = [
        ("t-based CI width:", f"{ci_high_t - ci_low_t:.5f} seconds"),
        ("Bootstrap CI width:", f"{ci_high_b - ci_low_b:.5f} seconds"),
        ("Difference:", f"{diff:.5f}s ({pct_diff:.1f}%)"),
        ("Data skewness:", f"{skewness:.4f}"),
        ("", ""),
        ("Key insight:", ""),
        ("", "The t-based CI assumes normality."),
        ("", "Bootstrap makes NO distributional"),
        ("", "assumptions."),
        ("", ""),
        ("", "From Problem 1, we know lap times"),
        ("", "are right-skewed. When skewness is"),
        ("", "mild, both methods agree closely."),
        ("", "When skewness is high (e.g. Monaco),"),
        ("", "bootstrap CIs are more reliable."),
    ]

    for i, (label, value) in enumerate(discussion_items):
        y = 0.95 - i * 0.063
        if label:
            ax3.text(0.05, y, label, fontsize=10, fontweight='bold',
                     color=COLORS['accent_cyan'], transform=ax3.transAxes, va='top')
            ax3.text(0.55, y, value, fontsize=10,
                     color=COLORS['text_primary'], transform=ax3.transAxes, va='top')
        else:
            ax3.text(0.05, y, value, fontsize=10,
                     color=COLORS['text_primary'], transform=ax3.transAxes, va='top')

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


def plot_mle_vs_mom(driver_laps, driver_name, race_label, filename):
    """
    FIGURE 4: Compare MLE vs Method of Moments parameter estimation
    for the Log-Normal distribution.
    """
    times = driver_laps['time_seconds'].values

    # Fit using both methods
    mu_mle, sigma_mle = mle_lognormal(times)
    mu_mom, sigma_mom = mom_lognormal(times)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Parameter Estimation: MLE vs. Method of Moments",
                 fontsize=20, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             f"Fitting a Log-Normal distribution to {driver_name}'s lap times -- {race_label}",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28,
                           left=0.08, right=0.95, top=0.9, bottom=0.07)

    # ── Panel 1: PDF overlay ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    ax1.hist(times, bins=40, density=True, alpha=0.35, color=COLORS['accent_blue'],
             edgecolor=COLORS['bg_dark'], linewidth=0.5, label='Observed', zorder=2)

    x_range = np.linspace(times.min() - 0.5, times.max() + 0.5, 500)

    # MLE fit
    pdf_mle = stats.lognorm.pdf(x_range, sigma_mle, loc=0, scale=np.exp(mu_mle))
    ax1.plot(x_range, pdf_mle, color=COLORS['accent_red'], linewidth=2.5,
             label=f'MLE: mu={mu_mle:.4f}, sigma={sigma_mle:.4f}', zorder=3)

    # MoM fit
    pdf_mom = stats.lognorm.pdf(x_range, sigma_mom, loc=0, scale=np.exp(mu_mom))
    ax1.plot(x_range, pdf_mom, color=COLORS['accent_cyan'], linewidth=2.5,
             linestyle='--',
             label=f'MoM: mu={mu_mom:.4f}, sigma={sigma_mom:.4f}', zorder=3)

    ax1.set_xlabel("Lap Time (seconds)", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title("Log-Normal Fit: MLE vs. Method of Moments", fontsize=14, pad=10)
    ax1.legend(fontsize=10, framealpha=0.9)

    # ── Panel 2: Parameter comparison bar chart ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    params = ['mu (location)', 'sigma (shape)']
    mle_vals = [mu_mle, sigma_mle]
    mom_vals = [mu_mom, sigma_mom]

    x_pos = np.arange(len(params))
    bar_width = 0.3

    bars1 = ax2.bar(x_pos - bar_width/2, mle_vals, bar_width,
                    label='MLE', color=COLORS['accent_red'], alpha=0.85, zorder=3)
    bars2 = ax2.bar(x_pos + bar_width/2, mom_vals, bar_width,
                    label='MoM', color=COLORS['accent_cyan'], alpha=0.85, zorder=3)

    # Add value labels
    for bar, val in zip(bars1, mle_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.5f}', ha='center', va='bottom', fontsize=10,
                 color=COLORS['text_primary'])
    for bar, val in zip(bars2, mom_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.5f}', ha='center', va='bottom', fontsize=10,
                 color=COLORS['text_primary'])

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(params, fontsize=12)
    ax2.set_ylabel("Parameter Value", fontsize=12)
    ax2.set_title("Estimated Parameters", fontsize=14, pad=10)
    ax2.legend(fontsize=10, framealpha=0.9)

    # ── Panel 3: Discussion ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_axis_off()

    ax3.set_title("MLE vs. Method of Moments", fontsize=14, fontweight='bold', pad=15)

    # Compute log-likelihoods for comparison
    ll_mle = np.sum(stats.lognorm.logpdf(times, sigma_mle, loc=0, scale=np.exp(mu_mle)))
    ll_mom = np.sum(stats.lognorm.logpdf(times, sigma_mom, loc=0, scale=np.exp(mu_mom)))

    discussion = [
        ("MLE", "Maximizes the likelihood function"),
        ("", "-> Asymptotically efficient estimator"),
        ("", "-> Optimal for large samples"),
        ("", f"   Log-likelihood: {ll_mle:.2f}"),
        ("", ""),
        ("MoM", "Matches sample moments to"),
        ("", "population moment formulas"),
        ("", "-> Simpler, closed-form solution"),
        ("", "-> Less efficient than MLE"),
        ("", f"   Log-likelihood: {ll_mom:.2f}"),
        ("", ""),
        ("Result", f"MLE achieves higher log-likelihood"),
        ("", f"(Delta = {ll_mle - ll_mom:.4f}), confirming"),
        ("", "MLE's theoretical superiority."),
        ("", "Both methods converge as n -> infinity."),
    ]

    for i, (label, value) in enumerate(discussion):
        y = 0.95 - i * 0.063
        if label:
            ax3.text(0.05, y, label + ":", fontsize=11, fontweight='bold',
                     color=COLORS['accent_cyan'], transform=ax3.transAxes, va='top')
        ax3.text(0.05 if not label else 0.22, y,
                 value if not label else "", fontsize=10,
                 color=COLORS['text_primary'], transform=ax3.transAxes, va='top')
        if label and label != "":
            # Print the text after the label on the same line
            pass

    # Rewrite the discussion panel more cleanly
    for txt in ax3.texts[:]:  # Clear and redo
        txt.remove()
    ax3.set_title("MLE vs. Method of Moments", fontsize=14, fontweight='bold', pad=15)

    lines = [
        (True,  "Maximum Likelihood (MLE)"),
        (False, "  Maximizes the likelihood function"),
        (False, f"  Asymptotically efficient estimator"),
        (False, f"  Log-likelihood = {ll_mle:.2f}"),
        (False, ""),
        (True,  "Method of Moments (MoM)"),
        (False, "  Matches sample moments to theoretical"),
        (False, "  Simpler closed-form, less efficient"),
        (False, f"  Log-likelihood = {ll_mom:.2f}"),
        (False, ""),
        (True,  "Conclusion"),
        (False, f"  MLE achieves higher log-likelihood"),
        (False, f"  (Delta = {ll_mle - ll_mom:.4f})"),
        (False, "  Both converge as n -> infinity"),
        (False, "  MLE is preferred for inference"),
    ]

    for i, (is_header, text) in enumerate(lines):
        y = 0.95 - i * 0.063
        if is_header:
            ax3.text(0.05, y, text, fontsize=11, fontweight='bold',
                     color=COLORS['accent_cyan'], transform=ax3.transAxes, va='top')
        else:
            ax3.text(0.05, y, text, fontsize=10,
                     color=COLORS['text_primary'], transform=ax3.transAxes, va='top')

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


def plot_multi_driver_bootstrap(all_driver_data, race_label, filename):
    """
    FIGURE 5: Bootstrap CI comparison across multiple drivers.
    Shows both parametric and bootstrap CIs side by side.
    """
    n_drivers = len(all_driver_data)
    fig, ax = plt.subplots(figsize=(14, max(7, n_drivers * 1.2 + 2)))

    fig.suptitle("Parametric vs. Bootstrap CIs -- All Drivers Compared",
                 fontsize=18, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94,
             f"{race_label}  |  Triangles = t-based, Circles = Bootstrap",
             ha='center', fontsize=11, color=COLORS['accent_cyan'])

    offset = 0.15  # vertical offset between parametric and bootstrap

    for i, (name, times, color) in enumerate(all_driver_data):
        # Parametric CI
        mean_t, ci_low_t, ci_high_t, _ = compute_t_ci(times)
        # Bootstrap CI
        mean_b, ci_low_b, ci_high_b, _ = compute_bootstrap_ci(times, n_boot=5000)

        # Parametric (triangle, slightly above)
        ax.errorbar(mean_t, i - offset, xerr=[[mean_t - ci_low_t], [ci_high_t - mean_t]],
                    fmt='^', markersize=8, color=color,
                    ecolor=color, elinewidth=2, capsize=6, capthick=1.5,
                    alpha=0.85, zorder=3)

        # Bootstrap (circle, slightly below)
        ax.errorbar(mean_b, i + offset, xerr=[[mean_b - ci_low_b], [ci_high_b - mean_b]],
                    fmt='o', markersize=8, color=color,
                    ecolor=color, elinewidth=2, capsize=6, capthick=1.5,
                    alpha=0.85, zorder=3, linestyle='--')

    ax.set_yticks(range(n_drivers))
    ax.set_yticklabels([d[0] for d in all_driver_data], fontsize=11, fontweight='bold')
    ax.set_xlabel("Mean Lap Time (seconds)", fontsize=13)
    ax.invert_yaxis()

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['text_muted'],
               markersize=10, label='Parametric (t-based)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['text_muted'],
               markersize=10, label='Bootstrap (B=5000)'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower right', framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


def plot_confidence_level_comparison(driver_laps, driver_name, race_label, filename):
    """
    FIGURE 6: How CI width changes with different confidence levels.
    Shows 90%, 95%, and 99% CIs side by side.
    """
    times = driver_laps['time_seconds'].values

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Effect of Confidence Level on Interval Width -- {driver_name}",
                 fontsize=18, fontweight='bold', y=1.02)
    fig.text(0.5, 0.96,
             "Higher confidence = wider interval = less precise but more reliable",
             ha='center', fontsize=12, color=COLORS['text_muted'], style='italic')

    conf_levels = [0.90, 0.95, 0.99]
    conf_colors = [COLORS['accent_green'], COLORS['accent_cyan'], COLORS['accent_red']]

    for ax, conf, clr in zip(axes, conf_levels, conf_colors):
        mean, ci_low, ci_high, se = compute_t_ci(times, confidence=conf)
        ci_width = ci_high - ci_low

        # Histogram
        ax.hist(times, bins=30, density=True, alpha=0.3, color=COLORS['accent_blue'],
                edgecolor=COLORS['bg_dark'], linewidth=0.5, zorder=2)

        # Normal fit
        x_range = np.linspace(times.min() - 0.5, times.max() + 0.5, 300)
        pdf = stats.norm.pdf(x_range, loc=np.mean(times), scale=np.std(times))
        ax.plot(x_range, pdf, color=COLORS['text_muted'], linewidth=1.5,
                alpha=0.7, zorder=3)

        # CI band
        ax.axvspan(ci_low, ci_high, alpha=0.2, color=clr, zorder=1)
        ax.axvline(x=mean, color=clr, linewidth=2.5, linestyle='--', alpha=0.9, zorder=4)
        ax.axvline(x=ci_low, color=clr, linewidth=1.5, linestyle=':', alpha=0.7, zorder=4)
        ax.axvline(x=ci_high, color=clr, linewidth=1.5, linestyle=':', alpha=0.7, zorder=4)

        # Stats box
        stats_text = (f"Confidence: {conf*100:.0f}%\n"
                      f"Mean: {mean:.3f}s\n"
                      f"CI: [{ci_low:.3f}, {ci_high:.3f}]\n"
                      f"Width: {ci_width:.4f}s")
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_card_alt'],
                          edgecolor=clr, alpha=0.95, linewidth=2))

        ax.set_title(f"{conf*100:.0f}% Confidence Interval",
                     fontsize=14, fontweight='bold', color=clr, pad=10)
        ax.set_xlabel("Lap Time (seconds)", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Probability Density", fontsize=11)

    plt.tight_layout()
    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


def print_detailed_results(driver_name, times, ci_results_t, ci_results_b):
    """Print formatted CI results to console."""
    mean_t, ci_low_t, ci_high_t, se_t = ci_results_t
    mean_b, ci_low_b, ci_high_b, _ = ci_results_b

    print(f"\n{'─' * 60}")
    print(f"  [RESULTS] {driver_name}")
    print(f"{'─' * 60}")
    print(f"  Sample size:      {len(times)} laps")
    print(f"  Mean:             {mean_t:.4f} seconds")
    print(f"  Std Dev:          {np.std(times, ddof=1):.4f} seconds")
    print(f"  Std Error:        {se_t:.5f} seconds")
    print(f"  Skewness:         {stats.skew(times):.4f}")
    print()
    print(f"  Parametric 95% CI:  [{ci_low_t:.4f}, {ci_high_t:.4f}]  width = {ci_high_t - ci_low_t:.5f}s")
    print(f"  Bootstrap  95% CI:  [{ci_low_b:.4f}, {ci_high_b:.4f}]  width = {ci_high_b - ci_low_b:.5f}s")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis():
    """Execute the full Problem 2 analysis."""

    lap_times, races, circuits, results, drivers = load_data()

    # Find the 2023 Italian GP (same primary race as Problem 1)
    primary_race_id = find_race_id(races, circuits, 'monza', 2023)
    if primary_race_id is None:
        for y in [2022, 2024, 2021, 2019]:
            primary_race_id = find_race_id(races, circuits, 'monza', y)
            if primary_race_id is not None:
                break

    race_info = races[races['raceId'] == primary_race_id].iloc[0]
    race_year = int(race_info['year'])
    race_name = race_info.get('name', 'Italian Grand Prix')
    race_label = f"{race_year} {race_name}"

    # Get top 10 finishers
    race_results = results[results['raceId'] == primary_race_id].copy()
    race_results['positionOrder'] = pd.to_numeric(race_results['positionOrder'], errors='coerce')
    race_results = race_results.sort_values('positionOrder').head(10)

    # Merge with driver names
    race_results = race_results.merge(
        drivers[['driverId', 'forename', 'surname']],
        on='driverId', how='left'
    )
    race_results['driver_name'] = race_results['forename'] + ' ' + race_results['surname']

    print(f"\n   Race: {race_label}")
    print(f"   Analyzing top {len(race_results)} finishers\n")

    # ═══════════════════════════════════════════════════════════════════════
    # PART A: Forest Plot - CIs for All Top Finishers
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  PART A: Confidence Intervals for Top Finishers")
    print("=" * 70)

    driver_ci_data = []
    all_driver_bootstrap = []

    for idx, row in race_results.iterrows():
        d_id = row['driverId']
        d_name = row['driver_name']

        clean, n_before, n_after = clean_driver_laps(lap_times, primary_race_id, d_id)
        if len(clean) < 10:
            print(f"   [WARN] Not enough clean laps for {d_name} ({len(clean)}), skipping...")
            continue

        times = clean['time_seconds'].values
        color = DRIVER_COLORS[len(driver_ci_data) % len(DRIVER_COLORS)]

        # Parametric CI
        ci_t = compute_t_ci(times)
        # Bootstrap CI
        ci_b = compute_bootstrap_ci(times)

        print_detailed_results(d_name, times, ci_t, ci_b)

        driver_ci_data.append((d_name, ci_t[0], ci_t[1], ci_t[2], len(times), color))
        all_driver_bootstrap.append((d_name, times, color))

    print("\n   [*] Generating visualizations...")

    # Figure 1: Forest plot
    plot_forest_plot(driver_ci_data, race_label, "fig1_forest_plot_CIs.png")

    # ═══════════════════════════════════════════════════════════════════════
    # PART B: Sample Size Effect (CLT Demonstration)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART B: Effect of Sample Size on CI Width (CLT)")
    print("=" * 70)

    # Use the race winner for the deep-dive
    winner_name = driver_ci_data[0][0]
    winner_laps = None
    for name, times, color in all_driver_bootstrap:
        if name == winner_name:
            winner_laps_df_mask = (
                (lap_times['raceId'] == primary_race_id) &
                (lap_times['driverId'] == race_results[
                    race_results['driver_name'] == winner_name].iloc[0]['driverId'])
            )
            winner_laps, _, _ = clean_driver_laps(
                lap_times, primary_race_id,
                race_results[race_results['driver_name'] == winner_name].iloc[0]['driverId']
            )
            break

    if winner_laps is not None and len(winner_laps) >= 10:
        plot_sample_size_effect(winner_laps, winner_name, race_label,
                                "fig2_sample_size_effect.png")

    # ═══════════════════════════════════════════════════════════════════════
    # PART C: Parametric vs. Bootstrap CI
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART C: Parametric vs. Bootstrap Confidence Intervals")
    print("=" * 70)

    if winner_laps is not None and len(winner_laps) >= 10:
        plot_bootstrap_vs_parametric(winner_laps, winner_name, race_label,
                                     "fig3_bootstrap_vs_parametric.png")

    # ═══════════════════════════════════════════════════════════════════════
    # PART D: MLE vs. Method of Moments
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART D: Parameter Estimation -- MLE vs. Method of Moments")
    print("=" * 70)

    if winner_laps is not None and len(winner_laps) >= 10:
        mu_mle, sigma_mle = mle_lognormal(winner_laps['time_seconds'].values)
        mu_mom, sigma_mom = mom_lognormal(winner_laps['time_seconds'].values)
        print(f"\n   Driver: {winner_name}")
        print(f"   MLE:  mu = {mu_mle:.6f}, sigma = {sigma_mle:.6f}")
        print(f"   MoM:  mu = {mu_mom:.6f}, sigma = {sigma_mom:.6f}")
        print(f"   Difference in mu: {abs(mu_mle - mu_mom):.6f}")
        print(f"   Difference in sigma: {abs(sigma_mle - sigma_mom):.6f}")

        plot_mle_vs_mom(winner_laps, winner_name, race_label,
                        "fig4_mle_vs_mom.png")

    # ═══════════════════════════════════════════════════════════════════════
    # PART E: Multi-Driver Bootstrap Comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART E: Multi-Driver Parametric vs. Bootstrap CI Comparison")
    print("=" * 70)

    plot_multi_driver_bootstrap(all_driver_bootstrap, race_label,
                                "fig5_multi_driver_bootstrap.png")

    # ═══════════════════════════════════════════════════════════════════════
    # PART F: Confidence Level Comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART F: Effect of Confidence Level (90% vs. 95% vs. 99%)")
    print("=" * 70)

    if winner_laps is not None and len(winner_laps) >= 10:
        plot_confidence_level_comparison(winner_laps, winner_name, race_label,
                                         "fig6_confidence_levels.png")

    # ═══════════════════════════════════════════════════════════════════════
    # CONCLUSIONS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  CONCLUSIONS")
    print("=" * 70)

    print("\n   Key Findings:")
    print("   1. Confidence intervals quantify UNCERTAINTY in pace estimates")
    print("      -> A single fastest lap is deceiving; CI shows the true range")
    print("   2. CI width shrinks proportionally to 1/sqrt(n) -- Central Limit Theorem")
    print("      -> More laps = more precise estimate (but diminishing returns)")
    print("   3. Parametric and bootstrap CIs agree closely when data is")
    print("      approximately symmetric (mild skewness)")
    print("   4. MLE is more efficient than Method of Moments, achieving higher")
    print("      log-likelihood with the same data")
    print("   5. Non-overlapping CIs between drivers indicate statistically")
    print("      significant pace differences (preview of Problem 3)")

    print(f"\n   All figures saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("  [DONE] Problem 2 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
