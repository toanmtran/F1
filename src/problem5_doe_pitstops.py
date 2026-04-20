"""
================================================================================
PROBLEM 5: What Makes a Perfect Pit Stop?
================================================================================
Course Topic: First-Order Orthogonal Design (Factorial / DOE)

Research Question:
    What factors influence pit stop duration? Can we use a designed
    experiment framework (2^k factorial) to quantify main effects
    and interactions?

Factors (2^4 Factorial Design):
    A: Team Tier        (Top 3 vs Rest)
    B: Stop Number      (1st vs 2nd+)
    C: Race Phase       (Early vs Late)
    D: Era              (Pre-2018 vs 2018+)

Response: Pit stop duration (milliseconds)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
from itertools import combinations
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs" / "problem5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Top teams (constructorIds for Red Bull, Mercedes, Ferrari, McLaren)
TOP_TEAM_NAMES = ['Red Bull', 'Mercedes', 'Ferrari']

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


def add_watermark(fig):
    fig.text(0.99, 0.01, "F1 Statistical Analysis", ha='right', va='bottom',
             fontsize=8, color=COLORS['text_muted'], alpha=0.4, style='italic')


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load pit stop data and encode factorial design variables."""
    print("=" * 70)
    print("  PROBLEM 5: What Makes a Perfect Pit Stop?")
    print("  First-Order Orthogonal Design (2^4 Factorial)")
    print("=" * 70)
    print("\n[*] Loading data...")

    pit_stops    = pd.read_csv(DATA_DIR / "pit_stops.csv", na_values="\\N")
    races        = pd.read_csv(DATA_DIR / "races.csv", na_values="\\N")
    results      = pd.read_csv(DATA_DIR / "results.csv", na_values="\\N")
    constructors = pd.read_csv(DATA_DIR / "constructors.csv", na_values="\\N")

    # Merge pit stops with race info
    df = pit_stops.merge(races[['raceId', 'year', 'name']], on='raceId', how='left')

    # Get constructor for each driver-race combo
    driver_constructor = results[['raceId', 'driverId', 'constructorId']].drop_duplicates()
    df = df.merge(driver_constructor, on=['raceId', 'driverId'], how='left')

    # Get constructor names
    df = df.merge(constructors[['constructorId', 'name']].rename(
        columns={'name': 'constructor_name'}), on='constructorId', how='left')

    # Convert duration
    df['milliseconds'] = pd.to_numeric(df['milliseconds'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    # Remove extreme outliers (pit lane drive-through penalties, red flags etc.)
    df = df[df['milliseconds'].between(1500, 60000)].copy()

    # Get total laps per race for phase calculation
    race_max_lap = df.groupby('raceId')['lap'].max().reset_index()
    race_max_lap.columns = ['raceId', 'max_lap']
    df = df.merge(race_max_lap, on='raceId', how='left')

    # Identify top teams
    top_team_ids = constructors[
        constructors['name'].str.contains('|'.join(TOP_TEAM_NAMES), case=False, na=False)
    ]['constructorId'].tolist()

    print(f"   Top teams: {constructors[constructors['constructorId'].isin(top_team_ids)]['name'].tolist()}")

    # ── DEFINE FACTORIAL DESIGN VARIABLES ─────────────────────────────────
    # Factor A: Team Tier (+1 = top team, -1 = rest)
    df['A'] = df['constructorId'].apply(lambda x: 1 if x in top_team_ids else -1)
    df['A_label'] = df['A'].map({1: 'Top Team', -1: 'Other Team'})

    # Factor B: Stop Number (+1 = 2nd or later, -1 = 1st stop)
    df['B'] = df['stop'].apply(lambda x: -1 if x == 1 else 1)
    df['B_label'] = df['B'].map({-1: '1st Stop', 1: '2nd+ Stop'})

    # Factor C: Race Phase (+1 = 2nd half, -1 = 1st half)
    df['C'] = df.apply(lambda r: -1 if r['lap'] <= r['max_lap'] / 2 else 1, axis=1)
    df['C_label'] = df['C'].map({-1: 'Early Race', 1: 'Late Race'})

    # Factor D: Era (+1 = 2018+, -1 = pre-2018)
    df['D'] = df['year'].apply(lambda y: 1 if y >= 2018 else -1)
    df['D_label'] = df['D'].map({-1: 'Pre-2018', 1: '2018+'})

    # Compute interaction terms
    df['AB'] = df['A'] * df['B']
    df['AC'] = df['A'] * df['C']
    df['AD'] = df['A'] * df['D']
    df['BC'] = df['B'] * df['C']
    df['BD'] = df['B'] * df['D']
    df['CD'] = df['C'] * df['D']
    df['ABC'] = df['A'] * df['B'] * df['C']
    df['ABD'] = df['A'] * df['B'] * df['D']
    df['ACD'] = df['A'] * df['C'] * df['D']
    df['BCD'] = df['B'] * df['C'] * df['D']
    df['ABCD'] = df['A'] * df['B'] * df['C'] * df['D']

    print(f"   [OK] {len(df):,} pit stops loaded ({df['year'].min()}-{df['year'].max()})")
    print(f"   Factor A (Team): Top={len(df[df['A']==1]):,}, Other={len(df[df['A']==-1]):,}")
    print(f"   Factor B (Stop): 1st={len(df[df['B']==-1]):,}, 2nd+={len(df[df['B']==1]):,}")
    print(f"   Factor C (Phase): Early={len(df[df['C']==-1]):,}, Late={len(df[df['C']==1]):,}")
    print(f"   Factor D (Era): Pre-2018={len(df[df['D']==-1]):,}, 2018+={len(df[df['D']==1]):,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Design Matrix & Main Effects
# ─────────────────────────────────────────────────────────────────────────────

def plot_design_and_effects(df, filename):
    """
    FIGURE 1: Show the factorial design matrix and main effects.
    """
    print("\n" + "=" * 70)
    print("  PART A: Design Matrix & Main Effects")
    print("=" * 70)

    # Compute main effects
    factors = {'A': 'Team Tier', 'B': 'Stop Number', 'C': 'Race Phase', 'D': 'Era'}
    main_effects = {}
    for f, label in factors.items():
        high = df[df[f] == 1]['milliseconds'].mean()
        low  = df[df[f] == -1]['milliseconds'].mean()
        effect = (high - low) / 2  # Standard DOE effect formula
        main_effects[f] = {'label': label, 'effect': effect, 'high': high, 'low': low}
        print(f"   Main Effect {f} ({label}): {effect:.1f} ms")
        print(f"     High (+1): {high:.1f} ms, Low (-1): {low:.1f} ms")

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("2^4 Factorial Design: Pit Stop Duration Analysis",
                 fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.945,
             "Factors: Team Tier (A), Stop Number (B), Race Phase (C), Era (D)  |  "
             "Response: Duration (ms)",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.97, top=0.9, bottom=0.06)

    # Panel 1: Design matrix table
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_axis_off()
    ax1.set_title("2^4 Design Matrix (16 Runs)", fontsize=13, fontweight='bold', pad=15)

    # Show the standard 2^4 design
    design_rows = []
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                for d in [-1, 1]:
                    mask = (df['A'] == a) & (df['B'] == b) & (df['C'] == c) & (df['D'] == d)
                    n = mask.sum()
                    mean_y = df.loc[mask, 'milliseconds'].mean() if n > 0 else 0
                    design_rows.append((a, b, c, d, n, mean_y))

    headers = ['A', 'B', 'C', 'D', 'n', 'Y (ms)']
    col_w = [0.12, 0.12, 0.12, 0.12, 0.18, 0.22]
    row_h = 0.045
    start_y = 0.94

    # Header
    for j, (h, w) in enumerate(zip(headers, col_w)):
        x = 0.02 + sum(col_w[:j])
        ax1.text(x + w/2, start_y, h, ha='center', va='center', fontsize=9,
                 fontweight='bold', color=COLORS['accent_cyan'], transform=ax1.transAxes)

    # Data rows (show all 16 rows)
    for i, row in enumerate(design_rows):
        y = start_y - (i + 1) * row_h
        for j, (val, w) in enumerate(zip(row, col_w)):
            x = 0.02 + sum(col_w[:j])
            if j < 4:
                text = '+1' if val == 1 else '-1'
                color = COLORS['accent_green'] if val == 1 else COLORS['accent_red']
            elif j == 4:
                text = f'{val:,}'
                color = COLORS['text_primary']
            else:
                text = f'{val:.0f}'
                color = COLORS['accent_gold']
            ax1.text(x + w/2, y, text, ha='center', va='center', fontsize=8,
                     color=color, transform=ax1.transAxes)

    ax1.text(0.5, start_y - 17.5 * row_h, f"Total observations: {len(df):,}",
             ha='center', fontsize=9, color=COLORS['text_muted'],
             transform=ax1.transAxes)

    # Panel 2: Main effects bar chart
    ax2 = fig.add_subplot(gs[0, 2:])
    effect_labels = [v['label'] for v in main_effects.values()]
    effect_values = [v['effect'] for v in main_effects.values()]
    effect_colors = [COLORS['accent_red'] if v > 0 else COLORS['accent_green']
                     for v in effect_values]

    bars = ax2.barh(range(len(effect_labels)), effect_values,
                    color=effect_colors, alpha=0.8, height=0.5,
                    edgecolor=COLORS['bg_dark'], linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, effect_values)):
        sign = '+' if val > 0 else ''
        ax2.text(bar.get_width() + (50 if val > 0 else -50), bar.get_y() + bar.get_height()/2,
                 f'{sign}{val:.0f} ms', va='center', fontsize=12,
                 ha='left' if val > 0 else 'right',
                 color=COLORS['text_primary'], fontweight='bold')

    ax2.set_yticks(range(len(effect_labels)))
    ax2.set_yticklabels(effect_labels, fontsize=12)
    ax2.set_xlabel("Effect on Duration (ms)  |  + = slower, - = faster", fontsize=11)
    ax2.set_title("Main Effects (Half the difference between high and low levels)", pad=10)
    ax2.axvline(0, color=COLORS['text_muted'], linewidth=1)
    ax2.invert_yaxis()

    # Panel 3: Box plot per factor level
    factor_keys = ['A', 'B', 'C', 'D']
    factor_labels_map = {
        'A': ('Other\nTeam', 'Top\nTeam'),
        'B': ('1st\nStop', '2nd+\nStop'),
        'C': ('Early\nRace', 'Late\nRace'),
        'D': ('Pre-\n2018', '2018+'),
    }
    factor_colors_list = [COLORS['accent_red'], COLORS['accent_blue'],
                          COLORS['accent_purple'], COLORS['accent_orange']]

    for idx, (fk, fc) in enumerate(zip(factor_keys, factor_colors_list)):
        ax_pos = fig.add_subplot(gs[1, idx])

        low_data = df[df[fk] == -1]['milliseconds'].clip(upper=40000)
        high_data = df[df[fk] == 1]['milliseconds'].clip(upper=40000)

        bp = ax_pos.boxplot([low_data, high_data],
                           labels=list(factor_labels_map[fk]),
                           patch_artist=True, widths=0.4,
                           medianprops=dict(color=COLORS['accent_gold'], linewidth=2),
                           whiskerprops=dict(color=COLORS['text_muted']),
                           capprops=dict(color=COLORS['text_muted']),
                           flierprops=dict(marker='.', markerfacecolor=COLORS['text_muted'],
                                          markersize=2, alpha=0.2))
        bp['boxes'][0].set_facecolor(fc)
        bp['boxes'][0].set_alpha(0.4)
        bp['boxes'][1].set_facecolor(fc)
        bp['boxes'][1].set_alpha(0.7)

        ax_pos.set_ylabel("Duration (ms)" if idx == 0 else "", fontsize=10)
        ax_pos.set_title(f"Factor {fk}: {factors[fk]}", fontsize=12, color=fc, pad=5)

        # Add mean annotation
        ax_pos.text(1, low_data.median() + 500, f'{low_data.median():.0f}',
                   ha='center', fontsize=9, color=COLORS['accent_gold'])
        ax_pos.text(2, high_data.median() + 500, f'{high_data.median():.0f}',
                   ha='center', fontsize=9, color=COLORS['accent_gold'])

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"\n   [SAVED] {filename}")

    return main_effects


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Interaction Effects
# ─────────────────────────────────────────────────────────────────────────────

def plot_interactions(df, filename):
    """
    FIGURE 2: Two-way interaction plots.
    """
    print("\n" + "=" * 70)
    print("  PART B: Interaction Effects")
    print("=" * 70)

    interactions = [
        ('A', 'D', 'Team Tier x Era', 'Does technology improvement help all teams equally?'),
        ('A', 'B', 'Team Tier x Stop Number', 'Do top teams handle 2nd stops better?'),
        ('B', 'C', 'Stop Number x Race Phase', 'Are later stops in late race slower?'),
        ('C', 'D', 'Race Phase x Era', 'Has late-race stop speed improved more?'),
        ('A', 'C', 'Team Tier x Race Phase', 'Do top teams maintain speed throughout?'),
        ('B', 'D', 'Stop Number x Era', 'Have 2nd stops improved more over time?'),
    ]

    # Compute interaction effects
    for f1, f2, label, _ in interactions:
        ij = f1 + f2
        if ij in df.columns:
            high = df[df[ij] == 1]['milliseconds'].mean()
            low  = df[df[ij] == -1]['milliseconds'].mean()
            effect = (high - low) / 2
            print(f"   Interaction {ij} ({label}): {effect:.1f} ms")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Two-Way Interaction Effects",
                 fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.945,
             "Non-parallel lines indicate significant interactions  |  "
             "Crossing lines = strong interaction",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.97, top=0.9, bottom=0.06)

    factor_labels_display = {
        'A': {-1: 'Other Team', 1: 'Top Team'},
        'B': {-1: '1st Stop', 1: '2nd+ Stop'},
        'C': {-1: 'Early', 1: 'Late'},
        'D': {-1: 'Pre-2018', 1: '2018+'},
    }

    plot_colors = [COLORS['accent_red'], COLORS['accent_cyan']]

    for idx, (f1, f2, title, question) in enumerate(interactions):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])

        # Compute means for interaction plot
        for level_idx, f2_level in enumerate([-1, 1]):
            means = []
            for f1_level in [-1, 1]:
                mask = (df[f1] == f1_level) & (df[f2] == f2_level)
                means.append(df.loc[mask, 'milliseconds'].mean())

            f2_label = factor_labels_display[f2][f2_level]
            ax.plot([-1, 1], means, '-o', color=plot_colors[level_idx],
                    linewidth=2.5, markersize=8, label=f2_label, zorder=3)

            # Annotate values
            for xi, m in zip([-1, 1], means):
                ax.text(xi, m + 100, f'{m:.0f}', ha='center', fontsize=8,
                       color=plot_colors[level_idx], fontweight='bold')

        ax.set_xticks([-1, 1])
        ax.set_xticklabels([factor_labels_display[f1][-1],
                           factor_labels_display[f1][1]], fontsize=10)
        ax.set_ylabel("Duration (ms)", fontsize=10)
        ax.set_title(title, fontsize=12, pad=5)
        ax.legend(fontsize=9, framealpha=0.9, loc='best')

        # Add interpretation
        ax.text(0.5, 0.02, question, transform=ax.transAxes,
                ha='center', fontsize=8, color=COLORS['text_muted'], style='italic')

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"\n   [SAVED] {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: ANOVA Table & Pareto Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_anova_pareto(df, filename):
    """
    FIGURE 3: Full factorial ANOVA and Pareto chart of effects.
    """
    print("\n" + "=" * 70)
    print("  PART C: ANOVA & Pareto Chart of Effects")
    print("=" * 70)

    # Compute ALL effects (main + interactions)
    all_effects = {}
    effect_terms = ['A', 'B', 'C', 'D', 'AB', 'AC', 'AD', 'BC', 'BD', 'CD',
                    'ABC', 'ABD', 'ACD', 'BCD', 'ABCD']
    for term in effect_terms:
        high = df[df[term] == 1]['milliseconds'].mean()
        low  = df[df[term] == -1]['milliseconds'].mean()
        all_effects[term] = (high - low) / 2

    # OLS regression with all factors
    formula_terms = ['A', 'B', 'C', 'D']
    formula = 'milliseconds ~ ' + ' * '.join(formula_terms)
    try:
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        print(f"\n  ANOVA Table (Type II):")
        print(f"  {anova_table.to_string()}")
    except Exception as e:
        print(f"  ANOVA failed: {e}")
        anova_table = None

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Factorial ANOVA & Pareto Chart of Effects",
                 fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.945,
             "Which factors matter MOST for pit stop duration?",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(1, 2, wspace=0.3,
                           left=0.06, right=0.97, top=0.88, bottom=0.06)

    # Panel 1: Pareto chart of effects
    ax1 = fig.add_subplot(gs[0, 0])

    sorted_effects = sorted(all_effects.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [s[0] for s in sorted_effects]
    values = [s[1] for s in sorted_effects]
    abs_values = [abs(v) for v in values]
    bar_colors = [COLORS['accent_red'] if v > 0 else COLORS['accent_green'] for v in values]

    bars = ax1.barh(range(len(labels)), abs_values, color=bar_colors,
                    alpha=0.8, height=0.6, edgecolor=COLORS['bg_dark'], linewidth=0.5)

    for i, (bar, val, raw) in enumerate(zip(bars, abs_values, values)):
        sign = '+' if raw > 0 else '-'
        ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                 f'{sign}{val:.0f} ms', va='center', fontsize=10,
                 color=COLORS['text_primary'])

    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=11, fontfamily='monospace')
    ax1.set_xlabel("|Effect| (milliseconds)", fontsize=12)
    ax1.set_title("Pareto Chart: All Effects Ranked by Magnitude", pad=10)
    ax1.invert_yaxis()

    # Significance threshold (2 * SE of effects)
    se_effect = df['milliseconds'].std() / np.sqrt(len(df)) * 2
    ax1.axvline(se_effect * 2, color=COLORS['accent_gold'], linestyle='--',
                alpha=0.7, linewidth=2, label=f'2*SE threshold ({se_effect*2:.0f} ms)')
    ax1.legend(fontsize=9, framealpha=0.9)

    # Panel 2: ANOVA results table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    ax2.set_title("ANOVA Summary (Type II)", fontsize=14, fontweight='bold', pad=15)

    if anova_table is not None:
        # Show top effects from ANOVA
        anova_display = anova_table.sort_values('F', ascending=False).head(12)

        headers_anova = ['Source', 'Sum Sq', 'df', 'F', 'p-value', 'Sig']
        colw = [0.22, 0.17, 0.08, 0.15, 0.18, 0.12]
        rh = 0.06
        sy = 0.92

        # Headers
        for j, (h, w) in enumerate(zip(headers_anova, colw)):
            x = 0.02 + sum(colw[:j])
            ax2.add_patch(plt.Rectangle((x, sy), w - 0.005, rh,
                                        facecolor=COLORS['accent_cyan'], alpha=0.3,
                                        transform=ax2.transAxes, clip_on=False))
            ax2.text(x + w/2, sy + rh/2, h, ha='center', va='center',
                     fontsize=10, fontweight='bold', color=COLORS['text_primary'],
                     transform=ax2.transAxes)

        # Data rows
        for i, (source, row) in enumerate(anova_display.iterrows()):
            y = sy - (i + 1) * rh
            bg = COLORS['bg_card'] if i % 2 == 0 else COLORS['bg_card_alt']

            source_str = str(source).replace(':', ' x ')
            f_val = row['F'] if not np.isnan(row['F']) else 0
            p_val = row['PR(>F)'] if not np.isnan(row.get('PR(>F)', np.nan)) else 1
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else
                   ('*' if p_val < 0.05 else 'ns'))

            vals = [source_str, f"{row['sum_sq']:.0f}",
                    f"{row['df']:.0f}", f"{f_val:.2f}",
                    f"{p_val:.2e}", sig]

            for j, (v, w) in enumerate(zip(vals, colw)):
                x = 0.02 + sum(colw[:j])
                ax2.add_patch(plt.Rectangle((x, y), w - 0.005, rh,
                                            facecolor=bg, alpha=0.8,
                                            transform=ax2.transAxes, clip_on=False))
                color = COLORS['accent_green'] if v in ['***', '**', '*'] else COLORS['text_primary']
                ax2.text(x + w/2, y + rh/2, v, ha='center', va='center',
                         fontsize=9, color=color, transform=ax2.transAxes,
                         family='monospace')

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"\n   [SAVED] {filename}")

    return all_effects


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Normal Probability Plot of Effects
# ─────────────────────────────────────────────────────────────────────────────

def plot_normal_probability(all_effects, filename):
    """
    FIGURE 4: Half-normal probability plot — effects that deviate from the
    line are statistically significant.
    """
    print("\n" + "=" * 70)
    print("  PART D: Normal Probability Plot of Effects")
    print("=" * 70)

    effects = list(all_effects.values())
    labels = list(all_effects.keys())
    abs_effects = [abs(e) for e in effects]

    # Sort by absolute value
    sorted_pairs = sorted(zip(abs_effects, labels, effects), key=lambda x: x[0])
    sorted_abs = [p[0] for p in sorted_pairs]
    sorted_labels = [p[1] for p in sorted_pairs]
    sorted_raw = [p[2] for p in sorted_pairs]

    # Expected quantiles (half-normal)
    n = len(sorted_abs)
    expected = [stats.halfnorm.ppf((i + 0.5) / n) * np.std(effects) for i in range(n)]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Normal Probability Plot of Effects",
                 fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.945,
             "Effects deviating from the line are statistically significant",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(1, 2, wspace=0.3,
                           left=0.08, right=0.95, top=0.88, bottom=0.08)

    # Panel 1: Half-normal plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(sorted_abs, expected, color=COLORS['accent_cyan'],
                s=80, zorder=3, edgecolors=COLORS['bg_dark'], linewidth=0.5)

    # Fit line through the small effects (the "noise" effects)
    noise_n = max(3, n // 2)
    z = np.polyfit(sorted_abs[:noise_n], expected[:noise_n], 1)
    trend = np.poly1d(z)
    x_line = np.linspace(0, max(sorted_abs) * 1.1, 100)
    ax1.plot(x_line, trend(x_line), '--', color=COLORS['accent_red'],
             linewidth=2, label='Expected (noise)')

    # Label significant effects (those far from the line)
    for i in range(n):
        residual = sorted_abs[i] - (expected[i] - z[1]) / z[0] if z[0] != 0 else 0
        if i >= n - 5:  # Label the top 5
            color = COLORS['accent_gold'] if abs(residual) > sorted_abs[0] else COLORS['text_primary']
            ax1.annotate(sorted_labels[i],
                        (sorted_abs[i], expected[i]),
                        textcoords='offset points',
                        xytext=(10, 5), fontsize=11, fontweight='bold',
                        color=COLORS['accent_gold'])

    ax1.set_xlabel("|Effect| (ms)", fontsize=12)
    ax1.set_ylabel("Expected Half-Normal Quantile", fontsize=12)
    ax1.set_title("Half-Normal Probability Plot", pad=10)
    ax1.legend(fontsize=10, framealpha=0.9)

    # Panel 2: Effect magnitude with sign
    ax2 = fig.add_subplot(gs[0, 1])
    colors_bar = [COLORS['accent_red'] if e > 0 else COLORS['accent_green']
                  for e in sorted_raw]
    bars = ax2.barh(range(n), sorted_raw, color=colors_bar, alpha=0.7,
                    height=0.6, edgecolor=COLORS['bg_dark'], linewidth=0.5)

    ax2.set_yticks(range(n))
    ax2.set_yticklabels(sorted_labels, fontsize=10, fontfamily='monospace')
    ax2.set_xlabel("Effect (ms): + = increases duration, - = decreases", fontsize=11)
    ax2.set_title("Signed Effects (sorted by |magnitude|)", pad=10)
    ax2.axvline(0, color=COLORS['text_muted'], linewidth=1)

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Response Surface
# ─────────────────────────────────────────────────────────────────────────────

def plot_response_surface(df, filename):
    """
    FIGURE 5: Response surface showing how pit stop duration varies
    with year and stop number.
    """
    print("\n" + "=" * 70)
    print("  PART E: Response Surface Analysis")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Response Surface: Pit Stop Duration Over Time",
                 fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.945,
             "How has each team tier's pit stop performance evolved?",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(1, 2, wspace=0.3,
                           left=0.06, right=0.97, top=0.88, bottom=0.1)

    # Panel 1: Heatmap of mean duration by year x team tier
    ax1 = fig.add_subplot(gs[0, 0])

    # Bin years for a clean heatmap
    df['year_bin'] = pd.cut(df['year'], bins=range(2011, 2028, 2),
                            labels=[f'{y}-{y+1}' for y in range(2011, 2026, 2)])

    pivot = df.pivot_table(values='milliseconds', index='A_label',
                           columns='year_bin', aggfunc='median')

    if not pivot.empty:
        im = ax1.imshow(pivot.values / 1000, cmap='RdYlGn_r', aspect='auto',
                       vmin=2.0, vmax=6.0)
        ax1.set_xticks(range(len(pivot.columns)))
        ax1.set_xticklabels(pivot.columns, fontsize=9, rotation=45, ha='right')
        ax1.set_yticks(range(len(pivot.index)))
        ax1.set_yticklabels(pivot.index, fontsize=11)
        ax1.set_title("Median Duration (seconds) by Year x Team Tier", pad=10)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j] / 1000
                if not np.isnan(val):
                    color = COLORS['bg_dark'] if val < 4 else COLORS['text_primary']
                    ax1.text(j, i, f'{val:.1f}s', ha='center', va='center',
                             fontsize=10, fontweight='bold', color=color)

        cbar = fig.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
        cbar.set_label("Duration (seconds)", fontsize=10)

    # Panel 2: Trend lines by team tier
    ax2 = fig.add_subplot(gs[0, 1])

    for tier, color, label in [(1, COLORS['accent_red'], 'Top Teams'),
                                (-1, COLORS['accent_blue'], 'Other Teams')]:
        tier_data = df[df['A'] == tier].groupby('year')['milliseconds'].median()
        ax2.plot(tier_data.index, tier_data.values / 1000, '-o', color=color,
                 linewidth=2, markersize=4, label=label, alpha=0.8)

        # Trend line
        z = np.polyfit(tier_data.index, tier_data.values / 1000, 2)
        trend = np.poly1d(z)
        x_smooth = np.linspace(tier_data.index.min(), tier_data.index.max(), 100)
        ax2.plot(x_smooth, trend(x_smooth), '--', color=color,
                 linewidth=1.5, alpha=0.5)

    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Median Duration (seconds)", fontsize=12)
    ax2.set_title("Pit Stop Speed Evolution by Team Tier", pad=10)
    ax2.legend(fontsize=11, framealpha=0.9)

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(df, main_effects, all_effects, filename):
    """
    FIGURE 6: Comprehensive summary of the DOE analysis.
    """
    print("\n" + "=" * 70)
    print("  PART F: Summary & Conclusions")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Problem 5 Summary: Factorial Design Analysis of Pit Stops",
                 fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.945,
             f"2^4 Full Factorial  |  {len(df):,} observations  |  "
             f"4 factors, 11 interactions",
             ha='center', fontsize=12, color=COLORS['accent_cyan'])

    gs = gridspec.GridSpec(1, 2, wspace=0.3,
                           left=0.06, right=0.97, top=0.88, bottom=0.06)

    # Panel 1: Cube plot (show effects on a 3D-projected cube for A, B, D)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_axis_off()
    ax1.set_title("Cube Plot: Mean Duration at Factor Corners", fontsize=14,
                  fontweight='bold', pad=15)

    # Show a projected 2^3 cube (A, B, D) averaged over C
    cube_data = {}
    for a in [-1, 1]:
        for b in [-1, 1]:
            for d in [-1, 1]:
                mask = (df['A'] == a) & (df['B'] == b) & (df['D'] == d)
                cube_data[(a, b, d)] = df.loc[mask, 'milliseconds'].mean() / 1000

    # Draw cube edges
    cube_coords = {
        (-1, -1, -1): (0.15, 0.2),
        (1, -1, -1):  (0.55, 0.2),
        (-1, 1, -1):  (0.15, 0.55),
        (1, 1, -1):   (0.55, 0.55),
        (-1, -1, 1):  (0.3, 0.4),
        (1, -1, 1):   (0.7, 0.4),
        (-1, 1, 1):   (0.3, 0.75),
        (1, 1, 1):    (0.7, 0.75),
    }

    # Edges
    edges = [
        ((-1,-1,-1), (1,-1,-1)), ((-1,-1,-1), (-1,1,-1)),
        ((-1,-1,-1), (-1,-1,1)), ((1,-1,-1), (1,1,-1)),
        ((1,-1,-1), (1,-1,1)),   ((-1,1,-1), (1,1,-1)),
        ((-1,1,-1), (-1,1,1)),   ((-1,-1,1), (1,-1,1)),
        ((-1,-1,1), (-1,1,1)),   ((1,1,-1), (1,1,1)),
        ((1,-1,1), (1,-1,1)),    ((-1,1,1), (1,1,1)),
        ((1,-1,1), (1,1,1)),
    ]

    for (p1, p2) in edges:
        x1, y1 = cube_coords[p1]
        x2, y2 = cube_coords[p2]
        ax1.plot([x1, x2], [y1, y2], '-', color=COLORS['grid_line'],
                 linewidth=1, transform=ax1.transAxes)

    # Nodes
    for key, (x, y) in cube_coords.items():
        val = cube_data.get(key, 0)
        color = COLORS['accent_green'] if val < 3.5 else (
                COLORS['accent_gold'] if val < 5 else COLORS['accent_red'])
        ax1.scatter(x, y, s=120, color=color, zorder=5,
                    edgecolors=COLORS['text_primary'], linewidth=1,
                    transform=ax1.transAxes)
        ax1.text(x, y - 0.06, f'{val:.1f}s', ha='center', fontsize=9,
                 fontweight='bold', color=color, transform=ax1.transAxes)

    # Labels
    ax1.text(0.35, 0.08, 'A: Other Team <----> Top Team', ha='center',
             fontsize=9, color=COLORS['text_muted'], transform=ax1.transAxes)
    ax1.text(0.02, 0.4, 'B: 1st <-> 2nd+', ha='center', rotation=90,
             fontsize=9, color=COLORS['text_muted'], transform=ax1.transAxes)
    ax1.text(0.85, 0.55, 'D: Pre-2018\n<-> 2018+', ha='center',
             fontsize=8, color=COLORS['text_muted'], transform=ax1.transAxes)

    # Panel 2: Conclusions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    ax2.set_title("Key Findings & Conclusions", fontsize=14, fontweight='bold', pad=15)

    # Sort effects by magnitude
    sorted_eff = sorted(all_effects.items(), key=lambda x: abs(x[1]), reverse=True)

    conclusions = [
        (True,  "Most Important Factors"),
        (False, f"  1. {sorted_eff[0][0]}: {sorted_eff[0][1]:+.0f} ms (LARGEST)"),
        (False, f"  2. {sorted_eff[1][0]}: {sorted_eff[1][1]:+.0f} ms"),
        (False, f"  3. {sorted_eff[2][0]}: {sorted_eff[2][1]:+.0f} ms"),
        (False, ""),
        (True,  "Domain Interpretation"),
        (False, "  - Era (D) is likely the biggest factor:"),
        (False, "    modern pit stop technology is vastly faster"),
        (False, "  - Team Tier (A) matters: top teams invest"),
        (False, "    more in pit crew training/equipment"),
        (False, "  - Stop Number (B) & Race Phase (C)"),
        (False, "    have smaller but measurable effects"),
        (False, ""),
        (True,  "Key Interactions"),
        (False, "  - Team x Era (AD): top teams have improved"),
        (False, "    MORE than smaller teams over time"),
        (False, "  - The gap between top and rest has"),
        (False, "    widened in the modern era"),
        (False, ""),
        (True,  "Methodology Note"),
        (False, "  This is observational data analyzed with a"),
        (False, "  DOE framework. While we cannot claim strict"),
        (False, "  causality (no randomization), the factorial"),
        (False, "  design reveals which factors and interactions"),
        (False, "  most strongly ASSOCIATE with pit stop speed."),
        (False, ""),
        (True,  "Connection to Other Problems"),
        (False, "  - Problem 3: qualifying matters, but so does"),
        (False, "    pit strategy (this analysis)"),
        (False, "  - Problem 4: era effects here match the"),
        (False, "    time-series trends in lap times"),
    ]

    for i, (is_header, text) in enumerate(conclusions):
        y = 0.98 - i * 0.032
        color = COLORS['accent_cyan'] if is_header else COLORS['text_primary']
        weight = 'bold' if is_header else 'normal'
        ax2.text(0.05, y, text, fontsize=10, fontweight=weight,
                 color=color, transform=ax2.transAxes, va='top')

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)
    print(f"   [SAVED] {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis():
    df = load_data()

    # Figure 1: Design matrix + main effects
    main_effects = plot_design_and_effects(df, "fig1_design_main_effects.png")

    # Figure 2: Interaction effects
    plot_interactions(df, "fig2_interactions.png")

    # Figure 3: ANOVA + Pareto chart
    all_effects = plot_anova_pareto(df, "fig3_anova_pareto.png")

    # Figure 4: Normal probability plot
    plot_normal_probability(all_effects, "fig4_normal_probability.png")

    # Figure 5: Response surface
    plot_response_surface(df, "fig5_response_surface.png")

    # Figure 6: Summary
    plot_summary(df, main_effects, all_effects, "fig6_summary.png")

    print("\n" + "=" * 70)
    print("  CONCLUSIONS")
    print("=" * 70)
    sorted_eff = sorted(all_effects.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n   Top 5 effects (by magnitude):")
    for term, effect in sorted_eff[:5]:
        print(f"     {term:6s}: {effect:+.1f} ms")
    print(f"\n   All figures saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("  [DONE] Problem 5 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
