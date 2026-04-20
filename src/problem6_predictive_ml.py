"""
================================================================================
PROBLEM 6: Advanced Predictive Modeling (Zero-Sum Framework)
================================================================================
Course Topic: Multinomial Logistic Regression & Learning to Rank

Research Question:
    Given pre-race information (grid, driver form, constructor strength),
    can we accurately predict a race outcome, recognizing that F1 is a
    zero-sum game (only exactly 3 drivers can reach the podium per race)?

Methodology updates (vs naïve models):
    1. Multinomial Logistic Regression: Instead of binary Yes/No, we model
       mutually exclusive outcome tiers (Win, Podium, Points, No Points).
    2. XGBoostRanker: A "Learning to Rank" algorithm that predicts the
       relative ordering of exactly the 20 drivers in a specific race,
       resolving the independent probability flaw.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, classification_report
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs" / "problem6"
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
    'text_primary': '#EAEAEA',
    'text_muted':   '#8892A0',
    'grid_line':    '#2A2E3A',
}

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
    'savefig.dpi':        200,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  COLORS['bg_dark'],
})


def add_watermark(fig):
    fig.text(0.99, 0.01, "F1 Statistical Analysis", ha='right', va='bottom',
             fontsize=8, color=COLORS['text_muted'], alpha=0.4, style='italic')


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features():
    """Build the feature matrix with ONLY pre-race information."""
    print("=" * 70)
    print("  PROBLEM 6: Advanced Predictive Modeling (Ranking)")
    print("=" * 70)
    print("\n[*] Loading and engineering features...")

    results  = pd.read_csv(DATA_DIR / "results.csv", na_values="\\N")
    races    = pd.read_csv(DATA_DIR / "races.csv", na_values="\\N")
    drivers  = pd.read_csv(DATA_DIR / "drivers.csv", na_values="\\N")
    constructors = pd.read_csv(DATA_DIR / "constructors.csv", na_values="\\N")
    status   = pd.read_csv(DATA_DIR / "status.csv", na_values="\\N")

    # Base merge
    df = results.merge(races[['raceId', 'year', 'round', 'name', 'circuitId', 'date']],
                       on='raceId', how='left').rename(columns={'name': 'race_name'})
    df = df.merge(drivers[['driverId', 'surname']], on='driverId', how='left')

    # Convert numerics
    for col in ['grid', 'positionOrder', 'points']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter: modern era with valid grid (2003+), exclude pit lane starts
    df = df[(df['year'] >= 2003) & (df['grid'] > 0)].copy()

    # CRITICAL: We must sort by chronological order for rolling features
    df = df.sort_values(['driverId', 'year', 'round']).reset_index(drop=True)

    # ── ENGINEER PRE-RACE FEATURES ────────────────────────────────────────

    # 1. Driver rolling average points (last 5 races)
    df['driver_avg_pts_last5'] = (
        df.groupby('driverId')['points']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # 2. Constructor cumulative points (BEFORE this race)
    df['constructor_cum_pts'] = (
        df.groupby(['constructorId', 'year'])['points']
        .transform(lambda x: x.shift(1).cumsum())
    )

    # 3. Driver DNF rate (last 10 races)
    df['statusId'] = pd.to_numeric(df['statusId'], errors='coerce')
    df['dnf'] = (df['statusId'] != 1).astype(int)
    df['dnf_rate_last10'] = (
        df.groupby('driverId')['dnf']
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # 4. Driver circuit history (average finish here)
    df['driver_circuit_avg_finish'] = (
        df.groupby(['driverId', 'circuitId'])['positionOrder']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Drop rows requiring lookback that don't have it yet
    feature_cols = ['grid', 'driver_avg_pts_last5', 'constructor_cum_pts',
                    'dnf_rate_last10', 'driver_circuit_avg_finish']
    df = df.dropna(subset=feature_cols + ['positionOrder', 'points']).copy()
    df[feature_cols] = df[feature_cols].fillna(0)

    # ── DEFINE TARGETS ────────────────────────────────────────────────────
    
    # Target 1: LTR Relevance
    # F1 points can exceed 31 (2014 double points, fastest lap, etc.), which crashes
    # XGBoost's exponential NDCG calculation. We use 24 - position order instead.
    df['relevance'] = (24 - df['positionOrder'].fillna(24)).clip(lower=0).astype(int)

    # Target 2: Outcome Tiers for Multinomial
    def get_tier(pos):
        if pos == 1: return 0      # Winner
        if pos <= 3: return 1      # Podium
        if pos <= 10: return 2     # Points
        return 3                   # Outside Points
    df['tier'] = df['positionOrder'].apply(get_tier)

    # CRITICAL: Re-sort by raceId, then by grid so groups are contiguous for XGBoost
    df = df.sort_values(['year', 'round', 'grid']).reset_index(drop=True)

    print(f"   [OK] {len(df):,} samples ({df['year'].min()}-{df['year'].max()})")
    print(f"   Features: {feature_cols}")

    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# PART A: Multinomial Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def run_multinomial_logit(df, feature_cols):
    """
    Train a Multinomial Logistic Regression model to predict exactly which 
    tier a driver will land in, plotting the coefficient shifts.
    """
    print("\n" + "=" * 70)
    print("  PART A: Multinomial Logistic Regression")
    print("=" * 70)

    X = df[feature_cols].values
    y = df['tier'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    # Note: 'multinomial' requires a solver like 'lbfgs'
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy (4-class): {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Winner', 'Podium', 'Points', 'Outside']))

    # ── Figure 1: Confusion Matrix & Coefficients ─────────────────────────
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Multinomial Logistic Regression: Predictions vs Reality",
                 fontsize=18, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.25, 
                           left=0.06, right=0.97, top=0.88, bottom=0.1)

    # Panel 1: Prediction Reliability (100% Stacked Bar)
    from sklearn.metrics import confusion_matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize by PREDICTED class (columns sum to 100%)
    cm_norm = cm / (cm.sum(axis=0) + 1e-9) * 100
    tier_names = ['Winner', 'Podium (P2-P3)', 'Points (P4-10)', 'Outside']
    bottoms = np.zeros(4)
    colors = [COLORS['accent_gold'], COLORS['accent_cyan'], COLORS['accent_blue'], COLORS['bg_card_alt']]
    
    for actual_idx in range(4):
        heights = cm_norm[actual_idx, :]
        ax1.bar(tier_names, heights, bottom=bottoms, 
                color=colors[actual_idx], edgecolor=COLORS['bg_dark'], linewidth=1.5,
                label=tier_names[actual_idx])
                       
        # Add percentage text if segment is visually large enough (e.g., > 8%)
        for pred_idx, h in enumerate(heights):
            if h > 8:
                ax1.text(pred_idx, bottoms[pred_idx] + h/2, f"{int(h)}%", 
                         ha='center', va='center', fontweight='bold', 
                         color=COLORS['bg_dark'] if actual_idx < 3 else COLORS['text_primary'],
                         fontsize=11)
        bottoms += heights
        
    ax1.set_ylabel("What Actually Happened (%)", fontsize=11)
    ax1.set_xlabel("The Model's Prediction", fontsize=12, fontweight='bold', color=COLORS['accent_cyan'])
    ax1.set_title("When the model predicts X, what happens?", pad=15)
    
    # Custom legend for Actuals
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], [f"Actually {l}" for l in labels[::-1]], 
               title="Reality:", loc='upper right', bbox_to_anchor=(1, 1), 
               fontsize=9, title_fontsize=10, framealpha=0.9)


    # Panel 2: Coefficients
    ax2 = fig.add_subplot(gs[0, 1])
    coefs = model.coef_
    tier_colors = [COLORS['accent_gold'], COLORS['accent_cyan'], COLORS['accent_blue'], COLORS['text_muted']]
    x = np.arange(len(feature_cols))
    width = 0.2
    for i in range(4):
        ax2.bar(x + (i - 1.5) * width, coefs[i], width, 
                label=tier_names[i], color=tier_colors[i], alpha=0.9)
    ax2.axhline(0, color=COLORS['text_muted'], linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_cols, fontsize=10)
    ax2.set_ylabel("Impact on Probability (Log-Odds)", fontsize=11)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.set_title("Why did it predict that? (Feature Importance)", pad=15)

    add_watermark(fig)
    fig.savefig(OUTPUT_DIR / "fig1_multinomial_coefficients.png")
    plt.close(fig)
    print(f"   [SAVED] fig1_multinomial_coefficients.png")


# ─────────────────────────────────────────────────────────────────────────────
# PART B: XGBoostRanker (Learning to Rank)
# ─────────────────────────────────────────────────────────────────────────────

def run_learning_to_rank(df, feature_cols):
    """
    Train XGBRanker to evaluate the zero-sum nature of F1.
    We train on pre-2022 races, and test on 2022+ races.
    """
    print("\n" + "=" * 70)
    print("  PART B: XGBoost Ranker (LTR)")
    print("=" * 70)

    # Time-based split: Train on past, simulate on recent races
    train_mask = df['year'] < 2022
    test_mask = df['year'] >= 2022

    df_train = df[train_mask]
    df_test = df[test_mask]

    # Ensure contiguous race groups
    sizes_train = df_train.groupby('raceId').size().values
    sizes_test = df_test.groupby('raceId').size().values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    y_train = df_train['relevance'].values

    X_test = scaler.transform(df_test[feature_cols].values)
    y_test = df_test['relevance'].values

    # Train LTR Model
    ranker = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        n_estimators=100,
        random_state=42,
        learning_rate=0.1
    )

    print(f"   Training XGBRanker on {len(sizes_train)} races (pre-2022)...")
    ranker.fit(
        X_train, y_train,
        group=sizes_train,
        eval_set=[(X_test, y_test)],
        eval_group=[sizes_test],
        verbose=False
    )

    # ── Evaluate the Ranker (Simulation) ──────────────────────────────────
    # We will score every race in the test set, simulate a predicted 1-to-N ranking,
    # and compute top-1 and top-3 accuracy per race.
    
    df_test['pred_score'] = ranker.predict(X_test)
    
    metrics = {'top1_correct': 0, 'top3_recall': [], 'perfect_podium': 0}
    test_races = df_test['raceId'].unique()

    # Find an interesting race for plotting (e.g., 2023 British GP)
    plot_race_df = None
    
    for rid in test_races:
        race_data = df_test[df_test['raceId'] == rid].copy()
        
        # Rank by predicted score (descending)
        race_data = race_data.sort_values('pred_score', ascending=False)
        predicted_podium = race_data.iloc[:3]['driverId'].values
        predicted_winner = predicted_podium[0]
        
        # Actual podium (safely handle missing positions)
        actual_sorted = race_data.sort_values('positionOrder')
        actual_podium = actual_sorted[actual_sorted['positionOrder'] <= 3]['driverId'].values
        actual_winner = actual_sorted.iloc[0]['driverId'] if len(actual_sorted) > 0 else None

        # Top 1 Accuracy (Did we predict the winner?)
        if predicted_winner == actual_winner:
            metrics['top1_correct'] += 1

        # Top 3 Recall (Out of the 3 real podium drivers, how many did we predict?)
        overlap = len(set(predicted_podium).intersection(set(actual_podium)))
        metrics['top3_recall'].append(overlap / 3.0)

        # Perfect podium (Predicted all 3 correctly, regardless of order)
        if overlap == 3:
            metrics['perfect_podium'] += 1
            
        # Save British GP 2023 for plotting (Silverstone, 2023)
        if race_data['year'].iloc[0] == 2023 and race_data['circuitId'].iloc[0] == 9:
            plot_race_df = race_data

    n_races = len(test_races)
    avg_top3 = np.mean(metrics['top3_recall']) * 3.0  # Average drivers correctly placed in top 3
    
    print(f"\n   Testing on {n_races} modern races (2022-2023):")
    print(f"     Predicted the exact Winner: {metrics['top1_correct']}/{n_races} ({metrics['top1_correct']/n_races*100:.1f}%)")
    print(f"     Average actual podiums found in predicted top-3: {avg_top3:.2f} / 3.0")
    print(f"     Perfect podium trio predicted: {metrics['perfect_podium']}/{n_races} ({metrics['perfect_podium']/n_races*100:.1f}%)")

    # ── Figure: Feature Importance & Race Simulation ──────────────────────
    if plot_race_df is not None:
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle("XGBoost Ranker: Learning to Rank F1 Drivers",
                     fontsize=20, fontweight='bold', y=0.98)
        fig.text(0.5, 0.94,
                 "Unlike independent classifiers, LTR scores evaluate drivers relative to each other in a zero-sum race",
                 ha='center', fontsize=12, color=COLORS['accent_cyan'])

        gs = gridspec.GridSpec(1, 2, wspace=0.25, left=0.06, right=0.97, top=0.88, bottom=0.1)

        # Panel 1: Feature Importance
        ax1 = fig.add_subplot(gs[0, 0])
        importances = ranker.feature_importances_
        idx = np.argsort(importances)
        bars = ax1.barh(range(len(importances)), importances[idx], color=COLORS['accent_gold'], alpha=0.8)
        ax1.set_yticks(range(len(importances)))
        ax1.set_yticklabels(np.array(feature_cols)[idx], fontsize=10)
        ax1.set_xlabel("Relative Importance (Gain)", fontsize=11)
        ax1.set_title("Ranking Feature Importance", pad=15)

        # Panel 2: Race Simulation (2023 British GP) - Slope Chart
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate predicted discrete rank (1=best, 2=second best)
        plot_race_df['predicted_rank'] = plot_race_df['pred_score'].rank(ascending=False, method='first')
        
        for _, row in plot_race_df.iterrows():
            y_pred = row['predicted_rank']
            y_act = row['positionOrder']
            if pd.isna(y_act) or y_act == 0: 
                y_act = 21 # Treat DNF/Missing as last
                
            y_act = int(y_act)
            
            # Color logic (Green = Perfect, Cyan = Close, Gray = Miss)
            err = abs(y_pred - y_act)
            if err == 0: color = COLORS['accent_green']
            elif err <= 2: color = COLORS['accent_cyan']
            else: color = COLORS['text_muted']
            
            # Draw line
            ax2.plot([0, 1], [y_pred, y_act], color=color, alpha=0.7, linewidth=2, marker='o', markersize=6)
            
            # Labels
            ax2.text(-0.05, y_pred, row['surname'], ha='right', va='center', fontsize=10)
            res_str = f"P{y_act}" if y_act <= 20 else "DNF"
            weight = 'bold' if y_act <= 3 else 'normal'
            res_color = COLORS['accent_gold'] if y_act <= 3 else COLORS['text_primary']
            ax2.text(1.05, y_act, res_str, ha='left', va='center', fontsize=10, 
                     fontweight=weight, color=res_color)

        ax2.set_xlim(-0.3, 1.3)
        ax2.set_ylim(21.5, 0.5) # Invert so P1 is at top
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["Model Predicted Order", "Actual F1 Race Result"], fontsize=12, fontweight='bold', color=COLORS['accent_cyan'])
        
        # Grid lines for positions
        ax2.set_yticks(range(1, 21))
        ax2.set_yticklabels([f"Rank {i}" for i in range(1, 21)], fontsize=9, color=COLORS['text_muted'])
        
        race_name = plot_race_df['race_name'].iloc[0]
        race_year = plot_race_df['year'].iloc[0]
        ax2.set_title(f"Rank Simulation: {race_year} {race_name}\n"
                      f"(Lines show error. Green = exact match, Blue = close)", pad=15)

        add_watermark(fig)
        fig.savefig(OUTPUT_DIR / "fig2_xgboost_ranker.png")
        plt.close(fig)
        print(f"\n   [SAVED] fig2_xgboost_ranker.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis():
    df, feature_cols = build_features()

    # Part A: Multinomial Logistic
    run_multinomial_logit(df, feature_cols)

    # Part B: Learning To Rank
    run_learning_to_rank(df, feature_cols)

    print("\n" + "=" * 70)
    print("  CONCLUSIONS: ZERO-SUM MODELING")
    print("=" * 70)
    print("\n   Standard classification evaluates drivers in a vacuum. Advanced")
    print("   approaches model the interactions:")
    print("     1. Multinomial Logistic extracts feature shifts uniquely targeting")
    print("        tier probability (e.g. grid purely impacts Winner/Podium).")
    print("     2. XGBoostRanker mathematically enforces that drivers compete ")
    print("        against the other 19 drivers on track, producing brilliant")
    print("        relative predictions.")
    print(f"\n   All figures saved to: {OUTPUT_DIR}")
    print("=" * 70)
    print("  [DONE] Problem 6 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
