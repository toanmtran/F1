# 🏎️ Understanding Formula 1 Racing Through the Data

> This document teaches you how F1 works **through the lens of the dataset**. By the end, you'll understand both the sport and every CSV file.

---

## Table of Contents

1. [What is Formula 1?](#what-is-formula-1)
2. [The Structure of an F1 Season](#the-structure-of-an-f1-season)
3. [A Race Weekend Explained](#a-race-weekend-explained)
4. [Points, Championships & Standings](#points-championships--standings)
5. [The Dataset: Entity-Relationship Overview](#the-dataset-entity-relationship-overview)
6. [File-by-File Data Dictionary](#file-by-file-data-dictionary)
7. [Key Relationships to Remember](#key-relationships-to-remember)

---

## What is Formula 1?

Formula 1 (F1) is the **highest class of international single-seater auto racing**. Think of it as the Champions League of motorsport. Here's what makes it unique:

- **Teams (Constructors)**: Companies like Ferrari, Mercedes, McLaren, and Red Bull **design and build their own cars**. Each team fields **exactly 2 drivers** per season. The team is called a "constructor" because they *construct* their own car — they don't just buy one off the shelf.

- **Drivers**: The athletes who race the cars. They are employed by a constructor. A driver can switch teams between seasons (e.g., Lewis Hamilton drove for McLaren, then Mercedes, then Ferrari across his career).

- **Circuits**: The racetracks. F1 races at ~20 different tracks around the world each season. Some are permanent circuits (Silverstone, Monza), others are street circuits built on public roads (Monaco, Singapore).

- **Regulations**: F1 has strict technical rules that change over the years (engine types, aerodynamics, tire compounds). These regulation changes create "eras" — e.g., the turbo-hybrid era (2014–present) where Mercedes dominated, or the ground-effect era (2022+) where Red Bull dominated.

### Why does this matter for the data?

The dataset spans **1950–2026** (77 seasons, 1,171 races). This long history means:
- The sport's rules have changed drastically (points systems, qualifying formats, sprint races)
- Technology has evolved (lap times have gotten much faster, then slower when regulations change)
- Some columns only exist for modern races (e.g., pit stop data starts ~2011, sprint races start 2021)

---

## The Structure of an F1 Season

```
Season (1 year)
├── Race 1 (Australian Grand Prix)
│   ├── Free Practice 1, 2, 3
│   ├── Qualifying (Q1, Q2, Q3)
│   ├── [Sprint Race — only at some events since 2021]
│   └── Main Race (the Grand Prix)
├── Race 2 (Bahrain Grand Prix)
│   └── ...
├── ...
└── Race 20-24 (Abu Dhabi Grand Prix — season finale)
```

**A season** consists of ~16–24 individual races (called **Grands Prix**), held at different circuits around the world over ~9 months (March–December).

After each race, **points are awarded** to the top finishers. At the end of the season, the driver with the most cumulative points wins the **Drivers' World Championship**, and the team whose two drivers scored the most combined points wins the **Constructors' World Championship**.

### In the data:
| Concept | File | Key columns |
|---|---|---|
| A season | `seasons.csv` | `year` — just the year and a Wikipedia link |
| The list of races in a season | `races.csv` | `year`, `round` (1st race, 2nd race…), `circuitId`, `name`, `date` |
| Which circuit each race takes place at | `circuits.csv` | `circuitId`, `name`, `location`, `country`, `lat`, `lng`, `alt` |

> [!NOTE]
> `races.csv` also has columns for `fp1_date`, `fp2_date`, `fp3_date`, `quali_date`, and `sprint_date`. These are `\N` (null) for older races where this data wasn't tracked.

---

## A Race Weekend Explained

A typical F1 race weekend spans **Friday → Sunday** and has several sessions:

### 1. Free Practice (FP1, FP2, FP3)
- **Purpose**: Teams test car setups, tire strategies, and drivers learn the track.
- **In the data**: Only the schedule is in `races.csv` (`fp1_date`, `fp1_time`, etc.). No lap-by-lap FP data in this dataset.

### 2. Qualifying (Saturday)
Qualifying determines the **starting order** (the "grid") for the race. It has 3 rounds:

| Round | What happens | Eliminated |
|---|---|---|
| **Q1** (18 min) | All 20 drivers set their fastest lap | Bottom 5 drivers are eliminated (start P16–P20) |
| **Q2** (15 min) | Remaining 15 drivers | Bottom 5 eliminated (start P11–P15) |
| **Q3** (12 min) | Top 10 drivers fight for **pole position** (P1) | Fastest driver starts at the front of the grid |

**"Pole position"** = starting 1st. This is a huge advantage because overtaking in F1 is difficult.

### In the data:
| File | What it captures |
|---|---|
| `qualifying.csv` | Each driver's lap time in Q1 (`q1`), Q2 (`q2`), and Q3 (`q3`). If a driver was eliminated in Q1, their `q2` and `q3` will be `\N`. |

> Columns: `qualifyId, raceId, driverId, constructorId, number, position, q1, q2, q3`
> 
> - `position` = their final qualifying position (1 = pole)
> - Times like `1:26.572` mean 1 minute 26.572 seconds

### 3. Sprint Race (select weekends only, since 2021)
- A **short race** (~100 km, about 1/3 of a normal race). Introduced in 2021 at a few events per season.
- Awards fewer points than the main race.

### In the data:
| File | What it captures |
|---|---|
| `sprint_results.csv` | Same structure as `results.csv` but for sprint races. Only **502 rows** (small — sprints are new and rare). |

### 4. The Grand Prix (Sunday) — The Main Race
This is the marquee event. ~20 drivers race for typically **50–70 laps** (~300 km). Key concepts:

#### Starting Grid
- Drivers line up in the order determined by qualifying (or sprint, at sprint weekends).
- The `grid` column in `results.csv` tells you where each driver started. `grid = 1` = started from pole position.

#### Laps
- Each lap is one complete loop of the circuit. The race is a fixed number of laps (varies by circuit: Monaco ~78 laps of a 3.3 km track, Spa ~44 laps of a 7 km track).
- Every lap time for every driver is recorded in `lap_times.csv` — **this is the largest file: 618,766 rows**.

#### Pit Stops
- During the race, drivers must stop in the **pit lane** (a service road alongside the track) to change tires. Rules mandate at least **1 pit stop** per race (must use at least 2 different tire compounds).
- A pit stop takes ~2–4 seconds for the tire change, plus ~20 seconds lost driving through the pit lane.
- Strategy around *when* and *how many times* to pit is a core part of F1 competition.

#### DNF (Did Not Finish)
- Not every driver finishes every race. Cars break down (engine failure, gearbox failure), crash (accident, collision), or have other issues.
- The `status.csv` file lists **140 different race outcomes**: "Finished", "Accident", "Engine", "Gearbox", "Spun off", "+1 Lap", etc.
- `statusId = 1` means "Finished" normally. Everything else is either a DNF or finishing laps behind the leader.

#### Race Result
- The winner is the first to cross the finish line after all laps are completed.
- Other drivers' times are recorded as a **gap to the winner** (e.g., `+5.478` seconds).

### In the data:

| File | Rows | What it captures |
|---|---|---|
| `results.csv` | 27,304 | **The core results table**. One row per driver per race. Contains grid position, finish position, points scored, laps completed, race time, fastest lap, and status. |
| `lap_times.csv` | 618,766 | Every lap of every driver. `raceId, driverId, lap, position, time, milliseconds`. |
| `pit_stops.csv` | 22,193 | Every pit stop. `raceId, driverId, stop` (1st stop, 2nd stop…), `lap`, `duration`, `milliseconds`. |
| `status.csv` | 140 | Lookup table: `statusId → status` (e.g., 1 = "Finished", 3 = "Accident", 5 = "Engine"). |

> [!IMPORTANT]
> **`results.csv` is the single most important file.** It links drivers, constructors, races, and outcomes together. Almost every analysis you do will start here.

#### Key columns in `results.csv`:
| Column | Meaning |
|---|---|
| `resultId` | Unique ID for this result row |
| `raceId` | Links to `races.csv` → tells you which Grand Prix |
| `driverId` | Links to `drivers.csv` → tells you who drove |
| `constructorId` | Links to `constructors.csv` → tells you which team |
| `number` | The driver's car number |
| `grid` | Starting position (from qualifying). `0` = started from pit lane |
| `position` | Finishing position. `\N` if the driver didn't finish |
| `positionOrder` | Finishing order (always a number, even for DNFs — they're placed at the back) |
| `points` | Points scored in this race |
| `laps` | Number of laps completed |
| `time` | Race time or gap to winner (e.g., `1:34:50.616` or `+5.478`) |
| `milliseconds` | Total race time in ms (only for finishers) |
| `fastestLap` | Which lap number was their fastest |
| `rank` | Their fastest lap rank compared to other drivers |
| `fastestLapTime` | Their fastest lap time (e.g., `1:27.452`) |
| `fastestLapSpeed` | Average speed on their fastest lap (km/h) |
| `statusId` | Links to `status.csv` → tells you why they stopped |

---

## Points, Championships & Standings

### Points Systems (have changed over the years!)

| Era | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th | Fastest Lap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2010–present | 25 | 18 | 15 | 12 | 10 | 8 | 6 | 4 | 2 | 1 | +1 (if in top 10) |
| 2003–2009 | 10 | 8 | 6 | 5 | 4 | 3 | 2 | 1 | — | — | — |
| 1991–2002 | 10 | 6 | 4 | 3 | 2 | 1 | — | — | — | — | — |
| Earlier eras | Various other systems |

> [!WARNING]
> Because the points system has changed, **you cannot directly compare raw points across eras.** A driver scoring 400 points in 2023 (24 races × 25 max) is not comparable to 100 points in 2005 (19 races × 10 max). Always normalize or compare within the same era.

### Standings

After each race, the **cumulative standings** are updated:

| File | What it tracks |
|---|---|
| `driver_standings.csv` | After each race: driver's cumulative points, position in the championship, and number of wins so far. **35,427 rows**. |
| `constructor_standings.csv` | Same but for teams. **13,664 rows**. |
| `constructor_results.csv` | Points scored by each constructor at each race. **12,898 rows**. |

Each row represents a snapshot **after** a specific race (`raceId`). So you can trace how the championship battle evolved race by race throughout a season.

---

## The Dataset: Entity-Relationship Overview

```mermaid
erDiagram
    SEASONS ||--o{ RACES : "has many"
    CIRCUITS ||--o{ RACES : "hosts"
    RACES ||--o{ RESULTS : "produces"
    RACES ||--o{ QUALIFYING : "has"
    RACES ||--o{ SPRINT_RESULTS : "may have"
    RACES ||--o{ LAP_TIMES : "records"
    RACES ||--o{ PIT_STOPS : "records"
    RACES ||--o{ DRIVER_STANDINGS : "updates"
    RACES ||--o{ CONSTRUCTOR_STANDINGS : "updates"
    RACES ||--o{ CONSTRUCTOR_RESULTS : "tallies"
    DRIVERS ||--o{ RESULTS : "achieves"
    DRIVERS ||--o{ QUALIFYING : "participates"
    DRIVERS ||--o{ LAP_TIMES : "sets"
    DRIVERS ||--o{ PIT_STOPS : "makes"
    DRIVERS ||--o{ DRIVER_STANDINGS : "holds"
    CONSTRUCTORS ||--o{ RESULTS : "fields"
    CONSTRUCTORS ||--o{ QUALIFYING : "enters"
    CONSTRUCTORS ||--o{ CONSTRUCTOR_STANDINGS : "holds"
    CONSTRUCTORS ||--o{ CONSTRUCTOR_RESULTS : "earns"
    STATUS ||--o{ RESULTS : "describes"

    SEASONS {
        int year PK
        string url
    }
    CIRCUITS {
        int circuitId PK
        string name
        string location
        string country
        float lat
        float lng
        int alt
    }
    RACES {
        int raceId PK
        int year FK
        int round
        int circuitId FK
        string name
        date date
    }
    DRIVERS {
        int driverId PK
        string forename
        string surname
        date dob
        string nationality
        int number
    }
    CONSTRUCTORS {
        int constructorId PK
        string name
        string nationality
    }
    RESULTS {
        int resultId PK
        int raceId FK
        int driverId FK
        int constructorId FK
        int grid
        int position
        float points
        int laps
        int milliseconds
        int statusId FK
    }
    STATUS {
        int statusId PK
        string status
    }
