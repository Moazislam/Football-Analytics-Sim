# Phoenix Opta – Football Analytics & Season Simulator ⚽📊

Phoenix Opta is a **desktop football analytics application** built with **Python + PyQt6**.  
It combines **Monte Carlo simulation**, **Poisson goal modeling**, and **historical data with time decay** to predict league outcomes and explore advanced football statistics interactively.

The app supports the **Top 5 European leagues** and provides both **future season projections** and **historical stats exploration** in a modern, dark-themed UI.

---

## 🚀 Features

### 🔮 Season Prediction (Monte Carlo)
- Simulates the remainder of the season **10,000 times**
- Uses:
  - Poisson goal model
  - Team attack & defense strengths
  - Home advantage
  - Time decay for historical matches
  - Optional xG data if available
- Outputs:
  - Expected points
  - Expected goal difference
  - Title probability
  - Top 4 / Top 6 probability
  - Relegation probability

### 🔥 Position Probability Heatmap
- Visual probability matrix for **finishing positions (1–20)**
- Color intensity scales with probability
- Team logos supported

### 📊 Stats Explorer
- Explore **any season or all seasons**
- League standings with:
  - W / D / L
  - GF / GA / GD
  - Points
- Generic metrics:
  - Goals
  - xG
  - Any numeric CSV column
- Automatic charts:
  - Leaderboards
  - Distributions (histograms)

---

## 🏆 Supported Leagues

| League | CSV File |
|------|---------|
| 🇬🇧 Premier League | `E0.csv` |
| 🇪🇸 La Liga | `SP1.csv` |
| 🇩🇪 Bundesliga | `D1.csv` |
| 🇮🇹 Serie A | `I1.csv` |
| 🇫🇷 Ligue 1 | `F1.csv` |

> CSVs must follow the **football-data.co.uk** format (or equivalent).

---

## 📂 Required CSV Columns

Minimum required:
- `Season`
- `HomeTeam`
- `AwayTeam`
- `FullTimeHomeGoals`
- `FullTimeAwayGoals`

Optional (recommended):
- `HomeTeamxG`
- `AwayTeamxG`

If xG is missing, the model **automatically falls back to goals**.

---

## 🧠 Model Logic Overview

- **Attack strength** = weighted goals (or xG) scored
- **Defense strength** = weighted goals (or xG) conceded
- **Time decay**:
  ```text
  weight = decay_factor^(current_season - match_season)
  Expected goals:

λ_home = attack_home × defense_away × home_advantage × league_avg
λ_away = attack_away × defense_home × league_avg


Monte Carlo simulation:

Generates scores using Poisson distribution

Updates table standings per simulation

Aggregates probabilities across all simulations
