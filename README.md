# Tournament Format Simulator

This project provides a Monte Carlo simulator to evaluate how different tournament formats reward skill.  
Teams are assigned numeric “strength” values, which are converted into match win probabilities using:

\[
P(A \text{ beats } B) = \frac{s_A}{s_A + s_B}
\]

## Implemented Formats
- **Worlds-style**  
  16-team Swiss stage (best-of-1, to 3 wins / 3 losses) → top-8 single-elimination playoff (best-of-5).  
- **Double Elimination**  
  Supports both 8-team and 16-team double-elimination brackets (best-of-5, single Grand Final).  
- **Round Robin**  
  16-team league where each team plays each other once or multiple times (configurable rounds).  
  - Supports best-of-N per matchup (e.g. bo1, bo3, bo5).  
  - Tie-break rules: head-to-head, Sonneborn–Berger, then random if still tied.  

## Features
- Monte Carlo simulation of thousands of full tournaments.  
- Sweep experiments: vary one team’s strength to measure its probability of winning in each format.  
- Plot skill curves: championship probability vs head-to-head win rate against the second-best team.  
- Configurable options:
  - Number of simulations  
  - Seeding method (random or fixed)  
  - Match length (bo1 / bo3 / bo5)  
  - Number of round-robin cycles  

## Example Usage
Run the main script to compare the three formats:

```bash
python sweep_formats_vs_headtohead_3formats.py

