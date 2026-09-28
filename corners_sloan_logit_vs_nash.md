# Penalty corner Nash first pass: logit check and results

## What we were trying to do

The Nash table is built from a matrix of goal rates, one cell for every pairing of an offensive call and a defensive formation. Most of the data sits in a few cells. The 3-1 to 3-1 row has thousands of corners, while many other cells have one or two. The linear program treats every cell as equally trustworthy, so a cell with 2 corners and 0 goals looks as convincing as one with 2,000 corners and a 14.8% goal rate. The logit is a check on that. It estimates P(goal) using all the data at once, so sparse cells borrow strength from the rest of the sample, and we can then rebuild the matrix from the model and see whether the equilibrium survives.

## What the logit does

A logistic regression models the probability of a goal on the log-odds scale:

log( p / (1 - p) ) = b0 + b1*x1 + b2*x2 + ...

Each x is a feature of the corner: the offensive call, the defensive formation, trap, zone, quarter, score state, corner number, and the Elo gap. Each coefficient shifts the log-odds of a goal relative to a baseline (Sweep for offense, 3-1 to 3-1 for defense). Exponentiating gives an odds ratio, so 1.08 means 8% higher odds of a goal and 0.33 means about two thirds lower.

Two details matter. Standard errors are clustered by game (897 games), because corners from the same match are not independent. And to compare with the Nash table, I convert coefficients into adjusted goal rates by predicting every corner as if it had used each action in turn and averaging (g-computation).

**Model A** has main effects only: one effect per call and one per formation, added on the log-odds scale. It cannot represent a matchup.

**Model B** adds a coefficient for every offense-by-defense cell, with a ridge penalty pulling each toward zero. The penalty strength was chosen by 10-fold cross-validation with folds assigned by game. Model B's g-computation matrix is what gets fed to the LP.

## The data

10,522 corners from 897 games after cleaning. Rows are defense, columns are offense (BTI is Back to Inserter, OSI is Opposite side insert).

| Defense | Sweep | Hit | Flick | Error | Stick | Glove | Other | BTI | OSI |
|---|---|---|---|---|---|---|---|---|---|
| 3-1 to 3-1 | 2120 | 2063 | 1613 | 1003 | 1049 | 697 | 445 | 249 | 11 |
| Other | 90 | 56 | 35 | 17 | 8 | 20 | 14 | 34 | 15 |
| 3-1 to 3-1 reverse | 56 | 41 | 69 | 35 | 15 | 18 | 13 | 6 | 7 |
| 3-1 to 2-2 low | 43 | 66 | 61 | 33 | 23 | 13 | 11 | 6 | 3 |
| 3-1 to 2-2 high | 34 | 30 | 16 | 23 | 20 | 13 | 17 | 6 | 0 |
| 2-2 to 2-2 high | 11 | 15 | 11 | 14 | 10 | 2 | 6 | 3 | 9 |
| 2-2 to 3-1 | 15 | 22 | 8 | 5 | 6 | 4 | 4 | 1 | 1 |
| 2-2 to 3-1 reverse | 4 | 12 | 12 | 4 | 2 | 1 | 3 | 1 | 24 |
| 3 man defense | 8 | 6 | 8 | 10 | 5 | 5 | 1 | 2 | 0 |
| Double runner | 3 | 5 | 16 | 6 | 2 | 2 | 0 | 1 | 0 |
| 2-2 to 2-2 low | 5 | 2 | 2 | 2 | 1 | 1 | 0 | 0 | 2 |

The top row holds the large majority of the data. Below it, most cells have fewer than 20 corners and many have fewer than 5. "Error" is a failed insertion, so it is not a strategy.

## Model A results

Odds ratios, game-clustered 95% CIs. Only the terms that matter are shown. Trap, zone, quarter, score state and corner number were all indistinguishable from zero.

| Term | OR | 95% CI | p |
|---|---|---|---|
| Elo difference (per 100) | 1.08 | 1.06 to 1.11 | 8.8e-11 |
| Offense: stick-side pass | 0.733 | 0.593 to 0.907 | 0.0042 |
| Offense: Back to Inserter | 1.26 | 0.924 to 1.72 | 0.14 |
| Offense: Flick | 1.03 | 0.868 to 1.23 | 0.72 |
| Offense: Hit | 0.853 | 0.728 to 1.00 | 0.050 |
| Defense: 2-2 to 3-1 | 0.328 | 0.133 to 0.811 | 0.016 |
| Defense: 3 man defense | 1.82 | 0.892 to 3.72 | 0.10 |

Adjusted goal rates: among calls, Back to Inserter is highest (21.0%), Sweep and Flick are close (17.4% and 17.9%), and stick-side passes are lowest (13.4%). Among formations, 2-2 to 3-1 is lowest (5.6% vs 15.2% for 3-1 to 3-1), but that row has only about 66 corners and is the best of ten alternatives, so the p-value overstates it.

## Model B results

Cross-validation picked the largest penalty on the path (lambda = 5.49), which shrinks every offense-by-defense interaction to about zero. Out of sample, letting cells deviate from "offense effect plus defense effect" made predictions worse. So Model B is additive, its matrix has a pure saddle point, and its LP answer (Back to Inserter against 2-2 to 3-1, 100% each) is just the best main effect on each side and not a mixed equilibrium.

## Comparing the two matrices

| | Laplace (alpha 0.2) | Ridge (Model B) |
|---|---|---|
| Game value | 15.72% | 8.12% |
| Observed play, same matrix | 15.36% | 15.23% |
| Nash offense vs observed defense | 16.77% | 21.01% |
| Nash defense vs observed offense | 11.91% | 5.65% |
| Offense support | Hit 38.5%, BTI 36.2%, Glove 24.7% | BTI 100% |
| Defense support | 2-2 to 3-1 41.9%, 3-1 reverse 30.0%, 2-2 to 3-1 reverse 27.6% | 2-2 to 3-1 100% |

The Laplace matrix agrees with the ridge matrix in the big top row and goes wild below it (91.7% for Double runner against glove passes, 50% in cells with 1 to 6 corners). The prior adds less than half a corner per cell, so it does little to stop that. Its equilibrium leans on cells like those, and its support shifted a lot from the version in the paper, which is what fitting noise looks like.

## Evaluating the models

I ran several checks to see whether either model is doing real work.

- **Out-of-sample fit (10-fold CV by game).** AUC is about 0.556 for every model from state-only up to Model B. Log loss is 0.442 against 0.444 for a constant, and Elo alone gets 0.441. Adding offense, defense, or interactions gave no improvement (paired z-scores between -0.3 and +0.3).
- **Calibration.** The slope for Model A is 0.68, so its predictions are too spread out (10% to 24% predicted, about 12% to 20% observed). That is overfitting.
- **Defense.** The joint test for the defense block gives p = 0.28, and a permutation test that shuffles formations gives p = 0.30. The most favorable formation (log-odds -1.00) is about what the best of ten looks like when formation does nothing (permutation median -0.95). In the game bootstrap, 2-2 to 3-1 is the best defense in only 49% of resamples.
- **Interactions.** A better-powered test (3-1 to 3-1 versus all other formations, crossed with call) gives p = 0.48. Model B beats Model A at no penalty level.
- **Offense.** The offense block is significant in sample (p = 0.008), driven mostly by stick-side passes, but the out-of-sample gain is negligible. Back to Inserter is the best call in 72% of bootstrap resamples, but that rests on about 350 corners.

## What this means

Neither model does much, and the reason, I believe, is mostly the data/general college field hockey playcalling. About 85% of corners are against a single formation, so almost every other cell has too few corners to estimate anything, and a goal happens on only about 1 in 7 corners, which makes each observation a very noisy measurement. The models pick up Elo and little else. The Nash equilibrium feels unreliable here, because the LP searches for the most favorable cells, and in tiny cells, those are mostly noise. I would not present the equilibrium supports as findings, and I would be unsure about claiming defenses under-mix. The data can neither confirm nor rule it out.

What I think is supportable, at this stage, is a descriptive result: Elo dominates, stick-side passes underperform, and no formation effect is detectable at this sample size.

## What would help

The data is heavily concentrated in one formation, and goals are a rare, noisy outcome, so a full-resolution payoff matrix doesn't seem estimable from this sample. I have some ideas to get around these problems, in order of how good I think they are:

- Fewer, coarser actions (for example 3-1 versus 2-2 family, and a handful of call types) so each cell has more counts.
- Held-out samples for the Nash, to test whether it beats observed play on corners it has not seen.
- A check on who plays 2-2 to 3-1, in case the effect is about the teams and not the formation.
- A denser outcome than goals, if we have it. (This would be hard at this point)

