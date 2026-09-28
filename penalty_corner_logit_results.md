# Penalty corner Nash first pass: logit check and results

## What we were trying to do

The Nash table in the paper is built from a 7x7 matrix of goal rates, one cell for every pairing of an offensive call and a defensive formation. Most of the data sits in a few cells. The 3-1 to 3-1 row alone has thousands of corners, while some other cells have one or two. The linear program treats every cell as equally trustworthy, so a cell with 2 corners and 0 goals looks just as convincing as a cell with 2,000 corners and a 14.8% goal rate. That is the worry with the Nash result, and the logit is a way of checking it.

The idea is to estimate P(goal) with a model that uses all the data at once, so that a sparse cell borrows strength from the rest of the sample. Then we can rebuild the matrix from the model, solve the same LP, and see whether the equilibrium survives.

## What the logit does

A logistic regression models the probability of a goal on the log-odds scale:

log( p / (1 - p) ) = b0 + b1*x1 + b2*x2 + ...

Each x is a feature of the corner: which offensive call was used, which defensive formation, the trap type, the injection zone, the quarter, the score state, the corner number, and the Elo gap between the two teams. Each coefficient says how much that feature shifts the log-odds of a goal compared with a baseline. Exponentiating a coefficient gives an odds ratio. An odds ratio of 1.08 means the odds of a goal are 8% higher for a one-unit increase in that variable. An odds ratio of 0.33 means the odds are about two thirds lower than baseline.

The baseline for the offensive call is Sweep, and for the defense it is 3-1 to 3-1, since those are the reference levels I set. Score state is compared with "drawing" and quarter with quarter 1. Trap and zone are also compared against a baseline level.

Two details matter for reading the output.

**Clustered standard errors.** Corners from the same game are not independent. The same two teams, the same day, the same goalkeeper. If we treated the 10,522 corners as independent draws, the standard errors would be too small. We cluster by game (897 games), which widens the intervals to reflect that the real amount of independent information is closer to the number of games than the number of corners.

**Adjusted goal rates.** An odds ratio is awkward to compare with the goal rates in the Nash table. So for each action I take every corner in the data, pretend it used that action, keep everything else about the corner the same, predict the goal probability, and average. The result is a goal rate for each action that has been adjusted for zone, trap, Elo, score and so on. This is often called g-computation.

We fit two models.

**Model A** has main effects only. Each offensive call has one effect and each defensive formation has one effect, and they add up on the log-odds scale. This tells us which actions look better or worse on average once the state is controlled for. It cannot represent a matchup, such as one call being especially good against one formation. That would require an interaction.

**Model B** adds a separate coefficient for each of the offense-by-defense cells, which is what a matchup effect is. With that many cells and so little data in most of them, we would overfit badly if we estimated them freely. So they get a ridge penalty, which pulls each one toward zero. When a cell is pulled to zero, its predicted goal rate is whatever the main effects say it should be. The strength of the penalty (lambda) was chosen by 10-fold cross-validation, with folds assigned by game so that corners from the same match never end up on both sides of a split. The state controls and the action main effects were not penalized.

Model B is the one that produces the full 7x7 matrix by g-computation. That matrix is then fed to the same minimax LP as before.

## The data after cleaning

10,522 corners from 897 games (31 corners were dropped because quarter was missing). Rows are the defensive formation, columns are the offensive call. "Stick" and "Glove" are the two pass-or-slip options, BTI is Back to Inserter, and OSI is Opposite side insert.

**Cell counts**

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

The top row holds the large majority of the data. Below it, most cells have fewer than 20 corners and many have fewer than 5.

## Model A results

Odds ratios with 95% confidence intervals (clustered by game). Offense is compared with Sweep, defense with 3-1 to 3-1.

| Term | OR | 95% CI | p |
|---|---|---|---|
| (Intercept) | 0.156 | 0.081 to 0.302 | 3.4e-08 |
| **Offense** | | | |
| Hit | 0.853 | 0.728 to 1.00 | 0.050 |
| Flick | 1.03 | 0.868 to 1.23 | 0.72 |
| Error | 0.337 | 0.263 to 0.431 | 6.1e-18 |
| Pass or slip to stick side | 0.733 | 0.593 to 0.907 | 0.0042 |
| Pass or slip to glove side | 0.855 | 0.680 to 1.08 | 0.18 |
| Other | 0.796 | 0.606 to 1.05 | 0.10 |
| Back to Inserter | 1.26 | 0.924 to 1.72 | 0.14 |
| Opposite side insert | 0.749 | 0.312 to 1.80 | 0.52 |
| **Defense** | | | |
| Other | 1.14 | 0.833 to 1.56 | 0.41 |
| 3-1 to 3-1 reverse | 0.908 | 0.633 to 1.30 | 0.60 |
| 3-1 to 2-2 low | 1.18 | 0.858 to 1.64 | 0.30 |
| 3-1 to 2-2 high | 0.812 | 0.458 to 1.44 | 0.48 |
| 2-2 to 2-2 high | 0.637 | 0.290 to 1.40 | 0.26 |
| 2-2 to 3-1 | 0.328 | 0.133 to 0.811 | 0.016 |
| 2-2 to 3-1 reverse | 0.652 | 0.355 to 1.20 | 0.17 |
| 3 man defense | 1.82 | 0.892 to 3.72 | 0.10 |
| Double runner | 1.16 | 0.516 to 2.59 | 0.73 |
| 2-2 to 2-2 low | 1.20 | 0.522 to 2.75 | 0.67 |
| **Trap** | | | |
| Other/Error | 0.934 | 0.643 to 1.36 | 0.72 |
| Outside 5m | 0.774 | 0.218 to 2.75 | 0.69 |
| Short | 0.991 | 0.523 to 1.88 | 0.98 |
| T1 | 0.889 | 0.662 to 1.19 | 0.43 |
| T2 | 0.833 | 0.611 to 1.13 | 0.25 |
| **Zone** | | | |
| 1 | 1.26 | 0.706 to 2.23 | 0.44 |
| 1.5 | 1.53 | 0.846 to 2.76 | 0.16 |
| 2 | 1.60 | 0.855 to 3.00 | 0.14 |
| 2.5 | 1.52 | 0.706 to 3.28 | 0.28 |
| Error | 1.55 | 0.396 to 6.10 | 0.53 |
| **State** | | | |
| Quarter 2 | 1.09 | 0.914 to 1.31 | 0.33 |
| Quarter 3 | 1.01 | 0.828 to 1.23 | 0.92 |
| Quarter 4 | 1.08 | 0.875 to 1.33 | 0.47 |
| Score: losing | 0.944 | 0.799 to 1.12 | 0.50 |
| Score: winning | 0.958 | 0.829 to 1.11 | 0.56 |
| Corner number (capped at 6) | 1.00 | 0.966 to 1.04 | 0.83 |
| Elo difference (per 100) | 1.08 | 1.06 to 1.11 | 8.8e-11 |

**Adjusted goal rates by offensive call**

| Call | Goal rate |
|---|---|
| Sweep | 17.4% |
| Hit | 15.3% |
| Flick | 17.9% |
| Error | 6.7% |
| Pass or slip to stick side | 13.4% |
| Pass or slip to glove side | 15.3% |
| Other | 14.4% |
| Back to Inserter | 21.0% |
| Opposite side insert | 13.7% |

**Adjusted goal rates by defensive formation**

| Formation | Goal rate |
|---|---|
| 3-1 to 3-1 | 15.2% |
| Other | 17.0% |
| 3-1 to 3-1 reverse | 14.0% |
| 3-1 to 2-2 low | 17.5% |
| 3-1 to 2-2 high | 12.8% |
| 2-2 to 2-2 high | 10.3% |
| 2-2 to 3-1 | 5.6% |
| 2-2 to 3-1 reverse | 10.5% |
| 3 man defense | 24.4% |
| Double runner | 17.2% |
| 2-2 to 2-2 low | 17.7% |

**What Model A says**

- Elo is the only state variable with a clear effect. A 100 point edge for the offense raises the odds of a goal by about 8%. Zone, trap, quarter, score state and corner number are all indistinguishable from zero at this sample size.
- "Error" is a failed insertion, so it is an outcome and not a choice the offense makes. It has no business in the game matrix.
- Among real calls, the stick-side pass is the only one that is clearly worse than Sweep. Sweep, Flick and Hit are within a couple of points of each other. Back to Inserter has the highest adjusted rate but its interval comfortably includes 1.
- On defense, 2-2 to 3-1 is the only formation that separates from the baseline, at 5.6% against 15.2%. That row has roughly 66 corners in total, the interval runs from 0.13 to 0.81, and it is the best of ten alternatives to the baseline, so the p-value of 0.016 is less impressive than it looks on its face.

## Model B results

Cross-validation chose lambda = 5.49. That was also the 1-SE value and also the largest lambda in the path. In other words, CV wanted the strongest penalty available, which shrinks every offense-by-defense interaction to essentially zero. Out of sample, letting individual cells deviate from "offense effect plus defense effect" made the predictions worse.

The g-computation matrix confirms this. Every row of it is nearly the same pattern of goal rates scaled up or down.

**Ridge g-computation matrix (goal rate, %)**

| Defense | Sweep | Hit | Flick | Error | Stick | Glove | Other | BTI | OSI |
|---|---|---|---|---|---|---|---|---|---|
| 3-1 to 3-1 | 17.5 | 15.3 | 18.0 | 6.7 | 13.5 | 15.4 | 14.5 | 21.1 | 13.9 |
| Other | 19.5 | 17.1 | 20.0 | 7.6 | 15.1 | 17.2 | 16.2 | 23.4 | 15.5 |
| 3-1 to 3-1 reverse | 16.2 | 14.1 | 16.6 | 6.1 | 12.4 | 14.2 | 13.3 | 19.5 | 12.8 |
| 3-1 to 2-2 low | 20.1 | 17.7 | 20.6 | 7.9 | 15.6 | 17.7 | 16.7 | 24.0 | 16.0 |
| 3-1 to 2-2 high | 14.7 | 12.9 | 15.1 | 5.5 | 11.3 | 12.9 | 12.1 | 17.9 | 11.6 |
| 2-2 to 2-2 high | 11.9 | 10.4 | 12.3 | 4.4 | 9.1 | 10.4 | 9.8 | 14.6 | 9.3 |
| 2-2 to 3-1 | 6.6 | 5.6 | 6.8 | 2.3 | 4.9 | 5.7 | 5.3 | 8.1 | 5.0 |
| 2-2 to 3-1 reverse | 12.2 | 10.6 | 12.5 | 4.5 | 9.2 | 10.6 | 10.0 | 14.9 | 9.5 |
| 3 man defense | 27.7 | 24.7 | 28.4 | 11.6 | 22.0 | 24.8 | 23.4 | 32.5 | 22.6 |
| Double runner | 19.7 | 17.3 | 20.2 | 7.7 | 15.3 | 17.4 | 16.4 | 23.6 | 15.7 |
| 2-2 to 2-2 low | 20.2 | 17.8 | 20.7 | 7.9 | 15.7 | 17.9 | 16.8 | 24.2 | 16.1 |

Because the matrix is additive, it has a pure-strategy saddle point. The best offensive call (Back to Inserter) and the best defensive formation (2-2 to 3-1) simply win in every row and column. The LP result below is a reading of the best main effect on each side, and it is not a mixed equilibrium in the sense the paper describes.

## Laplace-smoothed matrix (the current one, alpha = 0.2)

**Goal rate, %**

| Defense | Sweep | Hit | Flick | Error | Stick | Glove | Other | BTI | OSI |
|---|---|---|---|---|---|---|---|---|---|
| 3-1 to 3-1 | 17.0 | 14.5 | 19.3 | 7.4 | 13.8 | 14.1 | 14.6 | 19.7 | 10.5 |
| Other | 15.7 | 12.8 | 28.8 | 6.9 | 26.2 | 20.6 | 8.3 | 18.0 | 27.3 |
| 3-1 to 3-1 reverse | 19.9 | 22.2 | 17.6 | 6.2 | 7.8 | 1.1 | 9.0 | 18.7 | 16.2 |
| 3-1 to 2-2 low | 18.9 | 30.4 | 13.4 | 3.6 | 17.9 | 16.4 | 36.8 | 18.7 | 5.9 |
| 3-1 to 2-2 high | 18.0 | 7.2 | 19.5 | 5.1 | 15.7 | 1.5 | 24.1 | 34.4 | 50.0 |
| 2-2 to 2-2 high | 10.5 | 20.8 | 19.3 | 8.3 | 1.9 | 8.3 | 3.1 | 35.3 | 12.8 |
| 2-2 to 3-1 | 1.3 | 9.8 | 2.4 | 3.7 | 18.7 | 27.3 | 4.5 | 14.3 | 14.3 |
| 2-2 to 3-1 reverse | 4.5 | 17.7 | 33.9 | 4.5 | 8.3 | 14.3 | 5.9 | 14.3 | 9.0 |
| 3 man defense | 26.2 | 50.0 | 14.3 | 11.5 | 3.7 | 22.2 | 14.3 | 91.7 | 50.0 |
| Double runner | 35.3 | 22.2 | 7.3 | 3.1 | 50.0 | 91.7 | 50.0 | 14.3 | 50.0 |
| 2-2 to 2-2 low | 22.2 | 50.0 | 50.0 | 8.3 | 14.3 | 14.3 | 50.0 | 50.0 | 8.3 |

Compare the two matrices. In the smoothed one, the big cells (top row) look sensible and agree with the ridge matrix. Below that it gets wild: 91.7% for Double runner against glove passes, 50% in cells that have 1 to 6 corners, 1.1% and 1.3% in others. None of those cells has enough data to say anything. The Laplace prior with alpha = 0.2 adds less than half a corner of information to each cell, so it does very little to stop this.

## LP results side by side

All numbers are goal rates. "Observed play" is the raw observed mix of calls and formations evaluated on the same matrix that the LP uses, so it is comparable to the Nash rows. It replaces the raw 16.5% we were comparing to before.

| | Laplace (alpha 0.2) | Ridge (Model B) |
|---|---|---|
| Game value | 15.72% | 8.12% |
| Observed play | 15.36% | 15.23% |
| Nash offense vs observed defense | 16.77% | 21.01% |
| Nash defense vs observed offense | 11.91% | 5.65% |
| Best-response offense vs observed defense | 20.24% | 21.01% |
| Best-response defense vs observed offense | 8.07% | 5.65% |

**Equilibrium supports**

| | Laplace | Ridge |
|---|---|---|
| Offense | Hit 38.5%, Back to Inserter 36.2%, Glove pass 24.7%, Flick 0.5% | Back to Inserter 100% |
| Defense | 2-2 to 3-1 41.9%, 3-1 reverse 30.0%, 2-2 to 3-1 reverse 27.6% | 2-2 to 3-1 100% |

## How I read all of this

The strongest thing we have is that the offense side of the story is fairly consistent. The stick-side pass is bad, Error is not a choice, and Sweep, Flick and Hit are close to each other. There is a hint that Back to Inserter is good, and about 350 corners is not much to hang that on.

The Nash equilibrium from the smoothed matrix does not look like a real result. Its support changed a lot from the version in the paper. Part of that is that the action set changed (we now have Error, Other, and split high/low formations), but a support that reshuffles that much when the categories move is usually fitting noise in small cells. The offense mix leans on Back to Inserter, and about 250 of its roughly 350 corners are against 3-1 to 3-1. Every other cell in that column has between 1 and 34 corners.

The ridge model is a cleaner check but it says something uncomfortable. Cross-validation found no evidence of matchup effects, so the model is additive and the LP just picks the best action on each side. This does not prove there are no interactions. The test has low power, because the data is concentrated against one defense, so it is mostly testing offense interactions against 3-1 to 3-1. But it means we cannot currently claim a mixed-strategy equilibrium on the strength of this data.

On the under-mixing question, the finding that defenses use 3-1 to 3-1 about 90% of the time and would do better with something else rests almost entirely on the 2-2 to 3-1 row. That row has 66 corners and was picked out as the best of ten. The ridge model did not penalize the defense main effects, so it did not shrink that number either. Until that is checked, "defenses under-mix" is a hypothesis and not a result.

## What would change this picture

- Removing Error and Other from the offensive action set, since they are not strategies.
- Lumping the rare defensive formations so no row has fewer than roughly 150 corners.
- Penalizing the defense main effects too, and checking whether 2-2 to 3-1 stays near 5%.
- Bootstrapping by game and counting how often 2-2 to 3-1 comes out as the best defense.
- Checking who is actually playing 2-2 to 3-1. If it is a handful of teams, or mostly against weak opponents, the effect is about the teams and not the formation.
