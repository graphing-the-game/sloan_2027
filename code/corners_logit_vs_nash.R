# packages
# _________________________________________

library(tidyverse)
library(sandwich)
library(lmtest)
library(glmnet)
library(lpSolve)


# data prep
# _________________________________________

dat <- read_csv("corners.csv", na = c("", "NA")) %>% #this is what I called the csv, change it to your path!
  filter(observation_status == "complete", elo_match_status == "matched") %>%
  filter(!is.na(call), !is.na(defence), !is.na(is_goal)) %>%
  mutate(
    elo_diff = (offensive_team_pregame_elo - defensive_team_pregame_elo) / 100,
    corner_n = pmin(corner_number, 6),
    call     = fct_infreq(factor(call)),
    defence  = fct_infreq(factor(defence)),
    across(c(trap, zone, quarter, score_state), ~ factor(.x))
  ) %>%
  drop_na(is_goal, call, defence, trap, zone, quarter, score_state,
          corner_n, elo_diff) %>%
  droplevels()

acts_o <- levels(dat$call)
acts_d <- levels(dat$defence)
state_terms <- "trap + zone + quarter + score_state + corner_n + elo_diff"

cat("N corners:", nrow(dat), "| games:", n_distinct(dat$game_id), "\n")


# cell counts
# _________________________________________

n_cell <- unclass(xtabs(~ defence + call, dat))
print(n_cell)


# model A: main effects logit, clustered by game
# _________________________________________

form_A <- as.formula(paste("is_goal ~ call + defence +", state_terms))
mA <- glm(form_A, family = binomial, data = dat)

vcA <- vcovCL(mA, cluster = dat$game_id)
ct  <- coeftest(mA, vcov. = vcA)
ci  <- coefci(mA, vcov. = vcA)

orA <- tibble(term = names(coef(mA)),
              OR   = exp(coef(mA)),
              lo   = exp(ci[, 1]),
              hi   = exp(ci[, 2]),
              p    = ct[, 4])
print(orA, n = Inf)


# model A: adjusted goal rates
# _________________________________________

adj_rate <- function(fit, var, levs) {
  sapply(levs, function(l) {
    tmp <- dat
    tmp[[var]] <- factor(l, levels = levs)
    mean(predict(fit, newdata = tmp, type = "response"))
  })
}

cat("\nAdjusted goal rate by offense action:\n")
print(round(100 * adj_rate(mA, "call", acts_o), 1))
cat("\nAdjusted goal rate by defense action:\n")
print(round(100 * adj_rate(mA, "defence", acts_d), 1))


# model B: ridge logit with penalized offense x defense cells
# _________________________________________

build_X <- function(d) {
  main <- model.matrix(as.formula(paste("~ call + defence +", state_terms)), d)[, -1]
  cell <- model.matrix(~ 0 + call:defence, d)
  X <- cbind(main, cell)
  attr(X, "n_main") <- ncol(main)
  X
}

X  <- build_X(dat)
nm <- attr(X, "n_main")
pf <- c(rep(0, nm), rep(1, ncol(X) - nm))

set.seed(1)
games <- unique(dat$game_id)
fold_of_game <- setNames(sample(rep(1:10, length.out = length(games))), games)
foldid <- unname(fold_of_game[as.character(dat$game_id)])

cvB <- cv.glmnet(X, dat$is_goal, family = "binomial", alpha = 0,
                 penalty.factor = pf, foldid = foldid,
                 standardize = FALSE, type.measure = "deviance")
lam <- cvB$lambda.min
cat("lambda.min:", cvB$lambda.min, "| lambda.1se:", cvB$lambda.1se,
    "| largest lambda in path:", max(cvB$lambda), "\n")


# model B: g-computation payoff matrix
# _________________________________________

gcomp_matrix <- function(fit, lam) {
  P <- matrix(NA_real_, length(acts_d), length(acts_o),
              dimnames = list(acts_d, acts_o))
  for (dd in acts_d) for (oo in acts_o) {
    tmp <- dat
    tmp$call    <- factor(oo, levels = acts_o)
    tmp$defence <- factor(dd, levels = acts_d)
    P[dd, oo] <- mean(predict(fit, newx = build_X(tmp), s = lam, type = "response"))
  }
  P
}
P_ridge <- gcomp_matrix(cvB$glmnet.fit, lam)


# LP solver
# _________________________________________

solve_game <- function(P) {
  m <- nrow(P); n <- ncol(P)
  off <- lp("max", c(rep(0, n), 1),
            rbind(cbind(-P, 1), c(rep(1, n), 0)),
            c(rep("<=", m), "="), c(rep(0, m), 1))
  def <- lp("min", c(rep(0, m), 1),
            rbind(cbind(t(P), -1), c(rep(1, m), 0)),
            c(rep("<=", n), "="), c(rep(0, n), 1))
  list(p = setNames(off$solution[1:n], colnames(P)),
       q = setNames(def$solution[1:m], rownames(P)),
       v = off$solution[n + 1])
}

obs_p <- setNames(as.numeric(prop.table(table(dat$call))[acts_o]), acts_o)
obs_q <- setNames(as.numeric(prop.table(table(dat$defence))[acts_d]), acts_d)

report <- function(P, label) {
  s <- solve_game(P)
  cat("\n=====", label, "=====\n")
  cat("Game value:                 ", round(100 * s$v, 2), "%\n")
  cat("Observed play (same matrix):", round(100 * sum(obs_q * (P %*% obs_p)), 2), "%\n")
  cat("Nash offense vs obs defense:", round(100 * sum(obs_q * (P %*% s$p)), 2), "%\n")
  cat("Nash defense vs obs offense:", round(100 * sum((s$q %*% P) * obs_p), 2), "%\n")
  cat("Offense equilibrium mix (%):\n"); print(round(100 * s$p[s$p > 0.005], 1))
  cat("Defense equilibrium mix (%):\n"); print(round(100 * s$q[s$q > 0.005], 1))
  invisible(s)
}


# laplace matrix vs ridge matrix
# _________________________________________

g_cell <- unclass(xtabs(is_goal ~ defence + call, dat))
P_lap  <- (g_cell + 0.2) / (n_cell + 0.4)

s_lap   <- report(P_lap,   "Laplace alpha = 0.2")
s_ridge <- report(P_ridge, "Ridge logit g-computation (Model B)")


# eval data prep
# _________________________________________

dat <- dat %>%
  filter(!call %in% c("Error", "Other")) %>%
  mutate(trap = fct_lump_min(trap, 100),
         zone = fct_lump_min(zone, 100)) %>%
  droplevels()

dat_u <- dat
dat_l <- dat %>%
  mutate(defence = fct_lump_min(defence, 150, other_level = "Rare defence")) %>%
  droplevels()

print(table(dat_l$defence))


# out-of-sample fit: 10-fold CV by game
# _________________________________________

folds_for <- function(d, k, seed) {
  set.seed(seed)
  g <- unique(d$game_id)
  f <- setNames(sample(rep(1:k, length.out = length(g))), g)
  unname(f[as.character(d$game_id)])
}

oof_glm <- function(form, d, foldid) {
  p <- numeric(nrow(d))
  for (k in unique(foldid)) {
    tr  <- foldid != k
    fit <- suppressWarnings(glm(as.formula(form), binomial, d[tr, ]))
    p[!tr] <- suppressWarnings(predict(fit, d[!tr, ], type = "response"))
  }
  p
}

X_l <- build_X(dat_l)
nm  <- attr(X_l, "n_main")
pf  <- c(rep(0, nm), rep(1, ncol(X_l) - nm))
lams <- c(20, 5, 1, 0.3, 0.1)

oof_ridge <- function(X, y, foldid, lams) {
  out <- matrix(NA_real_, nrow(X), length(lams))
  for (k in unique(foldid)) {
    tr  <- foldid != k
    fit <- glmnet(X[tr, ], y[tr], family = "binomial", alpha = 0, lambda = lams,
                  penalty.factor = pf, standardize = FALSE)
    out[!tr, ] <- predict(fit, X[!tr, ], s = lams, type = "response")
  }
  out
}

forms <- c(
  "m0 intercept"         = "is_goal ~ 1",
  "m1 elo only"          = "is_goal ~ elo_diff",
  "m2 state"             = paste("is_goal ~", state_terms),
  "m3 state+offense"     = paste("is_goal ~ call +", state_terms),
  "m4 state+off+def (A)" = paste("is_goal ~ call + defence +", state_terms)
)
b_names <- paste0("B lambda=", lams)
model_names <- c(names(forms), b_names)

reps <- 3
y <- dat_l$is_goal
preds <- setNames(replicate(length(model_names), matrix(NA_real_, nrow(dat_l), reps),
                            simplify = FALSE), model_names)
for (r in 1:reps) {
  fid <- folds_for(dat_l, 10, r)
  for (m in names(forms)) preds[[m]][, r] <- oof_glm(forms[[m]], dat_l, fid)
  pr <- oof_ridge(X_l, y, fid, lams)
  for (j in seq_along(lams)) preds[[b_names[j]]][, r] <- pr[, j]
}

clip <- function(p) pmin(pmax(p, 1e-6), 1 - 1e-6)
ll   <- function(p) -(y * log(clip(p)) + (1 - y) * log(1 - clip(p)))
auc  <- function(p) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
loss <- sapply(preds, function(P) rowMeans(apply(P, 2, ll)))

paired <- function(ref) {
  t(sapply(model_names, function(m) {
    d   <- loss[, m] - loss[, ref]
    fit <- lm(d ~ 1)
    se  <- sqrt(vcovCL(fit, cluster = dat_l$game_id)[1, 1])
    c(diff = unname(coef(fit)), se = se, z = unname(coef(fit)) / se)
  }))
}

ladder <- tibble(
  model   = model_names,
  logloss = colMeans(loss),
  brier   = sapply(preds, function(P) mean(apply(P, 2, function(p) (y - p)^2))),
  auc     = sapply(preds, function(P) mean(apply(P, 2, auc)))
)
print(ladder, n = Inf)
print(round(paired("m2 state"), 5))
print(round(paired("m4 state+off+def (A)"), 5))


# calibration of model A
# _________________________________________

p4 <- rowMeans(preds[["m4 state+off+def (A)"]])
cal <- tibble(y = y, p = p4) %>%
  mutate(bin = ntile(p, 10)) %>%
  group_by(bin) %>%
  summarise(n = n(), predicted = mean(p), observed = mean(y), .groups = "drop")
print(cal)

slope_fit <- glm(y ~ qlogis(clip(p4)), family = binomial)
cat("Calibration slope:", round(coef(slope_fit)[2], 3), "\n")


# joint tests: offense and defense blocks
# _________________________________________

mA_u <- glm(form_A, binomial, dat_u)
vc_u <- function(x) vcovCL(x, cluster = dat_u$game_id)

for (t in c("call", "defence")) {
  w <- waldtest(mA_u, as.formula(paste(". ~ . -", t)), vcov = vc_u, test = "Chisq")
  cat(sprintf("%-8s Chisq = %6.2f  df = %d  p = %.4g\n",
              t, w$Chisq[2], abs(w$Df[2]), w$`Pr(>Chisq)`[2]))
}


# interaction test: 3-1 to 3-1 vs other defences, crossed with call
# _________________________________________

d_int <- dat_u %>%
  mutate(def_bin = factor(ifelse(defence == "3-1 to 3-1", "base", "alt")))
m_add <- glm(as.formula(paste("is_goal ~ call + def_bin +", state_terms)), binomial, d_int)
m_int <- glm(as.formula(paste("is_goal ~ call * def_bin +", state_terms)), binomial, d_int)
print(waldtest(m_int, m_add,
               vcov = function(x) vcovCL(x, cluster = d_int$game_id),
               test = "Chisq"))


# permutation test: defence formation
# _________________________________________

def_stat <- function(d, dev0) {
  fit <- glm(form_A, binomial, d)
  cf  <- coef(fit)[grep("^defence", names(coef(fit)))]
  c(min_coef = min(0, cf, na.rm = TRUE), lr = dev0 - deviance(fit))
}
dev0 <- deviance(glm(as.formula(paste("is_goal ~ call +", state_terms)), binomial, dat_u))
obs  <- def_stat(dat_u, dev0)

set.seed(1)
B <- 500
perm <- t(replicate(B, {
  d <- dat_u %>%
    group_by(game_id, offensive_team_id) %>%
    mutate(defence = defence[sample.int(n())]) %>%
    ungroup()
  def_stat(d, dev0)
}))

cat("Observed most favourable log-odds coef:", round(obs["min_coef"], 3),
    "| permutation median:", round(median(perm[, "min_coef"]), 3),
    "| p =", mean(perm[, "min_coef"] <= obs["min_coef"]), "\n")
cat("Observed LR for defence block:", round(obs["lr"], 2),
    "| permutation median:", round(median(perm[, "lr"]), 2),
    "| p =", mean(perm[, "lr"] >= obs["lr"]), "\n")


# game bootstrap: best defence and best offensive call
# _________________________________________

def_names <- paste0("defence", levels(dat_u$defence)[-1])
off_names <- paste0("call",    levels(dat_u$call)[-1])
by_game   <- split(dat_u, dat_u$game_id)

boot <- lapply(1:500, function(i) {
  d  <- do.call(rbind, by_game[sample(names(by_game), replace = TRUE)])
  cf <- coef(suppressWarnings(glm(form_A, binomial, d)))
  list(def = cf[def_names], off = cf[off_names])
})
bd <- do.call(rbind, lapply(boot, `[[`, "def")); colnames(bd) <- levels(dat_u$defence)[-1]
bo <- do.call(rbind, lapply(boot, `[[`, "off")); colnames(bo) <- levels(dat_u$call)[-1]

best_def <- apply(cbind(`3-1 to 3-1` = 0, bd), 1, function(r) names(which.min(r)))
best_off <- apply(cbind(Sweep = 0, bo),        1, function(r) names(which.max(r)))

print(round(sort(table(best_def) / length(best_def), decreasing = TRUE), 3))
print(round(sort(table(best_off) / length(best_off), decreasing = TRUE), 3))
