alpha <- 0.05

dat <- data.frame(
  F1 = factor(
    c("T", "T", "T", "R", "R", "R", "R", "R", "R", "R",
      "R", "R", "R", "R", "R", "R", "R", "R", "R", "R",
      "R", "R", "S", "S", "S", "S", "S", "S", "S", "S",
      "S", "S", "S", "S", "S", "S", "S", "S"),
    levels = c("R", "S", "T")
  ),
  F2 = factor(
    c("C", "C", "C", "A", "A", "A", "A", "A", "A", "A",
      "A", "B", "B", "B", "C", "C", "C", "C", "C", "C",
      "C", "C", "A", "A", "A", "A", "A", "B", "B", "B",
      "B", "B", "C", "C", "C", "C", "C", "C"),
    levels = c("A", "B", "C")
  ),
  Y = c(18.8, 17.5, 18.7, 15.4, 15.5, 15.8, 15.7, 15.7,
        15.9, 16.1, 15.6, 16.2, 15.8, 15.9, 15.2, 15.8,
        15.5, 15.4, 15.3, 15.5, 15.9, 15.5, 16.4, 16.5,
        16.4, 16.8, 16.5, 15.5, 16.6, 16.8, 16.8, 16.6,
        16.6, 16.4, 16.5, 16.6, 16.4, 16.5)
)

print(head(dat, 3))
stopifnot(nrow(dat) == 38)
stopifnot(isTRUE(all.equal(dat$Y[1:3], c(18.8, 17.5, 18.7))))

str(dat)
print(table(dat$F1, dat$F2))

print(aggregate(Y ~ F1, data = dat, FUN = function(x) c(n = length(x), mean = mean(x), sd = sd(x))))
print(aggregate(Y ~ F2, data = dat, FUN = function(x) c(n = length(x), mean = mean(x), sd = sd(x))))
print(aggregate(Y ~ F1 + F2, data = dat, FUN = function(x) c(n = length(x), mean = mean(x), sd = sd(x))))

one_f1 <- aov(Y ~ F1, data = dat)
one_f2 <- aov(Y ~ F2, data = dat)

print(summary(one_f1))
print(model.tables(one_f1, type = "means"))
p_one_f1 <- summary(one_f1)[[1]][["Pr(>F)"]][1]
if (!is.na(p_one_f1) && p_one_f1 < alpha) {
  print(TukeyHSD(one_f1))
}

print(summary(one_f2))
print(model.tables(one_f2, type = "means"))
p_one_f2 <- summary(one_f2)[[1]][["Pr(>F)"]][1]
if (!is.na(p_one_f2) && p_one_f2 < alpha) {
  print(TukeyHSD(one_f2))
}

additive <- aov(Y ~ F1 + F2, data = dat)
additive_reversed <- aov(Y ~ F2 + F1, data = dat)
interaction_model <- lm(Y ~ F1 * F2, data = dat)

print(summary(additive))
print(summary(additive_reversed))
print(drop1(additive, test = "F"))
print(alias(interaction_model))
print(anova(additive, interaction_model))

print(shapiro.test(residuals(one_f1)))
print(shapiro.test(residuals(one_f2)))
print(shapiro.test(residuals(additive)))

print(bartlett.test(Y ~ F1, data = dat))
print(bartlett.test(Y ~ F2, data = dat))
print(bartlett.test(Y ~ interaction(F1, F2), data = dat))

print(fligner.test(Y ~ F1, data = dat))
print(fligner.test(Y ~ F2, data = dat))
print(fligner.test(Y ~ interaction(F1, F2), data = dat))

pdf("zachet_anova_plots.pdf", width = 9, height = 7)
par(mfrow = c(2, 2))
boxplot(Y ~ F1, data = dat, main = "Y by F1", xlab = "F1", ylab = "Y")
boxplot(Y ~ F2, data = dat, main = "Y by F2", xlab = "F2", ylab = "Y")
interaction.plot(dat$F2, dat$F1, dat$Y, xlab = "F2", ylab = "mean(Y)", trace.label = "F1")
plot.new()
par(mfrow = c(2, 2))
plot(additive)
dev.off()