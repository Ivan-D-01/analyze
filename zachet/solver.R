df <- data.frame(
  F_1 = factor(
    c(
      "T", "T", "T", "R", "R", "R", "R", "R", "R", "R",
      "R", "R", "R", "R", "R", "R", "R", "R", "R", "R",
      "R", "R", "S", "S", "S", "S", "S", "S", "S", "S",
      "S", "S", "S", "S", "S", "S", "S", "S"
    ),
    levels = c("R", "S", "T")
  ),
  F_2 = factor(
    c(
      "C", "C", "C", "A", "A", "A", "A", "A", "A", "A",
      "A", "B", "B", "B", "C", "C", "C", "C", "C", "C",
      "C", "C", "A", "A", "A", "A", "A", "B", "B", "B",
      "B", "B", "C", "C", "C", "C", "C", "C"
    ),
    levels = c("A", "B", "C")
  ),
  F_3 = factor(
    c(
      "U", "V", "W", "U", "V", "W", "U", "V", "W", "U",
      "V", "W", "U", "V", "W", "U", "V", "W", "U", "V",
      "W", "U", "V", "W", "U", "V", "W", "U", "V", "W",
      "U", "V", "W", "U", "V", "W", "U", "V"
    ),
    levels = c("U", "V", "W")
  ),
  C = c(
    10.6, 10.5, 10.3, 11.9, 11.9, 11.5, 12.0, 12.0, 12.3, 12.4,
    12.2, 12.5, 12.3, 12.8, 12.8, 12.8, 12.8, 12.7, 12.7, 12.5,
    12.4, 12.6, 10.2, 10.5, 11.3, 11.7, 11.1, 11.5, 11.7, 11.4,
    11.5, 11.4, 12.1, 12.2, 12.4, 12.2, 11.8, 11.4
  ),
  G = c(
    20.0, 19.0, 20.0, 46.5, 44.0, 43.0, 43.0, 41.5, 45.5, 39.8,
    39.0, 45.7, 45.1, 39.0, 39.0, 38.0, 40.0, 38.0, 38.5, 38.5,
    37.0, 40.5, 22.6, 25.2, 23.1, 25.2, 24.0, 23.5, 25.0, 23.0,
    24.5, 24.5, 24.1, 28.0, 23.6, 24.3, 26.0, 23.0
  )
)

print_header <- function(title) {
  cat("\n")
  cat(title, "\n", sep = "")
  cat(strrep("-", nchar(title)), "\n", sep = "")
}

options(contrasts = c("contr.sum", "contr.poly"))

cat(sprintf("Shape: %d x %d\n\n", nrow(df), ncol(df)))

print_header("1. One-way ANOVA: C ~ F_1")
one_way_model <- lm(C ~ F_1, data = df)
print(anova(one_way_model))

print_header("2. Two-way ANOVA: C ~ F_1 * F_2")
two_way_model <- lm(C ~ F_1 * F_2, data = df)
print(anova(two_way_model))

print_header("3. Three-way ANOVA: C ~ F_1 * F_2 * F_3")
three_way_model <- lm(C ~ F_1 * F_2 * F_3, data = df)
print(anova(three_way_model))

print_header("MANOVA: [C, G] ~ F_1")
manova_model <- manova(cbind(C, G) ~ F_1, data = df)
print(summary(manova_model, test = "Wilks"))

cat("\n")
print(summary.aov(manova_model))

pdf("boxplots_R.pdf", width = 14, height = 4)
par(mfrow = c(1, 3), mar = c(5, 4, 3, 1))
for (factor_name in c("F_1", "F_2", "F_3")) {
  boxplot(
    as.formula(paste("C ~", factor_name)),
    data = df,
    main = factor_name,
    xlab = factor_name,
    ylab = "C",
    col = "#cfe8ff",
    border = "#222222"
  )
}
dev.off()

cat("\nSaved boxplots to boxplots_R.pdf\n")

