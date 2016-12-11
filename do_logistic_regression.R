# define expit() function
expit <- function(x) {
    return(exp(x) / (1 + exp(x)))
}

# load data
pcr <- read.csv("Data/pcr_HER2_positive.csv", header=TRUE)$pcr.num
treatment <- read.csv("Data/treatment_HER2_positive.csv", header=TRUE)$treatment.num
probes <- read.csv("Data/gse_50948_9_probes_HER2_positive_enriched.csv", header=TRUE)
ACTR3B <- probes$ACTR3B

# fit logistic regression and store predicted values
logistic.ACTR3B <-  glm(pcr~ACTR3B * treatment, family="binomial")
coef.logistic.ACTR3B <- coef(logistic.ACTR3B)
ACTR3B.control <- seq(from=min(ACTR3B[treatment==0]), to=max(ACTR3B[treatment==0]), length.out=1000)
ACTR3B.treatment <- seq(from=min(ACTR3B[treatment==1]), to=max(ACTR3B[treatment==1]), length.out=1000)
predicted.control <- expit(coef(logistic.ACTR3B)["(Intercept)"] + coef(logistic.ACTR3B)["ACTR3B"] * ACTR3B.control)
predicted.treatment <- expit(coef(logistic.ACTR3B)["(Intercept)"] + coef(logistic.ACTR3B)["ACTR3B"] * ACTR3B.treatment +
                               coef(logistic.ACTR3B)["treatment"] +
                               coef(logistic.ACTR3B)["ACTR3B:treatment"] * ACTR3B.treatment)

# make plot
png("Plots/logistic_regression_ACTR3B.png", width=480, height=480)
par(mar=c(5.1, 5.1, 4.1, 2.1))
plot(ACTR3B[treatment==0], pcr[treatment==0], col="DarkOrange", type="p", pch=16,
      xlab="ACTR3B", ylab="Probability of 3-year pCR", ylim=c(0, 1), cex=1.2, cex.lab=2, cex.axis=1.8)
points(ACTR3B[treatment==1], pcr[treatment==1], col="blue", type="p", cex=1.2, pch=16)
lines(ACTR3B.control, predicted.control, type="l", col="DarkOrange", lwd=6)
lines(ACTR3B.treatment, predicted.treatment, type="l", col="blue", lwd=6)
dev.off()

