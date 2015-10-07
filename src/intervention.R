suppressPackageStartupMessages(library("forecast"))

detectAO <- function(object, alpha=0.05, robust=TRUE) {
    # Programmed by Kung-Sik Chan, modified by Jeffrey Knockel
    # Note: this function was originally from the cran package TSA, which is
    # GPL >= 2 licensed
    #
    # This function detects whether there are any additive outliers. It
    # implements the test statistic lambda_{2,t} proposed by Chang, Chen and
    # Tiao (1988).
    #
    # Arguments:
    # object: an ARMA model
    # alpha: family significance level (5% is the default); Bonferroni rule is
    #        used to control the family error rate.
    # robust: if true, the noise standard deviation is estimated by mean
    #         absolute residuals times sqrt(pi / 2). Otherwise, it is the
    #         estimated by sqrt(sigma2) from the arima fit.
    #
    # Returns a list with components:
    # ind: a list containing the time indices
    # lambda2: the test statistics of the found AO

    resid <- residuals(object)
    if (robust) {
        sigma <- sqrt(pi / 2) * mean(abs(resid), na.rm=TRUE)
    } else {
        sigma <- sqrt(object$sigma2)
    }
    resid[is.na(resid)] <- 0
    piwt <- ARMAtoMA(ar=-object$mod$theta, ma=-object$mod$phi, lag.max=length(resid) - 1)
    piwt <- c(1, piwt)
    omega <- filter(c(0 * resid[-1], rev(resid)), filter=piwt, sides=1, method='convolution')
    rho2 <- 1 / cumsum(piwt * piwt)
    omega <- omega * rho2
    lambda2T <- omega / sigma / sqrt(rho2)
    lambda2T <- rev(lambda2T)
    cutoff <- qnorm(1 - alpha / 2 / length(lambda2T))
    out <- abs(lambda2T) > cutoff
    ind <- seq(lambda2T)[out]
    lambda2 <- lambda2T[out]
    list(lambda2=lambda2, ind=ind)
}

arima.order <- function(x, stationary=FALSE) {
    # Returns c(p, d, q), the fitted ARIMA(p, d, q) model for x.
    available <- x[!is.na(x)]
    if (length(available) == 0 || all(available == available[1])) {
        order <- c(0, 0, 0)
    } else {
        model <- auto.arima(x,
                            max.p=7,
                            max.q=7,
                            seasonal=FALSE,
                            stationary=stationary)
        order <- model$arma[c(1, 6, 2)]
    }
    order
}

intervention <- function(x, order, inputs, null.interventions, ao.alpha=NULL) {
    # Arguments:
    # x: time series
    # order: c(p, d, q), the ARIMA(p, d, q) to model
    # inputs: data.frame of intervention inputs of length length(x)
    # null.interventions: list of vectors of null interventions to test against
    # ao.alpha: iteratively remove additive outliers with (1 - ao.alpha)
    #           significance
    #
    # Returns a list with components:
    # model: the ARIMAX model
    # interventions: data frame of fitted interventions
    # null.interventions: the null interventions tested against
    # test.p.values: p-value for each one-sided alternative hypothesis that the
    #                intervention is less than null.intervention (the p-value
    #                for the alternative hypothesis that the intervention is
    #                greater than null.intervention is (1.0 - test.p.value))
    # aos: time indexes of removed additive outliers

    if (is.null(null.interventions)) {
        null.interventions <- numeric()
    }
    reg <- lm(x ~ ., data=inputs)
    standard.errors <- sqrt(diag(vcov(reg)))
    coefs <- coef(reg)
    resid <- residuals(reg)
    if ((all(resid == 0) || all(standard.errors < 1e-3)) && !any(is.na(coefs[-1]))) {
        # there is essentially no noise
        model <- NULL
        df <- NULL
        interventions <- coefs[-1]
        intervention.standard.errors <- standard.errors[-1]
        test.p.values <- vector("list", length(null.interventions))
        for (name in names(null.interventions)) {
            test.p.values[[name]] <- 1.0 * (interventions[name] >= null.interventions[[name]])
        }
        aos <- NULL
    } else {
        aos <- integer()
        model <- NULL
        input.names <- names(inputs)
        repeat {
            e <- try(newModel <- arima(x,
                                       order=order,
                                       xreg=inputs,
                                       method="ML"),
                     silent=TRUE)
            if (!inherits(e, "try-error")) {
                intervention.variances <- diag(newModel$var.coef)[input.names]
                if (!all(intervention.variances >= 0)) {
                    e <- try(stop("Error fitting ARIMAX model"), silent=TRUE)
                }
            }
            if (inherits(e, "try-error")) {
                if (is.null(model)) {
                    stop(e)
                }
                aos <- setdiff(aos, new.aos)
                break
            }
            model <- newModel
            if (!is.null(ao.alpha) && ao.alpha > 0.0) {
                capture.output(new.aos <- detectAO(model, alpha=ao.alpha)$ind)
                new.aos <- setdiff(na.omit(new.aos), aos)
                if (length(new.aos) > 0) {
                    aos <- c(aos, new.aos)
                    x[new.aos] <- NA
                    next
                }
            }
            break
        }
        df <- sum(!is.na(x)) - length(model$coef)
        variances <- diag(model$var.coef)
        standard.errors <- sqrt(variances)
        interventions <- model$coef[input.names]
        intervention.standard.errors <- standard.errors[input.names]
        test.p.values <- vector("list", length(null.interventions))
        for (name in names(null.interventions)) {
            test.t.statistics <- (interventions[[name]] - null.interventions[[name]]) / intervention.standard.errors[[name]]
            test.p.values[[name]] <- pt(test.t.statistics, df)
        }
    }
    list(model=model,
         interventions=interventions,
         null.interventions=null.interventions,
         test.p.values=test.p.values,
         aos=aos)
}

intervention.at.time <- function(x, ts, null.interventions, ao.alpha=NULL, stationary=FALSE) {
    # Arguments:
    # x: time series
    # ts: indexes where unique(sort(floor(ts))) are the indexes where each
    #     level shift intervention begins
    # null.interventions: vector of the null interventions to test max(ts)
    #                     against
    # ao.alpha: iteratively remove additive outliers with (1 - ao.alpha)
    #           significance
    # stationary: if TRUE, assume that time series is stationary
    #
    # Returns a list with components:
    # order: c(p, d, q), the ARIMA(p, d, q) order which was modeled
    # model: the ARIMAX model
    # intervention: the fitted intervention tested
    # interventions: all fitted interventions
    # null.interventions: the null interventions tested against
    # test.p.values: p-values for the one-sided alternative hypothesis that the
    #                intervention is less than null.intervention (the p-value
    #                for the alternative hypothesis that the intervention is
    #                greater than null.intervention is (1.0 - test.p.value))
    # aos: time indexes of removed additive outliers

    if (length(ts) < 1) {
        stop("length(ts) must be > 0")
    }
    ts <- unique(sort(floor(ts)))
    start <- ts[1]
    t <- ts[length(ts)]   
    if (start < 2) {
        stop("min(ts) must be > 1")
    }
    if (t > length(x)) {
        stop("max(ts) must be <= length(x)")
    }
    interventions <- data.frame(row.names=seq(x))
    if (length(ts) > 1) {
        for (i in seq(1, length(ts) - 1, 1)) {
            interventions[[paste("delta", i, sep="")]] <- 1 * (seq(x) >= ts[i] & seq(x) < ts[i + 1])
        }
    }
    interventions[["intervention"]] <- 1 * (seq(x) >= t)
    order <- arima.order(x[seq(1, start - 1, 1)], stationary)
    result <- intervention(x,
                           order,
                           interventions,
                           list(intervention=null.interventions),
                           ao.alpha)
    list(order=order,
         model=result$model,
         intervention=result$interventions[["intervention"]],
         interventions=result$interventions,
         null.interventions=result$null.interventions[["intervention"]],
         test.p.values=result$test.p.values$intervention,
         aos=result$aos)
}
