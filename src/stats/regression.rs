use crate::stats::normality::{check_normality, NormalityCheck};
use crate::stats::utils::{
    chi2_upper_p, correlation_matrix, covariance_matrix, fit_ols, mean, std_dev, t_two_tailed_p,
    OlsCoef, OlsFit,
};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VifEntry {
    pub variable: String,
    pub vif: f64,
    pub r_squared_aux: f64,
    pub severity: String, // "ok" | "warning" (>5) | "fail" (>10)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighCorrPair {
    pub var1: String,
    pub var2: String,
    pub r: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizedModel {
    pub method: String,
    pub lambda: f64,
    pub intercept: f64,
    pub coefficients: Vec<(String, f64)>,
    pub rmse: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MulticollinearityReport {
    pub predictors: Vec<String>,
    pub vifs: Vec<VifEntry>,
    pub max_vif: f64,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub high_corr_pairs: Vec<HighCorrPair>,
    pub ridge_fallback: Option<RegularizedModel>,
    pub lasso_fallback: Option<RegularizedModel>,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationReport {
    pub durbin_watson: f64,
    pub dw_assessment: String,
    pub bg_lags: usize,
    pub bg_lm_stat: f64,
    pub bg_p_value: f64,
    pub acf_lags: Vec<(usize, f64)>,
    pub acf_conf_bound: f64,
    pub newey_west_coefs: Vec<OlsCoef>,
    pub ar1_phi: f64,
    pub ar1_rmse: f64,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogYFallback {
    pub applicable: bool,
    pub reason: Option<String>,
    pub r_squared: Option<f64>,
    pub bp_p_value: Option<f64>,
    pub improved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroscedasticityReport {
    pub bp_lm_stat: f64,
    pub bp_df: usize,
    pub bp_p_value: f64,
    pub white_lm_stat: f64,
    pub white_df: usize,
    pub white_p_value: f64,
    pub hc3_coefs: Vec<OlsCoef>,
    pub log_y_fallback: LogYFallback,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluentialPoint {
    pub row_index: usize,
    pub cooks_d: f64,
    pub leverage: f64,
    pub residual: f64,
    pub mahalanobis_d2: f64,
    pub mahalanobis_p: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    pub removed_count: usize,
    pub orig_r_squared: f64,
    pub clean_r_squared: f64,
    pub orig_rmse: f64,
    pub clean_rmse: f64,
    pub orig_coefs: Vec<OlsCoef>,
    pub clean_coefs: Vec<OlsCoef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutliersReport {
    pub n: usize,
    pub cooks_threshold_4_over_n: f64,
    pub cooks_d: Vec<f64>,
    pub leverage: Vec<f64>,
    pub standardized_residuals: Vec<f64>,
    pub mahalanobis_d2: Vec<f64>,
    pub count_cooks_gt_1: usize,
    pub count_cooks_gt_4_n: usize,
    pub count_mahalanobis_outliers: usize,
    pub top_influential: Vec<InfluentialPoint>,
    pub sensitivity: Option<SensitivityAnalysis>,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonEntry {
    pub model_name: String,
    pub predictors: Vec<String>,
    pub k_params: usize,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub rmse: f64,
    pub mae: f64,
    pub aic: f64,
    pub bic: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionFullReport {
    pub target: String,
    pub predictors: Vec<String>,
    pub n: usize,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub rmse: f64,
    pub mae: f64,
    pub f_stat: f64,
    pub f_p_value: f64,
    pub aic: f64,
    pub bic: f64,
    pub coefficients: Vec<OlsCoef>,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub residual_normality: NormalityCheck,
    pub multicollinearity: Option<MulticollinearityReport>,
    pub autocorrelation: AutocorrelationReport,
    pub heteroscedasticity: HeteroscedasticityReport,
    pub outliers: OutliersReport,
    pub model_comparison: Vec<ModelComparisonEntry>,
}

pub fn check_multicollinearity(
    y: Option<&[f64]>,
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
    lambda: f64,
) -> Result<MulticollinearityReport, String> {
    let p = x_cols.len();
    if p < 2 {
        return Err("Multikollinearitet (VIF) krever minst 2 forklaringsvariabler.".to_string());
    }
    let n = x_cols[0].len();
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..p).map(|j| x_cols[j][i]).collect())
        .collect();
    let corr_mat = correlation_matrix(&rows);

    let mut high_corr_pairs = Vec::new();
    for i in 0..p {
        for j in (i + 1)..p {
            if corr_mat[i][j].abs() > 0.8 {
                high_corr_pairs.push(HighCorrPair {
                    var1: predictor_names[i].clone(),
                    var2: predictor_names[j].clone(),
                    r: corr_mat[i][j],
                });
            }
        }
    }

    let mut vifs = Vec::with_capacity(p);
    let mut max_vif = 1.0_f64;
    for j in 0..p {
        let other_x: Vec<Vec<f64>> = (0..p)
            .filter(|&k| k != j)
            .map(|k| x_cols[k].clone())
            .collect();
        let other_names: Vec<String> = (0..p)
            .filter(|&k| k != j)
            .map(|k| predictor_names[k].clone())
            .collect();
        let r2 = match fit_ols(&x_cols[j], &other_x, &other_names) {
            Ok(fit) => fit.r_squared,
            Err(_) => 0.0,
        };
        let vif = if (1.0 - r2) <= 1e-6 {
            999.9
        } else {
            1.0 / (1.0 - r2)
        };
        max_vif = max_vif.max(vif);
        let severity = if vif > 10.0 {
            "fail"
        } else if vif > 5.0 {
            "warning"
        } else {
            "ok"
        }
        .to_string();
        vifs.push(VifEntry {
            variable: predictor_names[j].clone(),
            vif,
            r_squared_aux: r2,
            severity,
        });
    }

    let (ridge_fallback, lasso_fallback) = if let Some(yv) = y {
        (
            fit_ridge(yv, x_cols, predictor_names, lambda).ok(),
            fit_lasso(yv, x_cols, predictor_names, lambda).ok(),
        )
    } else {
        (None, None)
    };

    let status = if max_vif > 10.0 || !high_corr_pairs.is_empty() {
        "fail"
    } else if max_vif > 5.0 {
        "warning"
    } else {
        "ok"
    }
    .to_string();

    let interpretation = if status == "ok" {
        format!(
            "Ingen bekymringsfull multikollinearitet (maks VIF = {:.2} < 5, ingen |r| > 0.8).",
            max_vif
        )
    } else {
        format!(
            "Multikollinearitet oppdaget! Maks VIF = {:.2}, {} variabelpar har |r| > 0.8.",
            max_vif,
            high_corr_pairs.len()
        )
    };

    let recommendation = if status == "ok" {
        "Alle forklaringsvariabler har akseptabel VIF (< 5).".to_string()
    } else {
        let worst = vifs
            .iter()
            .max_by(|a, b| a.vif.partial_cmp(&b.vif).unwrap_or(std::cmp::Ordering::Equal))
            .map(|v| v.variable.as_str())
            .unwrap_or("variabelen med høyest VIF");
        format!(
            "Vurder å fjerne eller slå sammen '{}' (høyest VIF), eller bruk Ridge-/Lasso-regresjon (beregnet under).",
            worst
        )
    };

    Ok(MulticollinearityReport {
        predictors: predictor_names.to_vec(),
        vifs,
        max_vif,
        correlation_matrix: corr_mat,
        high_corr_pairs,
        ridge_fallback,
        lasso_fallback,
        status,
        interpretation,
        recommendation,
    })
}

/// Standardized Ridge regression (L2 penalty) transformed back to original scale.
pub fn fit_ridge(
    y: &[f64],
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
    lambda: f64,
) -> Result<RegularizedModel, String> {
    let n = y.len();
    let p = x_cols.len();
    let y_mean = mean(y);
    let y_centered: Vec<f64> = y.iter().map(|v| v - y_mean).collect();

    let mut x_means = vec![0.0; p];
    let mut x_stds = vec![1.0; p];
    let mut z = DMatrix::zeros(n, p);
    for j in 0..p {
        x_means[j] = mean(&x_cols[j]);
        let s = std_dev(&x_cols[j]);
        x_stds[j] = if s > 1e-12 { s } else { 1.0 };
        for i in 0..n {
            z[(i, j)] = (x_cols[j][i] - x_means[j]) / x_stds[j];
        }
    }

    let yv = DVector::from_column_slice(&y_centered);
    let ztz = z.transpose() * &z + DMatrix::identity(p, p) * lambda.max(1e-6);
    let beta_z = ztz
        .try_inverse()
        .ok_or_else(|| "Kunne ikke invertere Ridge-matrise.".to_string())?
        * (z.transpose() * &yv);

    let mut orig_coefs = Vec::with_capacity(p);
    let mut intercept = y_mean;
    for j in 0..p {
        let bj = beta_z[j] / x_stds[j];
        intercept -= bj * x_means[j];
        orig_coefs.push((predictor_names[j].clone(), bj));
    }

    let mut rss = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let mut pred = intercept;
        for j in 0..p {
            pred += orig_coefs[j].1 * x_cols[j][i];
        }
        rss += (y[i] - pred).powi(2);
        tss += (y[i] - y_mean).powi(2);
    }
    let rmse = (rss / n as f64).sqrt();
    let r_squared = if tss > 1e-15 {
        (1.0 - rss / tss).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(RegularizedModel {
        method: "Ridge (L2)".to_string(),
        lambda,
        intercept,
        coefficients: orig_coefs,
        rmse,
        r_squared,
    })
}

/// Standardized Lasso regression (L1 penalty via coordinate descent) transformed back to original scale.
pub fn fit_lasso(
    y: &[f64],
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
    lambda: f64,
) -> Result<RegularizedModel, String> {
    let n = y.len();
    let p = x_cols.len();
    let y_mean = mean(y);
    let y_centered: Vec<f64> = y.iter().map(|v| v - y_mean).collect();

    let mut x_means = vec![0.0; p];
    let mut x_stds = vec![1.0; p];
    let mut z: Vec<Vec<f64>> = vec![vec![0.0; n]; p];
    for j in 0..p {
        x_means[j] = mean(&x_cols[j]);
        let s = std_dev(&x_cols[j]);
        x_stds[j] = if s > 1e-12 { s } else { 1.0 };
        for i in 0..n {
            z[j][i] = (x_cols[j][i] - x_means[j]) / x_stds[j];
        }
    }

    let mut beta = vec![0.0; p];
    let nf = n as f64;
    let alpha_l1 = lambda.max(1e-6);

    for _iter in 0..500 {
        let mut max_delta = 0.0_f64;
        for j in 0..p {
            let mut rho = 0.0;
            let mut z_sq = 0.0;
            for i in 0..n {
                let mut pred_excl = 0.0;
                for k in 0..p {
                    if k != j {
                        pred_excl += z[k][i] * beta[k];
                    }
                }
                rho += z[j][i] * (y_centered[i] - pred_excl);
                z_sq += z[j][i] * z[j][i];
            }
            let soft = if rho > alpha_l1 {
                rho - alpha_l1
            } else if rho < -alpha_l1 {
                rho + alpha_l1
            } else {
                0.0
            };
            let new_b = if z_sq > 1e-12 { soft / z_sq } else { 0.0 };
            max_delta = max_delta.max((new_b - beta[j]).abs());
            beta[j] = new_b;
        }
        if max_delta < 1e-7 {
            break;
        }
    }
    let _ = nf;

    let mut orig_coefs = Vec::with_capacity(p);
    let mut intercept = y_mean;
    for j in 0..p {
        let bj = beta[j] / x_stds[j];
        intercept -= bj * x_means[j];
        orig_coefs.push((predictor_names[j].clone(), bj));
    }

    let mut rss = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let mut pred = intercept;
        for j in 0..p {
            pred += orig_coefs[j].1 * x_cols[j][i];
        }
        rss += (y[i] - pred).powi(2);
        tss += (y[i] - y_mean).powi(2);
    }
    let rmse = (rss / n as f64).sqrt();
    let r_squared = if tss > 1e-15 {
        (1.0 - rss / tss).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(RegularizedModel {
        method: "Lasso (L1)".to_string(),
        lambda,
        intercept,
        coefficients: orig_coefs,
        rmse,
        r_squared,
    })
}

/// Autocorrelation: Durbin-Watson, Breusch-Godfrey LM test, ACF, Newey-West HAC SEs, AR(1) fit.
pub fn check_autocorrelation(
    ols: &OlsFit,
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
    lags: usize,
    alpha: f64,
) -> Result<AutocorrelationReport, String> {
    let e = &ols.residuals;
    let n = e.len();
    if n < 6 {
        return Err("Autokorrelasjonssjekk krever minst 6 observasjoner.".to_string());
    }

    // Durbin-Watson
    let ss_e: f64 = e.iter().map(|v| v * v).sum();
    let ss_diff: f64 = e.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let dw = if ss_e > 1e-15 { ss_diff / ss_e } else { 2.0 };
    let dw_assessment = if dw < 1.5 {
        "Positiv autokorrelasjon (DW < 1.5)".to_string()
    } else if dw > 2.5 {
        "Negativ autokorrelasjon (DW > 2.5)".to_string()
    } else {
        "Nær 2.0 — lite tegn til 1. ordens autokorrelasjon".to_string()
    };

    // Breusch-Godfrey LM test (auxiliary regression of e_t on X_t and e_{t-1}...e_{t-p}, zero-padded)
    let p_lags = lags.clamp(1, (n / 4).max(1).min(8));
    let mut aux_x: Vec<Vec<f64>> = x_cols.to_vec();
    let mut aux_names: Vec<String> = predictor_names.to_vec();
    for l in 1..=p_lags {
        let mut lagged = vec![0.0; n];
        for t in l..n {
            lagged[t] = e[t - l];
        }
        aux_x.push(lagged);
        aux_names.push(format!("lag_{}", l));
    }
    let bg_r2 = fit_ols(e, &aux_x, &aux_names).map(|f| f.r_squared).unwrap_or(0.0);
    let bg_lm_stat = (n as f64) * bg_r2;
    let bg_p_value = chi2_upper_p(bg_lm_stat, p_lags as f64);

    // Sample ACF up to max_lag
    let max_acf_lag = (n / 3).min(20).max(1);
    let e_mean = mean(e);
    let denom: f64 = e.iter().map(|v| (v - e_mean).powi(2)).sum();
    let mut acf_lags = Vec::with_capacity(max_acf_lag);
    for lag in 1..=max_acf_lag {
        let mut num = 0.0;
        for t in lag..n {
            num += (e[t] - e_mean) * (e[t - lag] - e_mean);
        }
        let r_k = if denom > 1e-15 { num / denom } else { 0.0 };
        acf_lags.push((lag, r_k));
    }
    let acf_conf_bound = 1.96 / (n as f64).sqrt();

    // Newey-West HAC robust standard errors (Bartlett kernel, automatic lag L = floor(4*(n/100)^(2/9)))
    let nw_lag = ((4.0 * ((n as f64) / 100.0).powf(2.0 / 9.0)).floor() as usize).max(1).min(n / 3);
    let newey_west_coefs = compute_newey_west(ols, nw_lag);

    // Simple AR(1) on residuals: e_t = phi * e_{t-1} + u_t
    let mut num_ar = 0.0;
    let mut den_ar = 0.0;
    for t in 1..n {
        num_ar += e[t] * e[t - 1];
        den_ar += e[t - 1] * e[t - 1];
    }
    let ar1_phi = if den_ar > 1e-15 {
        (num_ar / den_ar).clamp(-0.99, 0.99)
    } else {
        0.0
    };
    let ar1_rss: f64 = (1..n).map(|t| (e[t] - ar1_phi * e[t - 1]).powi(2)).sum();
    let ar1_rmse = (ar1_rss / (n - 1) as f64).sqrt();

    let violated = bg_p_value < alpha || dw < 1.5 || dw > 2.5;
    let status = if bg_p_value < alpha {
        "fail"
    } else if dw < 1.5 || dw > 2.5 {
        "warning"
    } else {
        "ok"
    }
    .to_string();

    let interpretation = if !violated {
        format!(
            "Ingen signifikant autokorrelasjon (DW = {:.3}, Breusch-Godfrey LM({}) = {:.3}, p = {:.4}).",
            dw, p_lags, bg_lm_stat, bg_p_value
        )
    } else {
        format!(
            "Autokorrelasjon påvist! DW = {:.3}, Breusch-Godfrey LM({}) = {:.3} (p = {:.4}), AR(1) φ = {:.3}.",
            dw, p_lags, bg_lm_stat, bg_p_value, ar1_phi
        )
    };

    let recommendation = if !violated {
        "Uavhengige feilledd ser ut til å være oppfylt.".to_string()
    } else {
        format!(
            "Bruk Newey-West HAC robuste standardfeil (beregnet under med lag L={}) eller tidsseriemodell som ARIMA/AR(1) (reduserer residual-RMSE fra {:.4} til {:.4}).",
            nw_lag, ols.rmse, ar1_rmse
        )
    };

    Ok(AutocorrelationReport {
        durbin_watson: dw,
        dw_assessment,
        bg_lags: p_lags,
        bg_lm_stat,
        bg_p_value,
        acf_lags,
        acf_conf_bound,
        newey_west_coefs,
        ar1_phi,
        ar1_rmse,
        status,
        interpretation,
        recommendation,
    })
}

fn compute_newey_west(ols: &OlsFit, max_lag: usize) -> Vec<OlsCoef> {
    let n = ols.n;
    let k = ols.k;
    let x = &ols.x_matrix;
    let e = &ols.residuals;

    // S_0 = sum_t e_t^2 x_t x_t^T
    let mut s_mat = DMatrix::zeros(k, k);
    for t in 0..n {
        let xt = x.row(t).transpose();
        s_mat += (&xt * xt.transpose()) * (e[t] * e[t]);
    }
    for l in 1..=max_lag {
        let w_l = 1.0 - (l as f64) / ((max_lag + 1) as f64);
        let mut gamma_l = DMatrix::zeros(k, k);
        for t in l..n {
            let xt = x.row(t).transpose();
            let xtl = x.row(t - l).transpose();
            let term = (&xt * xtl.transpose()) * (e[t] * e[t - l]);
            gamma_l += &term + term.transpose();
        }
        s_mat += gamma_l * w_l;
    }

    // Small-sample adjustment n / (n - k)
    let adj = (n as f64) / ((n - k).max(1) as f64);
    let cov_nw = &ols.xtx_inv * (s_mat * adj) * &ols.xtx_inv;
    let df = (n - k) as f64;

    (0..k)
        .map(|j| {
            let se = cov_nw[(j, j)].max(0.0).sqrt();
            let t = if se > 1e-15 { ols.beta[j] / se } else { 0.0 };
            OlsCoef {
                name: ols.names[j].clone(),
                estimate: ols.beta[j],
                std_error: se,
                t_stat: t,
                p_value: t_two_tailed_p(t, df),
            }
        })
        .collect()
}

/// Heteroscedasticity: Breusch-Pagan, White's test, HC3 robust SEs, Log(Y) refit.
pub fn check_heteroscedasticity(
    ols: &OlsFit,
    y: &[f64],
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
    alpha: f64,
) -> Result<HeteroscedasticityReport, String> {
    let n = ols.n;
    let p = x_cols.len();
    let e2: Vec<f64> = ols.residuals.iter().map(|r| r * r).collect();

    // Breusch-Pagan (Koenker studentized version: n * R^2 from e^2 ~ X)
    let bp_r2 = fit_ols(&e2, x_cols, predictor_names)
        .map(|f| f.r_squared)
        .unwrap_or(0.0);
    let bp_lm_stat = (n as f64) * bp_r2;
    let bp_df = p.max(1);
    let bp_p_value = chi2_upper_p(bp_lm_stat, bp_df as f64);

    // White's test: e^2 ~ X + X^2 + cross-products (or fitted + fitted^2 if p is large relative to n)
    let mut white_x: Vec<Vec<f64>> = Vec::new();
    let mut white_names: Vec<String> = Vec::new();
    let full_white_terms = p + p + (p * p.saturating_sub(1)) / 2;
    if full_white_terms + 3 < n && p <= 6 {
        for j in 0..p {
            white_x.push(x_cols[j].clone());
            white_names.push(predictor_names[j].clone());
        }
        for j in 0..p {
            white_x.push(x_cols[j].iter().map(|v| v * v).collect());
            white_names.push(format!("{}^2", predictor_names[j]));
        }
        for i in 0..p {
            for j in (i + 1)..p {
                white_x.push(
                    x_cols[i]
                        .iter()
                        .zip(x_cols[j].iter())
                        .map(|(&a, &b)| a * b)
                        .collect(),
                );
                white_names.push(format!("{}*{}", predictor_names[i], predictor_names[j]));
            }
        }
    } else {
        // White's compact test using yhat and yhat^2
        white_x.push(ols.fitted.clone());
        white_names.push("yhat".to_string());
        white_x.push(ols.fitted.iter().map(|v| v * v).collect());
        white_names.push("yhat^2".to_string());
    }
    let white_df = white_x.len().max(1);
    let white_r2 = fit_ols(&e2, &white_x, &white_names)
        .map(|f| f.r_squared)
        .unwrap_or(0.0);
    let white_lm_stat = (n as f64) * white_r2;
    let white_p_value = chi2_upper_p(white_lm_stat, white_df as f64);

    // HC3 robust standard errors: (X^T X)^-1 X^T diag(e_i^2 / (1 - h_ii)^2) X (X^T X)^-1
    let k = ols.k;
    let mut meat = DMatrix::zeros(k, k);
    for i in 0..n {
        let denom = (1.0 - ols.leverage[i]).max(0.05);
        let w = (ols.residuals[i] / denom).powi(2);
        let xi = ols.x_matrix.row(i).transpose();
        meat += (&xi * xi.transpose()) * w;
    }
    let cov_hc3 = &ols.xtx_inv * meat * &ols.xtx_inv;
    let df = (n - k) as f64;
    let hc3_coefs: Vec<OlsCoef> = (0..k)
        .map(|j| {
            let se = cov_hc3[(j, j)].max(0.0).sqrt();
            let t = if se > 1e-15 { ols.beta[j] / se } else { 0.0 };
            OlsCoef {
                name: ols.names[j].clone(),
                estimate: ols.beta[j],
                std_error: se,
                t_stat: t,
                p_value: t_two_tailed_p(t, df),
            }
        })
        .collect();

    // Log-Y fallback if all Y > 0
    let log_y_fallback = if y.iter().all(|&v| v > 0.0) {
        let log_y: Vec<f64> = y.iter().map(|v| v.ln()).collect();
        match fit_ols(&log_y, x_cols, predictor_names) {
            Ok(log_fit) => {
                let le2: Vec<f64> = log_fit.residuals.iter().map(|r| r * r).collect();
                let lbp_r2 = fit_ols(&le2, x_cols, predictor_names)
                    .map(|f| f.r_squared)
                    .unwrap_or(0.0);
                let lbp_p = chi2_upper_p((n as f64) * lbp_r2, bp_df as f64);
                LogYFallback {
                    applicable: true,
                    reason: None,
                    r_squared: Some(log_fit.r_squared),
                    bp_p_value: Some(lbp_p),
                    improved: lbp_p > bp_p_value,
                }
            }
            Err(e) => LogYFallback {
                applicable: false,
                reason: Some(e),
                r_squared: None,
                bp_p_value: None,
                improved: false,
            },
        }
    } else {
        LogYFallback {
            applicable: false,
            reason: Some("Y inneholder verdier <= 0; log-transformasjon krever Y > 0.".to_string()),
            r_squared: None,
            bp_p_value: None,
            improved: false,
        }
    };

    let violated = bp_p_value < alpha || white_p_value < alpha;
    let status = if bp_p_value < alpha && white_p_value < alpha {
        "fail"
    } else if violated {
        "warning"
    } else {
        "ok"
    }
    .to_string();

    let interpretation = if !violated {
        format!(
            "Homoskedastisitet (lik residualvarians) er oppfylt (Breusch-Pagan p = {:.4}, White p = {:.4}).",
            bp_p_value, white_p_value
        )
    } else {
        format!(
            "Heteroskedastisitet oppdaget (Breusch-Pagan p = {:.4}, White p = {:.4}).",
            bp_p_value, white_p_value
        )
    };

    let recommendation = if !violated {
        "Standard OLS-standardfeil er gyldige med hensyn til residualvarians.".to_string()
    } else if log_y_fallback.applicable && log_y_fallback.improved {
        format!(
            "Bruk robuste standardfeil (HC3, beregnet under) eller log-transformasjon av Y (forbedrer BP p-verdi til {:.4}).",
            log_y_fallback.bp_p_value.unwrap_or(0.0)
        )
    } else {
        "Bruk HC3 robuste standardfeil (beregnet under) i stedet for vanlige OLS-standardfeil.".to_string()
    };

    Ok(HeteroscedasticityReport {
        bp_lm_stat,
        bp_df,
        bp_p_value,
        white_lm_stat,
        white_df,
        white_p_value,
        hc3_coefs,
        log_y_fallback,
        status,
        interpretation,
        recommendation,
    })
}

/// Outliers & Influence: Cook's distance, Mahalanobis distance, Leverage, and Sensitivity Analysis.
pub fn check_outliers(
    ols: &OlsFit,
    y: &[f64],
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
) -> Result<OutliersReport, String> {
    let n = ols.n;
    let p = x_cols.len();
    let k = ols.k as f64;
    let mse = (ols.rss / (n - ols.k).max(1) as f64).max(1e-15);
    let cooks_thresh = 4.0 / n as f64;

    let mut cooks_d = Vec::with_capacity(n);
    let mut std_resids = Vec::with_capacity(n);
    for i in 0..n {
        let h = ols.leverage[i].clamp(0.0, 0.999);
        let e_i = ols.residuals[i];
        let d_i = (e_i * e_i / (k * mse)) * (h / (1.0 - h).powi(2));
        let sr = e_i / (mse * (1.0 - h)).max(1e-15).sqrt();
        cooks_d.push(d_i);
        std_resids.push(sr);
    }

    // Mahalanobis distance across predictors X
    let rows_x: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..p).map(|j| x_cols[j][i]).collect())
        .collect();
    let cov = covariance_matrix(&rows_x);
    let cov_inv = cov
        .clone()
        .try_inverse()
        .unwrap_or_else(|| cov.pseudo_inverse(1e-8).unwrap_or_else(|_| DMatrix::identity(p, p)));
    let means_x: Vec<f64> = (0..p).map(|j| mean(&x_cols[j])).collect();

    let mut mahal_d2 = Vec::with_capacity(n);
    let mut mahal_p = Vec::with_capacity(n);
    for i in 0..n {
        let diff = DVector::from_iterator(p, (0..p).map(|j| x_cols[j][i] - means_x[j]));
        let d2 = (diff.transpose() * &cov_inv * &diff)[(0, 0)].max(0.0);
        let pv = chi2_upper_p(d2, p as f64);
        mahal_d2.push(d2);
        mahal_p.push(pv);
    }

    let count_cooks_gt_1 = cooks_d.iter().filter(|&&d| d > 1.0).count();
    let count_cooks_gt_4_n = cooks_d.iter().filter(|&&d| d > cooks_thresh).count();
    let count_mahal = mahal_p.iter().filter(|&&pv| pv < 0.001).count();

    let mut points: Vec<InfluentialPoint> = (0..n)
        .map(|i| InfluentialPoint {
            row_index: i + 1,
            cooks_d: cooks_d[i],
            leverage: ols.leverage[i],
            residual: ols.residuals[i],
            mahalanobis_d2: mahal_d2[i],
            mahalanobis_p: mahal_p[i],
        })
        .collect();
    points.sort_by(|a, b| {
        b.cooks_d
            .partial_cmp(&a.cooks_d)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_influential: Vec<InfluentialPoint> = points.into_iter().take(10).collect();

    // Sensitivity analysis: remove points with Cook's D > 4/n (if enough points remain)
    let sensitivity = if count_cooks_gt_4_n > 0 && (n - count_cooks_gt_4_n) > ols.k + 2 {
        let keep_idx: Vec<usize> = (0..n).filter(|&i| cooks_d[i] <= cooks_thresh).collect();
        let y_clean: Vec<f64> = keep_idx.iter().map(|&i| y[i]).collect();
        let x_clean: Vec<Vec<f64>> = (0..p)
            .map(|j| keep_idx.iter().map(|&i| x_cols[j][i]).collect())
            .collect();
        fit_ols(&y_clean, &x_clean, predictor_names)
            .ok()
            .map(|clean_fit| SensitivityAnalysis {
                removed_count: count_cooks_gt_4_n,
                orig_r_squared: ols.r_squared,
                clean_r_squared: clean_fit.r_squared,
                orig_rmse: ols.rmse,
                clean_rmse: clean_fit.rmse,
                orig_coefs: ols.coefs(),
                clean_coefs: clean_fit.coefs(),
            })
    } else {
        None
    };

    let status = if count_cooks_gt_1 > 0 || count_mahal > 0 {
        "fail"
    } else if count_cooks_gt_4_n > 0 {
        "warning"
    } else {
        "ok"
    }
    .to_string();

    let interpretation = format!(
        "Cook's D > 1: {} obs., Cook's D > 4/n ({:.4}): {} obs., Mahalanobis-uteliggere (p < 0.001): {} obs.",
        count_cooks_gt_1, cooks_thresh, count_cooks_gt_4_n, count_mahal
    );

    let recommendation = if status == "ok" {
        "Ingen sterkt innflytelsesrike uteliggere oppdaget.".to_string()
    } else {
        "Sjekk utpekte observasjoner for tastefeil og vurder sensitivitetsanalysen (med og uten innflytelsesrike observasjoner nedenfor).".to_string()
    };

    Ok(OutliersReport {
        n,
        cooks_threshold_4_over_n: cooks_thresh,
        cooks_d,
        leverage: ols.leverage.clone(),
        standardized_residuals: std_resids,
        mahalanobis_d2: mahal_d2,
        count_cooks_gt_1,
        count_cooks_gt_4_n,
        count_mahalanobis_outliers: count_mahal,
        top_influential,
        sensitivity,
        status,
        interpretation,
        recommendation,
    })
}

/// Full linear regression diagnostic pipeline (Part 1 + Part 2 metrics & AIC/BIC comparison)
pub fn full_linear_regression_diagnostics(
    target: &str,
    y: &[f64],
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
    alpha: f64,
    bg_lags: usize,
    ridge_lambda: f64,
) -> Result<LinearRegressionFullReport, String> {
    let ols = fit_ols(y, x_cols, predictor_names)?;
    let residual_normality = check_normality(&format!("Residualer ({})", target), &ols.residuals, alpha)?;

    let multicollinearity = if x_cols.len() >= 2 {
        Some(check_multicollinearity(
            Some(y),
            x_cols,
            predictor_names,
            ridge_lambda,
        )?)
    } else {
        None
    };

    let autocorrelation = check_autocorrelation(&ols, x_cols, predictor_names, bg_lags, alpha)?;
    let heteroscedasticity = check_heteroscedasticity(&ols, y, x_cols, predictor_names, alpha)?;
    let outliers = check_outliers(&ols, y, x_cols, predictor_names)?;

    // Build AIC/BIC model comparison table:
    // 1. Full model
    // 2. Null model (intercept only)
    // 3. Reduced model dropping least significant predictor (if p >= 2)
    // 4. Reduced model keeping only significant predictors (p < 0.10, if subset differs)
    let mut model_comparison = Vec::new();
    model_comparison.push(ModelComparisonEntry {
        model_name: "Full modell".to_string(),
        predictors: predictor_names.to_vec(),
        k_params: ols.k + 1,
        r_squared: ols.r_squared,
        adj_r_squared: ols.adj_r_squared,
        rmse: ols.rmse,
        mae: ols.mae,
        aic: ols.aic,
        bic: ols.bic,
    });

    // Null model (intercept only)
    if let Ok(null_fit) = fit_ols(y, &[], &[]) {
        model_comparison.push(ModelComparisonEntry {
            model_name: "Nullmodell (kun konstantledd)".to_string(),
            predictors: vec![],
            k_params: null_fit.k + 1,
            r_squared: null_fit.r_squared,
            adj_r_squared: null_fit.adj_r_squared,
            rmse: null_fit.rmse,
            mae: null_fit.mae,
            aic: null_fit.aic,
            bic: null_fit.bic,
        });
    }

    if x_cols.len() >= 2 {
        // Drop predictor with largest p-value
        let worst_idx = (0..x_cols.len())
            .max_by(|&a, &b| {
                ols.p_values[a + 1]
                    .partial_cmp(&ols.p_values[b + 1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let sub_x: Vec<Vec<f64>> = (0..x_cols.len())
            .filter(|&j| j != worst_idx)
            .map(|j| x_cols[j].clone())
            .collect();
        let sub_names: Vec<String> = (0..x_cols.len())
            .filter(|&j| j != worst_idx)
            .map(|j| predictor_names[j].clone())
            .collect();
        if let Ok(red_fit) = fit_ols(y, &sub_x, &sub_names) {
            model_comparison.push(ModelComparisonEntry {
                model_name: format!("Redusert (uten {})", predictor_names[worst_idx]),
                predictors: sub_names,
                k_params: red_fit.k + 1,
                r_squared: red_fit.r_squared,
                adj_r_squared: red_fit.adj_r_squared,
                rmse: red_fit.rmse,
                mae: red_fit.mae,
                aic: red_fit.aic,
                bic: red_fit.bic,
            });
        }

        // Only significant predictors (p < 0.05)
        let sig_indices: Vec<usize> = (0..x_cols.len())
            .filter(|&j| ols.p_values[j + 1] < 0.05)
            .collect();
        if !sig_indices.is_empty() && sig_indices.len() < x_cols.len() - 1 {
            let sig_x: Vec<Vec<f64>> = sig_indices.iter().map(|&j| x_cols[j].clone()).collect();
            let sig_names: Vec<String> = sig_indices
                .iter()
                .map(|&j| predictor_names[j].clone())
                .collect();
            if let Ok(sig_fit) = fit_ols(y, &sig_x, &sig_names) {
                model_comparison.push(ModelComparisonEntry {
                    model_name: "Kun signifikante (p < 0.05)".to_string(),
                    predictors: sig_names,
                    k_params: sig_fit.k + 1,
                    r_squared: sig_fit.r_squared,
                    adj_r_squared: sig_fit.adj_r_squared,
                    rmse: sig_fit.rmse,
                    mae: sig_fit.mae,
                    aic: sig_fit.aic,
                    bic: sig_fit.bic,
                });
            }
        }
    }

    // Sort comparison by AIC ascending
    model_comparison.sort_by(|a, b| a.aic.partial_cmp(&b.aic).unwrap_or(std::cmp::Ordering::Equal));

    Ok(LinearRegressionFullReport {
        target: target.to_string(),
        predictors: predictor_names.to_vec(),
        n: ols.n,
        r_squared: ols.r_squared,
        adj_r_squared: ols.adj_r_squared,
        rmse: ols.rmse,
        mae: ols.mae,
        f_stat: ols.f_stat,
        f_p_value: ols.f_p_value,
        aic: ols.aic,
        bic: ols.bic,
        coefficients: ols.coefs(),
        fitted: ols.fitted,
        residuals: ols.residuals,
        residual_normality,
        multicollinearity,
        autocorrelation,
        heteroscedasticity,
        outliers,
        model_comparison,
    })
}
