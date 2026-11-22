use crate::stats::utils::{chi2_upper_p, mean, normal_two_tailed_p, pearson_corr};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxCoef {
    pub name: String,
    pub coef: f64,
    pub hazard_ratio: f64,
    pub std_error: f64,
    pub z_stat: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchoenfeldCovariateTest {
    pub variable: String,
    pub rho_time: f64,
    pub chi2_stat: f64,
    pub p_value: f64,
    pub ph_met: bool,
    pub hr_early_t25: f64,
    pub hr_late_t75: f64,
    pub points: Vec<(f64, f64)>, // (event_time, scaled_schoenfeld + beta)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxSurvivalReport {
    pub time_col: String,
    pub event_col: String,
    pub predictors: Vec<String>,
    pub n: usize,
    pub n_events: usize,
    pub coefficients: Vec<CoxCoef>,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub concordance_c: f64,
    pub schoenfeld_global_chi2: f64,
    pub schoenfeld_global_df: usize,
    pub schoenfeld_global_p: f64,
    pub ph_assumption_met: bool,
    pub schoenfeld_tests: Vec<SchoenfeldCovariateTest>,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

/// Cox Proportional Hazards model (Breslow partial likelihood via Newton-Raphson)
/// + Grambsch-Therneau scaled Schoenfeld residual test for Proportional Hazards assumption.
pub fn analyze_cox_ph(
    time_col: &str,
    event_col: &str,
    times: &[f64],
    events: &[f64],
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
    alpha: f64,
) -> Result<CoxSurvivalReport, String> {
    let n = times.len();
    let p = x_cols.len();
    if p == 0 {
        return Err("Cox-regresjon krever minst 1 kovariat.".to_string());
    }
    if n < p + 5 {
        return Err(format!("For få observasjoner (n={}) for Cox-regresjon.", n));
    }

    let n_events = events.iter().filter(|&&e| e > 0.5).count();
    if n_events < 4 {
        return Err(format!(
            "For få hendelser (n_events={}) for Cox proporsjonal hasard-analyse.",
            n_events
        ));
    }

    // Standardize predictors internally for stable partial likelihood optimization
    let mut x_means = vec![0.0; p];
    let mut x_stds = vec![1.0; p];
    let mut z = DMatrix::zeros(n, p);
    for j in 0..p {
        let m = mean(&x_cols[j]);
        let v = x_cols[j].iter().map(|val| (val - m).powi(2)).sum::<f64>() / (n - 1) as f64;
        let s = v.sqrt();
        x_means[j] = m;
        x_stds[j] = if s > 1e-12 { s } else { 1.0 };
        for i in 0..n {
            z[(i, j)] = (x_cols[j][i] - m) / x_stds[j];
        }
    }

    // Sort indices by time ascending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Helper: evaluate Breslow partial log-likelihood, gradient, and negative Hessian (information matrix)
    let eval_partial = |beta: &DVector<f64>| -> (f64, DVector<f64>, DMatrix<f64>) {
        let mut ll = 0.0;
        let mut grad = DVector::zeros(p);
        let mut info = DMatrix::zeros(p, p);

        // Compute linear predictors & exp(eta)
        let eta: Vec<f64> = (0..n)
            .map(|i| {
                let mut s = 0.0;
                for j in 0..p {
                    s += z[(i, j)] * beta[j];
                }
                s.clamp(-30.0, 30.0)
            })
            .collect();
        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        // Traverse unique event times
        let mut idx = 0;
        while idx < n {
            let t_curr = times[order[idx]];
            let mut end = idx;
            while end < n && (times[order[end]] - t_curr).abs() <= 1e-12 {
                end += 1;
            }
            // Risk set: all k in order[idx..n] (since sorted ascending, times >= t_curr)
            let mut d_events = 0usize;
            let mut sum_x_ev = DVector::zeros(p);
            let mut sum_eta_ev = 0.0;
            for k in idx..end {
                let i = order[k];
                if events[i] > 0.5 {
                    d_events += 1;
                    sum_eta_ev += eta[i];
                    for j in 0..p {
                        sum_x_ev[j] += z[(i, j)];
                    }
                }
            }

            if d_events > 0 {
                let mut s0 = 0.0;
                let mut s1 = DVector::zeros(p);
                let mut s2 = DMatrix::zeros(p, p);
                for k in idx..n {
                    let i = order[k];
                    let w = exp_eta[i];
                    s0 += w;
                    let zi = z.row(i).transpose();
                    s1 += &zi * w;
                    s2 += (&zi * zi.transpose()) * w;
                }
                let d_f = d_events as f64;
                ll += sum_eta_ev - d_f * s0.max(1e-30).ln();
                let x_bar = &s1 / s0;
                grad += sum_x_ev - &x_bar * d_f;
                let v_mat = (&s2 / s0) - (&x_bar * x_bar.transpose());
                info += v_mat * d_f;
            }
            idx = end;
        }
        (ll, grad, info)
    };

    let zero_beta = DVector::zeros(p);
    let (null_ll, _, _) = eval_partial(&zero_beta);

    let mut beta_s = DVector::zeros(p);
    for _iter in 0..40 {
        let (_ll, mut grad, mut info) = eval_partial(&beta_s);
        // Tiny ridge for stability
        for j in 0..p {
            grad[j] -= 1e-5 * beta_s[j];
            info[(j, j)] += 1e-5;
        }
        let info_inv = info
            .clone()
            .try_inverse()
            .unwrap_or_else(|| info.pseudo_inverse(1e-8).unwrap_or_else(|_| DMatrix::identity(p, p)));
        let step = &info_inv * grad;
        let max_step = step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let damp = if max_step > 2.0 { 2.0 / max_step } else { 1.0 };
        beta_s += step * damp;
        if max_step < 1e-7 {
            break;
        }
    }

    let (ll, _, mut info_s) = eval_partial(&beta_s);
    for j in 0..p {
        info_s[(j, j)] += 1e-8;
    }
    let cov_s = info_s
        .clone()
        .try_inverse()
        .unwrap_or_else(|| info_s.pseudo_inverse(1e-8).unwrap_or_else(|_| DMatrix::identity(p, p)));

    // Convert beta & covariance to original scale
    let mut beta_orig = DVector::zeros(p);
    let mut std_errors = vec![0.0; p];
    let mut coefficients = Vec::with_capacity(p);
    for j in 0..p {
        beta_orig[j] = beta_s[j] / x_stds[j];
        std_errors[j] = (cov_s[(j, j)].max(0.0).sqrt()) / x_stds[j];
        let z_stat = if std_errors[j] > 1e-15 {
            beta_orig[j] / std_errors[j]
        } else {
            0.0
        };
        coefficients.push(CoxCoef {
            name: predictor_names[j].clone(),
            coef: beta_orig[j],
            hazard_ratio: beta_orig[j].clamp(-30.0, 30.0).exp(),
            std_error: std_errors[j],
            z_stat,
            p_value: normal_two_tailed_p(z_stat),
        });
    }

    // Harrell's C-index (concordance)
    let risk_scores: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| z[(i, j)] * beta_s[j]).sum::<f64>())
        .collect();
    let mut concordant = 0.0;
    let mut comparable = 0.0;
    for i in 0..n {
        if events[i] <= 0.5 {
            continue;
        }
        for j in 0..n {
            if times[j] > times[i] {
                comparable += 1.0;
                if risk_scores[i] > risk_scores[j] {
                    concordant += 1.0;
                } else if (risk_scores[i] - risk_scores[j]).abs() <= 1e-12 {
                    concordant += 0.5;
                }
            }
        }
    }
    let concordance_c = if comparable > 0.0 {
        concordant / comparable
    } else {
        0.5
    };

    // Schoenfeld residuals at each event (Grambsch & Therneau 1994)
    let exp_eta: Vec<f64> = risk_scores.iter().map(|&e| e.clamp(-30.0, 30.0).exp()).collect();
    let mut event_times: Vec<f64> = Vec::new();
    let mut unscaled_resids: Vec<DVector<f64>> = Vec::new();

    let mut idx = 0;
    while idx < n {
        let t_curr = times[order[idx]];
        let mut end = idx;
        while end < n && (times[order[end]] - t_curr).abs() <= 1e-12 {
            end += 1;
        }
        // Risk set weighted mean x_bar(t)
        let mut s0 = 0.0;
        let mut s1 = DVector::zeros(p);
        for k in idx..n {
            let i = order[k];
            let w = exp_eta[i];
            s0 += w;
            s1 += z.row(i).transpose() * w;
        }
        let x_bar = &s1 / s0.max(1e-30);

        for k in idx..end {
            let i = order[k];
            if events[i] > 0.5 {
                let r_i = z.row(i).transpose() - &x_bar;
                event_times.push(times[i]);
                unscaled_resids.push(r_i);
            }
        }
        idx = end;
    }

    let d = event_times.len() as f64;
    // Scaled Schoenfeld residuals (standardized space): r*_i = d * cov_s * r_i
    let scaled_s: Vec<DVector<f64>> = unscaled_resids
        .iter()
        .map(|r| (&cov_s * r) * d)
        .collect();

    // Time transform g(t) = rank(t) / d (Kaplan-Meier / rank-time transform, standard in survival::cox.zph)
    let g_vals: Vec<f64> = (0..event_times.len())
        .map(|i| (i as f64 + 1.0) / d)
        .collect();
    let g_mean = mean(&g_vals);
    let g_centered: Vec<f64> = g_vals.iter().map(|&g| g - g_mean).collect();
    let sum_g2: f64 = g_centered.iter().map(|v| v * v).sum::<f64>().max(1e-12);

    // Grambsch-Therneau global test: T = (sum g_c_i r_i)^T * (d * cov_s / sum_g2) * (sum g_c_i r_i)
    let mut sum_gr = DVector::zeros(p);
    for (i, r) in unscaled_resids.iter().enumerate() {
        sum_gr += r * g_centered[i];
    }
    let global_chi2 = ((sum_gr.transpose() * (&cov_s * (d / sum_g2)) * &sum_gr)[(0, 0)]).max(0.0);
    let global_p = chi2_upper_p(global_chi2, p as f64);

    let mut schoenfeld_tests = Vec::with_capacity(p);
    for j in 0..p {
        // Per-covariate Grambsch-Therneau chi2 = (sum g_c_i * r*_ij)^2 / (d * cov_s_jj * sum_g2)
        // Equivalently using scaled_s:
        let s_col: Vec<f64> = scaled_s.iter().map(|v| v[j]).collect();
        let num_j: f64 = g_centered
            .iter()
            .zip(s_col.iter())
            .map(|(&gc, &rs)| gc * rs)
            .sum();
        let denom_j = (d * cov_s[(j, j)] * sum_g2).max(1e-12);
        let chi2_j = (num_j * num_j / denom_j).max(0.0);
        let p_j = chi2_upper_p(chi2_j, 1.0);
        let rho_j = pearson_corr(&g_vals, &s_col);

        // Points on original coefficient scale: beta_orig[j] + scaled_s[i][j] / x_stds[j]
        let pts: Vec<(f64, f64)> = event_times
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, beta_orig[j] + scaled_s[i][j] / x_stds[j]))
            .collect();

        // Time-varying beta(t) = a + b * ln(t) fallback for early (25%) vs late (75%) hazard ratio
        let log_ts: Vec<f64> = pts.iter().map(|&(t, _)| t.max(1e-6).ln()).collect();
        let betas_t: Vec<f64> = pts.iter().map(|&(_, b)| b).collect();
        let m_lt = mean(&log_ts);
        let m_bt = mean(&betas_t);
        let mut s_num = 0.0;
        let mut s_den = 0.0;
        for (&lt, &bt) in log_ts.iter().zip(betas_t.iter()) {
            s_num += (lt - m_lt) * (bt - m_bt);
            s_den += (lt - m_lt).powi(2);
        }
        let slope_lt = if s_den > 1e-12 { s_num / s_den } else { 0.0 };
        let inter_lt = m_bt - slope_lt * m_lt;
        let t25 = log_ts[log_ts.len() / 4];
        let t75 = log_ts[(log_ts.len() * 3 / 4).min(log_ts.len() - 1)];
        let hr_early_t25 = (inter_lt + slope_lt * t25).clamp(-20.0, 20.0).exp();
        let hr_late_t75 = (inter_lt + slope_lt * t75).clamp(-20.0, 20.0).exp();

        schoenfeld_tests.push(SchoenfeldCovariateTest {
            variable: predictor_names[j].clone(),
            rho_time: rho_j,
            chi2_stat: chi2_j,
            p_value: p_j,
            ph_met: p_j >= alpha,
            hr_early_t25,
            hr_late_t75,
            points: pts,
        });
    }

    let aic = 2.0 * p as f64 - 2.0 * ll;
    let bic = (n_events as f64).ln() * p as f64 - 2.0 * ll;
    let ph_met = global_p >= alpha && schoenfeld_tests.iter().all(|t| t.ph_met);
    let status = if ph_met {
        "ok"
    } else if global_p >= alpha {
        "warning"
    } else {
        "fail"
    }
    .to_string();

    let interpretation = if ph_met {
        if crate::i18n::is_en() {
            format!(
                "Proportional hazards assumption is met (Schoenfeld global χ²({}) = {:.3}, p = {:.4} > {:.2}). Concordance (C-index) = {:.3}.",
                p, global_chi2, global_p, alpha, concordance_c
            )
        } else {
            format!(
                "Proporsjonal hasard-forutsetningen er oppfylt (Schoenfeld global χ²({}) = {:.3}, p = {:.4} > {:.2}). Konkordans (C-index) = {:.3}.",
                p, global_chi2, global_p, alpha, concordance_c
            )
        }
    } else if crate::i18n::is_en() {
        format!(
            "Proportional hazards assumption appears violated (Schoenfeld global χ²({}) = {:.3}, p = {:.4}). Hazard ratio varies over time.",
            p, global_chi2, global_p
        )
    } else {
        format!(
            "Proporsjonal hasard-forutsetningen ser ut til å være brutt (Schoenfeld global χ²({}) = {:.3}, p = {:.4}). Risikoforholdet endrer seg over tid.",
            p, global_chi2, global_p
        )
    };

    let recommendation = if ph_met {
        crate::i18n::tr(
            "Standard Cox proporsjonal hasard-modell er gyldig — risikoforholdet er tilnærmet konstant over tid.",
            "Standard Cox proportional hazards model is valid — hazard ratio is approximately constant over time.",
        )
        .to_string()
    } else {
        crate::i18n::tr(
            "PH-forutsetningen er brutt: tidsavhengig hazard-ratio (tidlig t25 vs. sen t75 basert på beta(t) ~ ln(tid)) er beregnet per kovariat i tabellen under.",
            "PH assumption violated: time-varying hazard ratio (early t25 vs. late t75 based on beta(t) ~ ln(time)) is computed per covariate in the table below.",
        )
        .to_string()
    };

    Ok(CoxSurvivalReport {
        time_col: time_col.to_string(),
        event_col: event_col.to_string(),
        predictors: predictor_names.to_vec(),
        n,
        n_events,
        coefficients,
        log_likelihood: ll,
        null_log_likelihood: null_ll,
        aic,
        bic,
        concordance_c,
        schoenfeld_global_chi2: global_chi2,
        schoenfeld_global_df: p,
        schoenfeld_global_p: global_p,
        ph_assumption_met: ph_met,
        schoenfeld_tests,
        status,
        interpretation,
        recommendation,
    })
}
