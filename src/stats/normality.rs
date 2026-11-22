use crate::stats::utils::{
    chi2_upper_p, excess_kurtosis, mean, normal_cdf, normal_inv_cdf, normal_two_tailed_p, quantile,
    rank_average, skewness, std_dev,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapiroWilkResult {
    pub w: f64,
    pub p_value: f64,
    pub n: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KsTestResult {
    pub d_stat: f64,
    pub p_value_asymptotic: f64,
    pub p_value_lilliefors: f64,
    pub n: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityCheck {
    pub variable: String,
    pub n: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub excess_kurtosis: f64,
    pub recommended_test: String,
    pub shapiro_wilk: Option<ShapiroWilkResult>,
    pub ks_test: KsTestResult,
    pub primary_statistic: f64,
    pub primary_p_value: f64,
    pub is_normal: bool,
    pub status: String, // "ok" | "warning" | "fail"
    pub interpretation: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MannWhitneyResult {
    pub group1: String,
    pub group2: String,
    pub n1: usize,
    pub n2: usize,
    pub median1: f64,
    pub median2: f64,
    pub u_stat: f64,
    pub z_stat: f64,
    pub p_value: f64,
    pub significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KruskalWallisResult {
    pub groups: Vec<String>,
    pub h_stat: f64,
    pub df: usize,
    pub p_value: f64,
    pub significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub target: String,
    pub resamples: usize,
    pub observed_mean: f64,
    pub bootstrap_se: f64,
    pub ci_low_95: f64,
    pub ci_high_95: f64,
}

fn poly_eval(coeffs: &[f64], x: f64) -> f64 {
    // coeffs from lowest to highest power
    let mut res = 0.0;
    for &c in coeffs.iter().rev() {
        res = res * x + c;
    }
    res
}

/// Shapiro-Wilk normality test using Royston's (1992/1995) AS R94 algorithm.
/// Valid for 3 <= n <= 5000.
pub fn shapiro_wilk(data: &[f64]) -> Result<ShapiroWilkResult, String> {
    let n = data.len();
    if n < 3 {
        return Err("Shapiro-Wilk krever minst 3 observasjoner.".to_string());
    }
    if n > 5000 {
        return Err("Shapiro-Wilk er kun definert for n <= 5000 (bruk Kolmogorov-Smirnov).".to_string());
    }

    let mut x = data.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let range = x[n - 1] - x[0];
    if range <= 1e-15 {
        return Ok(ShapiroWilkResult {
            w: 1.0,
            p_value: 1.0,
            n,
        });
    }

    let nn2 = n / 2;
    let nf = n as f64;
    let mut m = vec![0.0; n];
    for i in 0..n {
        let p = ((i + 1) as f64 - 0.375) / (nf + 0.25);
        m[i] = normal_inv_cdf(p);
    }
    let m_sum_sq: f64 = m.iter().map(|v| v * v).sum();

    let mut a = vec![0.0; n];
    let u = 1.0 / nf.sqrt();

    if n == 3 {
        let sqrth = std::f64::consts::FRAC_1_SQRT_2;
        a[0] = -sqrth;
        a[2] = sqrth;
    } else {
        let c1 = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
        let c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];

        let an = m[n - 1] / m_sum_sq.sqrt() + poly_eval(&c1, u);
        a[n - 1] = an;
        a[0] = -an;

        if n <= 5 {
            let phi = (m_sum_sq - 2.0 * m[n - 1] * m[n - 1]) / (1.0 - 2.0 * an * an);
            let phi_sqrt = phi.sqrt();
            for i in 1..(n - 1) {
                a[i] = m[i] / phi_sqrt;
            }
        } else {
            let an_1 = m[n - 2] / m_sum_sq.sqrt() + poly_eval(&c2, u);
            a[n - 2] = an_1;
            a[1] = -an_1;
            let phi = (m_sum_sq - 2.0 * m[n - 1].powi(2) - 2.0 * m[n - 2].powi(2))
                / (1.0 - 2.0 * an.powi(2) - 2.0 * an_1.powi(2));
            let phi_sqrt = phi.max(1e-15).sqrt();
            for i in 2..(n - 2) {
                a[i] = m[i] / phi_sqrt;
            }
        }
    }

    // Compute W = (sum a_i x_i)^2 / sum (x_i - x_bar)^2
    let x_bar = mean(&x);
    let ss: f64 = x.iter().map(|&xi| (xi - x_bar).powi(2)).sum();
    let mut num = 0.0;
    for i in 0..nn2 {
        num += a[n - 1 - i] * (x[n - 1 - i] - x[i]);
    }
    let w = ((num * num) / ss).clamp(0.0, 1.0);

    // Royston p-value transformation
    let p_value = if n == 3 {
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = (0.75_f64).sqrt().asin();
        let pw = pi6 * (w.sqrt().asin() - stqr);
        pw.clamp(0.0, 1.0)
    } else if n <= 11 {
        let c3 = [0.5440, -0.39978, 0.025054, -0.0006714];
        let c4 = [1.3822, -0.77857, 0.062767, -0.0020322];
        let y = (1.0 - w).max(1e-15).ln();
        let gamma_r = -2.273 + 0.459 * nf;
        let m_r = poly_eval(&c3, nf);
        let s_r = poly_eval(&c4, nf).exp();
        if y >= gamma_r {
            1e-12
        } else {
            let y_trans = -(gamma_r - y).ln();
            let z95 = (y_trans - m_r) / s_r;
            (1.0 - normal_cdf(z95)).clamp(0.0, 1.0)
        }
    } else {
        // 12 <= n <= 5000 (exact R swilk.c / Royston 1995)
        let ln_n = nf.ln();
        let c5 = [-1.5861, -0.31082, -0.083751, 0.0038915];
        let c6 = [-0.4803, -0.082676, 0.0030302];
        let mu = poly_eval(&c5, ln_n);
        let sigma = poly_eval(&c6, ln_n).exp();
        let y = (1.0 - w).max(1e-15).ln();
        let z = (y - mu) / sigma;
        (1.0 - normal_cdf(z)).clamp(0.0, 1.0)
    };

    Ok(ShapiroWilkResult { w, p_value, n })
}

/// Kolmogorov-Smirnov test against fitted Normal(mean, sd) with both asymptotic KS p-value
/// and Lilliefors-corrected p-value (Dallal-Wilkinson / Molin-Abdi approximation).
pub fn kolmogorov_smirnov_normal(data: &[f64]) -> Result<KsTestResult, String> {
    let n = data.len();
    if n < 3 {
        return Err("Kolmogorov-Smirnov krever minst 3 observasjoner.".to_string());
    }
    let m = mean(data);
    let s = std_dev(data);
    if s <= 1e-15 {
        return Ok(KsTestResult {
            d_stat: 1.0,
            p_value_asymptotic: 0.0,
            p_value_lilliefors: 0.0,
            n,
        });
    }

    let mut x = data.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let nf = n as f64;
    let mut d_max = 0.0_f64;
    for (i, &xi) in x.iter().enumerate() {
        let f_theo = normal_cdf((xi - m) / s);
        let f_emp_hi = (i + 1) as f64 / nf;
        let f_emp_lo = i as f64 / nf;
        let d1 = (f_emp_hi - f_theo).abs();
        let d2 = (f_theo - f_emp_lo).abs();
        d_max = d_max.max(d1).max(d2);
    }

    // Standard asymptotic KS distribution (Stephens/Marsaglia series)
    let lambda = (nf.sqrt() + 0.12 + 0.11 / nf.sqrt()) * d_max;
    let mut p_asym = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * (k as f64 * lambda).powi(2)).exp();
        p_asym += term;
        if term.abs() < 1e-12 {
            break;
        }
    }
    let p_value_asymptotic = (2.0 * p_asym).clamp(0.0, 1.0);

    // Lilliefors p-value approximation (Dallal & Wilkinson 1986, used in R nortest::lillie.test)
    let p_value_lilliefors = lilliefors_p_value(d_max, n);

    Ok(KsTestResult {
        d_stat: d_max,
        p_value_asymptotic,
        p_value_lilliefors,
        n,
    })
}

fn lilliefors_p_value(d: f64, n: usize) -> f64 {
    let nf = n as f64;
    let d_adj = if n > 100 {
        d * (nf / 100.0).powf(0.49)
    } else {
        d
    };
    let n_eff = nf.min(100.0);
    let p = (-7.01256 * d_adj * d_adj * (n_eff + 2.78019)
        + 2.99587 * d_adj * (n_eff + 2.78019).sqrt()
        - 0.122119
        + 0.974598 / n_eff.sqrt()
        + 1.67997 / n_eff)
        .exp();
    // For very large p (>0.1) Dallal-Wilkinson formula can exceed 1 or underestimate; blend smoothly
    if d_adj < 0.01 {
        return 1.0;
    }
    if p > 0.1 {
        // Stephens (1974) upper tail adjustment for Lilliefors
        let z = d * (nf.sqrt() - 0.01 + 0.85 / nf.sqrt());
        if z < 0.6 {
            return 0.99;
        }
        let p2 = (-1.2337141 / (z * z) - 2.0 * z * z).exp();
        let _ = p2;
        return p.clamp(0.1, 1.0);
    }
    p.clamp(0.0, 1.0)
}

pub fn check_normality(variable: &str, data: &[f64], alpha: f64) -> Result<NormalityCheck, String> {
    let n = data.len();
    if n < 3 {
        return Err(format!("'{}' har for få observasjoner (n={}).", variable, n));
    }
    let m = mean(data);
    let s = std_dev(data);
    let skew = skewness(data);
    let kurt = excess_kurtosis(data);
    let sw = if n <= 5000 {
        shapiro_wilk(data).ok()
    } else {
        None
    };
    let ks = kolmogorov_smirnov_normal(data)?;

    let (recommended_test, primary_statistic, primary_p_value) = if n < 50 {
        if let Some(ref sw_res) = sw {
            ("Shapiro-Wilk (n < 50)".to_string(), sw_res.w, sw_res.p_value)
        } else {
            (
                "Kolmogorov-Smirnov / Lilliefors".to_string(),
                ks.d_stat,
                ks.p_value_lilliefors,
            )
        }
    } else {
        (
            "Kolmogorov-Smirnov / Lilliefors (n >= 50)".to_string(),
            ks.d_stat,
            ks.p_value_lilliefors,
        )
    };

    let is_normal = primary_p_value >= alpha;
    let status = if is_normal && skew.abs() < 1.0 && kurt.abs() < 2.0 {
        "ok"
    } else if is_normal || (skew.abs() < 1.5 && kurt.abs() < 3.0) {
        "warning"
    } else {
        "fail"
    }
    .to_string();

    let interpretation = if is_normal {
        if crate::i18n::is_en() {
            format!(
                "No significant deviation from normality ({}, p = {:.4} >= {:.2}). Skewness = {:.3}, excess kurtosis = {:.3}.",
                recommended_test, primary_p_value, alpha, skew, kurt
            )
        } else {
            format!(
                "Ingen signifikant avvik fra normalfordeling ({}, p = {:.4} >= {:.2}). Skjevhet = {:.3}, eksess-kurtose = {:.3}.",
                recommended_test, primary_p_value, alpha, skew, kurt
            )
        }
    } else if crate::i18n::is_en() {
        format!(
            "Significant deviation from normality ({}, p = {:.4} < {:.2}). Skewness = {:.3}, excess kurtosis = {:.3}.",
            recommended_test, primary_p_value, alpha, skew, kurt
        )
    } else {
        format!(
            "Signifikant avvik fra normalfordeling ({}, p = {:.4} < {:.2}). Skjevhet = {:.3}, eksess-kurtose = {:.3}.",
            recommended_test, primary_p_value, alpha, skew, kurt
        )
    };

    let recommendation = if is_normal {
        crate::i18n::tr(
            "Parametriske metoder (t-test, ANOVA, OLS) er egnet med hensyn til normalitet.",
            "Parametric methods (t-test, ANOVA, OLS) are appropriate with respect to normality.",
        )
        .to_string()
    } else {
        crate::i18n::tr(
            "Normalitetsforutsetningen er brutt. Bruk ikke-parametrisk test (Mann-Whitney U / Kruskal-Wallis) eller bootstrapping (beregnet nedenfor).",
            "Normality assumption is violated. Use non-parametric tests (Mann-Whitney U / Kruskal-Wallis) or bootstrapping (computed below).",
        )
        .to_string()
    };

    Ok(NormalityCheck {
        variable: variable.to_string(),
        n,
        mean: m,
        std_dev: s,
        skewness: skew,
        excess_kurtosis: kurt,
        recommended_test,
        shapiro_wilk: sw,
        ks_test: ks,
        primary_statistic,
        primary_p_value,
        is_normal,
        status,
        interpretation,
        recommendation,
    })
}

/// Mann-Whitney U test (two independent samples, two-tailed normal approximation with tie correction)
pub fn mann_whitney_u(
    g1_name: &str,
    g1: &[f64],
    g2_name: &str,
    g2: &[f64],
    alpha: f64,
) -> Result<MannWhitneyResult, String> {
    let n1 = g1.len();
    let n2 = g2.len();
    if n1 == 0 || n2 == 0 {
        return Err("Begge grupper må ha minst 1 observasjon for Mann-Whitney U.".to_string());
    }

    let mut combined = Vec::with_capacity(n1 + n2);
    combined.extend_from_slice(g1);
    combined.extend_from_slice(g2);

    let (ranks, tie_sum) = rank_average(&combined);
    let r1: f64 = ranks[..n1].iter().sum();
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let n_tot = (n1 + n2) as f64;

    let u1 = r1 - n1f * (n1f + 1.0) / 2.0;
    let u2 = n1f * n2f - u1;
    let u_stat = u1.min(u2);

    let mu_u = n1f * n2f / 2.0;
    let tie_corr = if n_tot > 1.0 {
        tie_sum / (n_tot * (n_tot - 1.0))
    } else {
        0.0
    };
    let sigma_u = ((n1f * n2f / 12.0) * ((n_tot + 1.0) - tie_corr)).max(1e-15).sqrt();
    // Continuity correction
    let z = ((u_stat - mu_u).abs() - 0.5).max(0.0) / sigma_u;
    let p_value = normal_two_tailed_p(z);

    Ok(MannWhitneyResult {
        group1: g1_name.to_string(),
        group2: g2_name.to_string(),
        n1,
        n2,
        median1: quantile(g1, 0.5),
        median2: quantile(g2, 0.5),
        u_stat,
        z_stat: z,
        p_value,
        significant: p_value < alpha,
    })
}

/// Kruskal-Wallis H test for k >= 2 independent groups (with tie correction)
pub fn kruskal_wallis(
    groups: &[(String, Vec<f64>)],
    alpha: f64,
) -> Result<KruskalWallisResult, String> {
    if groups.len() < 2 {
        return Err("Kruskal-Wallis krever minst 2 grupper.".to_string());
    }
    let mut combined = Vec::new();
    let mut offsets = Vec::new();
    for (_, vals) in groups {
        if vals.is_empty() {
            return Err("Ingen gruppe kan være tom i Kruskal-Wallis.".to_string());
        }
        offsets.push((combined.len(), vals.len()));
        combined.extend_from_slice(vals);
    }

    let n_tot = combined.len() as f64;
    let (ranks, tie_sum) = rank_average(&combined);

    let mut sum_term = 0.0;
    for &(start, len) in &offsets {
        let r_sum: f64 = ranks[start..start + len].iter().sum();
        sum_term += (r_sum * r_sum) / len as f64;
    }

    let h_raw = (12.0 / (n_tot * (n_tot + 1.0))) * sum_term - 3.0 * (n_tot + 1.0);
    let denom = 1.0 - tie_sum / (n_tot.powi(3) - n_tot);
    let h_stat = if denom > 1e-12 {
        (h_raw / denom).max(0.0)
    } else {
        h_raw.max(0.0)
    };
    let df = groups.len() - 1;
    let p_value = chi2_upper_p(h_stat, df as f64);

    Ok(KruskalWallisResult {
        groups: groups.iter().map(|(n, _)| n.clone()).collect(),
        h_stat,
        df,
        p_value,
        significant: p_value < alpha,
    })
}

/// Non-parametric percentile bootstrap for sample mean (deterministic seed)
pub fn bootstrap_mean(
    target: &str,
    data: &[f64],
    resamples: usize,
) -> Result<BootstrapResult, String> {
    if data.len() < 2 {
        return Err("Bootstrapping krever minst 2 observasjoner.".to_string());
    }
    let b = resamples.clamp(200, 10000);
    let n = data.len();
    let mut rng = StdRng::seed_from_u64(42);
    let mut boot_means = Vec::with_capacity(b);

    for _ in 0..b {
        let mut s = 0.0;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            s += data[idx];
        }
        boot_means.push(s / n as f64);
    }
    let observed_mean = mean(data);
    let bootstrap_se = std_dev(&boot_means);
    let ci_low_95 = quantile(&boot_means, 0.025);
    let ci_high_95 = quantile(&boot_means, 0.975);

    Ok(BootstrapResult {
        target: target.to_string(),
        resamples: b,
        observed_mean,
        bootstrap_se,
        ci_low_95,
        ci_high_95,
    })
}
