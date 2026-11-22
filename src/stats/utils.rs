use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal, StudentsT};

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn variance(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64
}

pub fn std_dev(x: &[f64]) -> f64 {
    variance(x).sqrt()
}

pub fn median(x: &[f64]) -> f64 {
    quantile(x, 0.5)
}

pub fn quantile(x: &[f64], p: f64) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 1 {
        return s[0];
    }
    let pos = p.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    s[lo] * (1.0 - frac) + s[hi] * frac
}

/// Sample adjusted Fisher-Pearson skewness (G1)
pub fn skewness(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 3 {
        return f64::NAN;
    }
    let m = mean(x);
    let s = std_dev(x);
    if s <= 1e-15 {
        return 0.0;
    }
    let m3: f64 = x.iter().map(|v| ((v - m) / s).powi(3)).sum();
    (n / ((n - 1.0) * (n - 2.0))) * m3
}

/// Sample excess kurtosis (G2, 0 for normal distribution)
pub fn excess_kurtosis(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 4 {
        return f64::NAN;
    }
    let m = mean(x);
    let s = std_dev(x);
    if s <= 1e-15 {
        return 0.0;
    }
    let m4: f64 = x.iter().map(|v| ((v - m) / s).powi(4)).sum();
    let term1 = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * m4;
    let term2 = (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0));
    term1 - term2
}

pub fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let mx = mean(x);
    let my = mean(y);
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for (&a, &b) in x.iter().zip(y.iter()) {
        let da = a - mx;
        let db = b - my;
        num += da * db;
        dx2 += da * da;
        dy2 += db * db;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom <= 1e-15 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

pub fn correlation_matrix(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if rows.is_empty() {
        return vec![];
    }
    let p = rows[0].len();
    let cols: Vec<Vec<f64>> = (0..p)
        .map(|j| rows.iter().map(|r| r[j]).collect())
        .collect();
    let mut mat = vec![vec![0.0; p]; p];
    for i in 0..p {
        mat[i][i] = 1.0;
        for j in (i + 1)..p {
            let r = pearson_corr(&cols[i], &cols[j]);
            mat[i][j] = r;
            mat[j][i] = r;
        }
    }
    mat
}

pub fn covariance_matrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let n = rows.len();
    let p = rows[0].len();
    let mut means = vec![0.0; p];
    for r in rows {
        for j in 0..p {
            means[j] += r[j];
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }
    let mut cov = DMatrix::zeros(p, p);
    let denom = (n.saturating_sub(1)).max(1) as f64;
    for r in rows {
        for i in 0..p {
            let di = r[i] - means[i];
            for j in i..p {
                let dj = r[j] - means[j];
                cov[(i, j)] += di * dj;
            }
        }
    }
    for i in 0..p {
        for j in i..p {
            cov[(i, j)] /= denom;
            cov[(j, i)] = cov[(i, j)];
        }
    }
    cov
}

/// Average ranks (1-based) with tie handling; also returns sum of (t^3 - t) for tie correction.
pub fn rank_average(values: &[f64]) -> (Vec<f64>, f64) {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut tie_sum = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() <= 1e-12 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_sum += t.powi(3) - t;
        }
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    (ranks, tie_sum)
}

pub fn normal_cdf(x: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().cdf(x)
}

pub fn normal_inv_cdf(p: f64) -> f64 {
    Normal::new(0.0, 1.0)
        .unwrap()
        .inverse_cdf(p.clamp(1e-15, 1.0 - 1e-15))
}

pub fn normal_two_tailed_p(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    2.0 * (1.0 - normal_cdf(z.abs()))
}

pub fn t_two_tailed_p(t: f64, df: f64) -> f64 {
    if !t.is_finite() || !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => (2.0 * (1.0 - dist.cdf(t.abs()))).clamp(0.0, 1.0),
        Err(_) => f64::NAN,
    }
}

pub fn chi2_upper_p(stat: f64, df: f64) -> f64 {
    if !stat.is_finite() || !df.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    if stat <= 0.0 {
        return 1.0;
    }
    match ChiSquared::new(df) {
        Ok(dist) => (1.0 - dist.cdf(stat)).clamp(0.0, 1.0),
        Err(_) => f64::NAN,
    }
}

pub fn f_upper_p(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if !f_stat.is_finite() || !df1.is_finite() || !df2.is_finite() || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if f_stat <= 0.0 {
        return 1.0;
    }
    match FisherSnedecor::new(df1, df2) {
        Ok(dist) => (1.0 - dist.cdf(f_stat)).clamp(0.0, 1.0),
        Err(_) => f64::NAN,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlsCoef {
    pub name: String,
    pub estimate: f64,
    pub std_error: f64,
    pub t_stat: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone)]
pub struct OlsFit {
    pub names: Vec<String>,
    pub beta: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub leverage: Vec<f64>,
    pub x_matrix: DMatrix<f64>,
    pub xtx_inv: DMatrix<f64>,
    pub n: usize,
    pub k: usize, // number of parameters including intercept
    pub rss: f64,
    pub tss: f64,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub rmse: f64,
    pub mae: f64,
    pub f_stat: f64,
    pub f_p_value: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
}

impl OlsFit {
    pub fn coefs(&self) -> Vec<OlsCoef> {
        (0..self.k)
            .map(|i| OlsCoef {
                name: self.names[i].clone(),
                estimate: self.beta[i],
                std_error: self.std_errors[i],
                t_stat: self.t_stats[i],
                p_value: self.p_values[i],
            })
            .collect()
    }
}

/// Fit OLS regression Y = b0 + b1*X1 + ... + bp*Xp using SVD pseudo-inverse for numerical stability.
pub fn fit_ols(
    y: &[f64],
    x_cols: &[Vec<f64>],
    predictor_names: &[String],
) -> Result<OlsFit, String> {
    let n = y.len();
    let p = x_cols.len();
    let k = p + 1;
    if n <= k + 1 {
        return Err(format!(
            "For få observasjoner (n={}) for {} parametere (inkludert konstantledd).",
            n, k
        ));
    }
    for (j, col) in x_cols.iter().enumerate() {
        if col.len() != n {
            return Err(format!("Kolonne {} har feil lengde.", j));
        }
    }

    let mut x = DMatrix::zeros(n, k);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        for j in 0..p {
            x[(i, j + 1)] = x_cols[j][i];
        }
    }
    let y_vec = DVector::from_column_slice(y);

    let svd = x.clone().svd(true, true);
    let beta_vec = svd
        .solve(&y_vec, 1e-10)
        .map_err(|e| format!("Kunne ikke løse OLS: {}", e))?;

    let xtx = x.transpose() * &x;
    let xtx_inv = xtx
        .clone()
        .try_inverse()
        .unwrap_or_else(|| xtx.pseudo_inverse(1e-10).unwrap_or_else(|_| DMatrix::identity(k, k)));

    let fitted_vec = &x * &beta_vec;
    let resid_vec = &y_vec - &fitted_vec;

    let fitted: Vec<f64> = fitted_vec.iter().copied().collect();
    let residuals: Vec<f64> = resid_vec.iter().copied().collect();

    let rss: f64 = residuals.iter().map(|e| e * e).sum();
    let y_mean = mean(y);
    let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if tss > 1e-15 {
        (1.0 - rss / tss).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let df_resid = (n - k) as f64;
    let adj_r_squared = if n > k && tss > 1e-15 {
        1.0 - (1.0 - r_squared) * ((n - 1) as f64 / df_resid)
    } else {
        r_squared
    };

    let sigma2 = rss / df_resid;
    let rmse = (rss / n as f64).sqrt();
    let mae = residuals.iter().map(|e| e.abs()).sum::<f64>() / n as f64;

    let mut std_errors = Vec::with_capacity(k);
    let mut t_stats = Vec::with_capacity(k);
    let mut p_values = Vec::with_capacity(k);
    for j in 0..k {
        let se = (sigma2 * xtx_inv[(j, j)].max(0.0)).sqrt();
        let t = if se > 1e-15 { beta_vec[j] / se } else { 0.0 };
        let pv = t_two_tailed_p(t, df_resid);
        std_errors.push(se);
        t_stats.push(t);
        p_values.push(pv);
    }

    // Leverage h_ii = x_i^T (X^T X)^-1 x_i
    let mut leverage = Vec::with_capacity(n);
    for i in 0..n {
        let xi = x.row(i);
        let h = (&xi * &xtx_inv * xi.transpose())[(0, 0)];
        leverage.push(h.clamp(0.0, 0.999999));
    }

    let (f_stat, f_p_value) = if p > 0 && rss > 1e-15 {
        let ess = (tss - rss).max(0.0);
        let f = (ess / p as f64) / (rss / df_resid);
        (f, f_upper_p(f, p as f64, df_resid))
    } else {
        (0.0, 1.0)
    };

    // Gaussian log-likelihood (counting sigma^2 as an additional parameter k+1 for standard R/statsmodels AIC/BIC)
    let sigma2_mle = (rss / n as f64).max(1e-30);
    let log_likelihood = -0.5 * (n as f64) * ((2.0 * std::f64::consts::PI * sigma2_mle).ln() + 1.0);
    let num_params = (k + 1) as f64;
    let aic = 2.0 * num_params - 2.0 * log_likelihood;
    let bic = (n as f64).ln() * num_params - 2.0 * log_likelihood;

    let mut names = vec!["(Konstantledd)".to_string()];
    names.extend(predictor_names.iter().cloned());

    Ok(OlsFit {
        names,
        beta: beta_vec.iter().copied().collect(),
        std_errors,
        t_stats,
        p_values,
        fitted,
        residuals,
        leverage,
        x_matrix: x,
        xtx_inv,
        n,
        k,
        rss,
        tss,
        r_squared,
        adj_r_squared,
        rmse,
        mae,
        f_stat,
        f_p_value,
        log_likelihood,
        aic,
        bic,
    })
}

pub fn round4(v: f64) -> f64 {
    if v.is_finite() {
        (v * 10000.0).round() / 10000.0
    } else {
        v
    }
}
