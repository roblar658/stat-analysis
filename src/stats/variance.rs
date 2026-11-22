use crate::stats::utils::{chi2_upper_p, f_upper_p, mean, median, t_two_tailed_p, variance};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupSummary {
    pub name: String,
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeveneResult {
    pub center: String,
    pub f_stat: f64,
    pub df1: usize,
    pub df2: usize,
    pub p_value: f64,
    pub equal_variances: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartlettResult {
    pub chi2_stat: f64,
    pub df: usize,
    pub p_value: f64,
    pub equal_variances: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentTTestResult {
    pub group1: String,
    pub group2: String,
    pub n1: usize,
    pub n2: usize,
    pub mean1: f64,
    pub mean2: f64,
    pub mean_diff: f64,
    pub pooled_sd: f64,
    pub std_error: f64,
    pub t_stat: f64,
    pub df: usize,
    pub p_value: f64,
    pub cohens_d: f64,
    pub significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardAnovaResult {
    pub ss_between: f64,
    pub ss_within: f64,
    pub ms_between: f64,
    pub ms_within: f64,
    pub f_stat: f64,
    pub df1: usize,
    pub df2: usize,
    pub p_value: f64,
    pub eta_squared: f64,
    pub significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelchTTestResult {
    pub group1: String,
    pub group2: String,
    pub t_stat: f64,
    pub df: f64,
    pub p_value: f64,
    pub mean_diff: f64,
    pub significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelchAnovaResult {
    pub f_stat: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
    pub significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceHomogeneityReport {
    pub value_col: String,
    pub group_col: String,
    pub group_summaries: Vec<GroupSummary>,
    pub levene_mean: LeveneResult,
    pub levene_median: LeveneResult,
    pub bartlett: BartlettResult,
    pub student_t_test: Option<StudentTTestResult>,
    pub standard_anova: StandardAnovaResult,
    pub welch_t_test: Option<WelchTTestResult>,
    pub welch_anova: WelchAnovaResult,
    pub status: String,
    pub interpretation: String,
    pub recommendation: String,
}

pub fn levene_test(
    groups: &[(String, Vec<f64>)],
    use_median: bool,
    alpha: f64,
) -> Result<LeveneResult, String> {
    let k = groups.len();
    if k < 2 {
        return Err("Levenes test krever minst 2 grupper.".to_string());
    }
    let mut z_groups: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut n_tot = 0usize;
    for (_, vals) in groups {
        if vals.len() < 2 {
            return Err("Hver gruppe må ha minst 2 observasjoner for Levenes test.".to_string());
        }
        let c = if use_median { median(vals) } else { mean(vals) };
        let zs: Vec<f64> = vals.iter().map(|v| (v - c).abs()).collect();
        n_tot += zs.len();
        z_groups.push(zs);
    }

    if n_tot <= k {
        return Err("For få observasjoner totalt for Levenes test.".to_string());
    }

    let z_bar_all: f64 = z_groups.iter().flatten().sum::<f64>() / n_tot as f64;
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for zg in &z_groups {
        let ni = zg.len() as f64;
        let z_bar_i = mean(zg);
        ss_between += ni * (z_bar_i - z_bar_all).powi(2);
        ss_within += zg.iter().map(|&z| (z - z_bar_i).powi(2)).sum::<f64>();
    }

    let df1 = k - 1;
    let df2 = n_tot - k;
    let f_stat = if ss_within > 1e-15 {
        (ss_between / df1 as f64) / (ss_within / df2 as f64)
    } else {
        0.0
    };
    let p_value = f_upper_p(f_stat, df1 as f64, df2 as f64);

    Ok(LeveneResult {
        center: if use_median {
            "Median (Brown-Forsythe)".to_string()
        } else {
            "Gjennomsnitt (Levene)".to_string()
        },
        f_stat,
        df1,
        df2,
        p_value,
        equal_variances: p_value >= alpha,
    })
}

pub fn bartlett_test(
    groups: &[(String, Vec<f64>)],
    alpha: f64,
) -> Result<BartlettResult, String> {
    let k = groups.len();
    if k < 2 {
        return Err("Bartletts test krever minst 2 grupper.".to_string());
    }

    let mut n_minus_1_sum = 0.0;
    let mut weighted_var_sum = 0.0;
    let mut log_var_sum = 0.0;
    let mut inv_df_sum = 0.0;

    for (_, vals) in groups {
        if vals.len() < 2 {
            return Err("Hver gruppe må ha minst 2 observasjoner for Bartletts test.".to_string());
        }
        let df_i = (vals.len() - 1) as f64;
        let v_i = variance(vals).max(1e-15);
        n_minus_1_sum += df_i;
        weighted_var_sum += df_i * v_i;
        log_var_sum += df_i * v_i.ln();
        inv_df_sum += 1.0 / df_i;
    }

    let sp2 = (weighted_var_sum / n_minus_1_sum).max(1e-15);
    let num = n_minus_1_sum * sp2.ln() - log_var_sum;
    let c = 1.0 + (1.0 / (3.0 * (k - 1) as f64)) * (inv_df_sum - 1.0 / n_minus_1_sum);
    let chi2_stat = (num / c).max(0.0);
    let df = k - 1;
    let p_value = chi2_upper_p(chi2_stat, df as f64);

    Ok(BartlettResult {
        chi2_stat,
        df,
        p_value,
        equal_variances: p_value >= alpha,
    })
}

pub fn student_t_test(
    g1_name: &str,
    g1: &[f64],
    g2_name: &str,
    g2: &[f64],
    alpha: f64,
) -> Result<StudentTTestResult, String> {
    if g1.len() < 2 || g2.len() < 2 {
        return Err("Student t-test krever minst 2 observasjoner i hver gruppe.".to_string());
    }
    let n1 = g1.len();
    let n2 = g2.len();
    let m1 = mean(g1);
    let m2 = mean(g2);
    let v1 = variance(g1);
    let v2 = variance(g2);
    let df = n1 + n2 - 2;
    let sp2 = (((n1 - 1) as f64 * v1) + ((n2 - 1) as f64 * v2)) / (df as f64);
    let pooled_sd = sp2.max(0.0).sqrt();
    let se = (sp2 * (1.0 / n1 as f64 + 1.0 / n2 as f64)).max(1e-18).sqrt();
    let mean_diff = m1 - m2;
    let t_stat = if se > 1e-15 { mean_diff / se } else { 0.0 };
    let p_value = t_two_tailed_p(t_stat, df as f64);
    let cohens_d = if pooled_sd > 1e-15 {
        mean_diff / pooled_sd
    } else {
        0.0
    };

    Ok(StudentTTestResult {
        group1: g1_name.to_string(),
        group2: g2_name.to_string(),
        n1,
        n2,
        mean1: m1,
        mean2: m2,
        mean_diff,
        pooled_sd,
        std_error: se,
        t_stat,
        df,
        p_value,
        cohens_d,
        significant: p_value < alpha,
    })
}

pub fn standard_one_way_anova(
    groups: &[(String, Vec<f64>)],
    alpha: f64,
) -> Result<StandardAnovaResult, String> {
    let k = groups.len();
    if k < 2 {
        return Err("Enveis ANOVA krever minst 2 grupper.".to_string());
    }
    let n_tot: usize = groups.iter().map(|(_, v)| v.len()).sum();
    if n_tot <= k {
        return Err("For få observasjoner for enveis ANOVA.".to_string());
    }
    let grand_mean: f64 = groups.iter().flat_map(|(_, v)| v.iter().copied()).sum::<f64>() / n_tot as f64;
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for (_, vals) in groups {
        let m_i = mean(vals);
        ss_between += (vals.len() as f64) * (m_i - grand_mean).powi(2);
        ss_within += vals.iter().map(|&x| (x - m_i).powi(2)).sum::<f64>();
    }
    let df1 = k - 1;
    let df2 = n_tot - k;
    let ms_between = ss_between / df1 as f64;
    let ms_within = (ss_within / df2 as f64).max(1e-15);
    let f_stat = (ms_between / ms_within).max(0.0);
    let p_value = f_upper_p(f_stat, df1 as f64, df2 as f64);
    let ss_total = (ss_between + ss_within).max(1e-15);
    let eta_squared = (ss_between / ss_total).clamp(0.0, 1.0);

    Ok(StandardAnovaResult {
        ss_between,
        ss_within,
        ms_between,
        ms_within,
        f_stat,
        df1,
        df2,
        p_value,
        eta_squared,
        significant: p_value < alpha,
    })
}

pub fn welch_t_test(
    g1_name: &str,
    g1: &[f64],
    g2_name: &str,
    g2: &[f64],
    alpha: f64,
) -> Result<WelchTTestResult, String> {
    if g1.len() < 2 || g2.len() < 2 {
        return Err("Welchs t-test krever minst 2 observasjoner i hver gruppe.".to_string());
    }
    let m1 = mean(g1);
    let m2 = mean(g2);
    let v1 = variance(g1);
    let v2 = variance(g2);
    let n1 = g1.len() as f64;
    let n2 = g2.len() as f64;

    let se2 = v1 / n1 + v2 / n2;
    if se2 <= 1e-15 {
        return Ok(WelchTTestResult {
            group1: g1_name.to_string(),
            group2: g2_name.to_string(),
            t_stat: 0.0,
            df: n1 + n2 - 2.0,
            p_value: 1.0,
            mean_diff: m1 - m2,
            significant: false,
        });
    }
    let t_stat = (m1 - m2) / se2.sqrt();
    let df = se2.powi(2) / ((v1 / n1).powi(2) / (n1 - 1.0) + (v2 / n2).powi(2) / (n2 - 1.0));
    let p_value = t_two_tailed_p(t_stat, df);

    Ok(WelchTTestResult {
        group1: g1_name.to_string(),
        group2: g2_name.to_string(),
        t_stat,
        df,
        p_value,
        mean_diff: m1 - m2,
        significant: p_value < alpha,
    })
}

pub fn welch_anova(
    groups: &[(String, Vec<f64>)],
    alpha: f64,
) -> Result<WelchAnovaResult, String> {
    let k = groups.len();
    if k < 2 {
        return Err("Welchs ANOVA krever minst 2 grupper.".to_string());
    }
    let kf = k as f64;
    let mut weights = Vec::with_capacity(k);
    let mut means = Vec::with_capacity(k);
    let mut ns = Vec::with_capacity(k);

    for (_, vals) in groups {
        if vals.len() < 2 {
            return Err("Hver gruppe må ha minst 2 observasjoner for Welchs ANOVA.".to_string());
        }
        let n_i = vals.len() as f64;
        let v_i = variance(vals).max(1e-12);
        weights.push(n_i / v_i);
        means.push(mean(vals));
        ns.push(n_i);
    }

    let w_sum: f64 = weights.iter().sum();
    let x_bar_w: f64 = weights
        .iter()
        .zip(means.iter())
        .map(|(&w, &m)| w * m)
        .sum::<f64>()
        / w_sum;

    let num: f64 = weights
        .iter()
        .zip(means.iter())
        .map(|(&w, &m)| w * (m - x_bar_w).powi(2))
        .sum::<f64>()
        / (kf - 1.0);

    let lambda_sum: f64 = weights
        .iter()
        .zip(ns.iter())
        .map(|(&w, &n_i)| (1.0 - w / w_sum).powi(2) / (n_i - 1.0))
        .sum();

    let denom = 1.0 + (2.0 * (kf - 2.0) / (kf * kf - 1.0)) * lambda_sum;
    let f_stat = (num / denom).max(0.0);
    let df1 = kf - 1.0;
    let df2 = if lambda_sum > 1e-15 {
        (kf * kf - 1.0) / (3.0 * lambda_sum)
    } else {
        ns.iter().sum::<f64>() - kf
    };
    let p_value = f_upper_p(f_stat, df1, df2);

    Ok(WelchAnovaResult {
        f_stat,
        df1,
        df2,
        p_value,
        significant: p_value < alpha,
    })
}

pub fn check_variance_homogeneity(
    value_col: &str,
    group_col: &str,
    groups: &[(String, Vec<f64>)],
    alpha: f64,
) -> Result<VarianceHomogeneityReport, String> {
    let valid_groups: Vec<(String, Vec<f64>)> = groups
        .iter()
        .filter(|(_, v)| v.len() >= 2)
        .cloned()
        .collect();
    if valid_groups.len() < 2 {
        return Err("Trenger minst 2 grupper med >= 2 observasjoner hver.".to_string());
    }

    let group_summaries = valid_groups
        .iter()
        .map(|(name, vals)| {
            let var = variance(vals);
            GroupSummary {
                name: name.clone(),
                n: vals.len(),
                mean: mean(vals),
                median: median(vals),
                std_dev: var.sqrt(),
                variance: var,
            }
        })
        .collect();

    let levene_mean = levene_test(&valid_groups, false, alpha)?;
    let levene_median = levene_test(&valid_groups, true, alpha)?;
    let bartlett = bartlett_test(&valid_groups, alpha)?;

    let student_t = if valid_groups.len() == 2 {
        Some(student_t_test(
            &valid_groups[0].0,
            &valid_groups[0].1,
            &valid_groups[1].0,
            &valid_groups[1].1,
            alpha,
        )?)
    } else {
        None
    };
    let standard_a = standard_one_way_anova(&valid_groups, alpha)?;

    let welch_t = if valid_groups.len() == 2 {
        Some(welch_t_test(
            &valid_groups[0].0,
            &valid_groups[0].1,
            &valid_groups[1].0,
            &valid_groups[1].1,
            alpha,
        )?)
    } else {
        None
    };
    let welch_a = welch_anova(&valid_groups, alpha)?;

    let ok = levene_median.equal_variances && bartlett.equal_variances;
    let status = if ok {
        "ok"
    } else if levene_median.equal_variances {
        "warning"
    } else {
        "fail"
    }
    .to_string();

    let interpretation = if levene_median.equal_variances {
        format!(
            "Varianshomogenitet ser ut til å være oppfylt (Brown-Forsythe Levene p = {:.4}, Bartlett p = {:.4}).",
            levene_median.p_value, bartlett.p_value
        )
    } else {
        format!(
            "Varianshomogenitet er brutt (Levene median p = {:.4} < {:.2}, Bartlett p = {:.4}).",
            levene_median.p_value, alpha, bartlett.p_value
        )
    };

    let recommendation = if levene_median.equal_variances {
        "Standard uavhengig t-test eller vanlig enveis ANOVA kan brukes.".to_string()
    } else if valid_groups.len() == 2 {
        "Bruk Welchs t-test (beregnet under) i stedet for standard Student t-test.".to_string()
    } else {
        "Bruk Welchs ANOVA (beregnet under) i stedet for standard ANOVA.".to_string()
    };

    Ok(VarianceHomogeneityReport {
        value_col: value_col.to_string(),
        group_col: group_col.to_string(),
        group_summaries,
        levene_mean,
        levene_median,
        bartlett,
        student_t_test: student_t,
        standard_anova: standard_a,
        welch_t_test: welch_t,
        welch_anova: welch_a,
        status,
        interpretation,
        recommendation,
    })
}
