use crate::auto::AutoAnalysisReport;
use crate::data::Dataset;
use crate::plots::{
    svg_acf_plot, svg_boxplot, svg_calibration_plot, svg_clusters_2d, svg_corr_heatmap,
    svg_elbow_silhouette, svg_histogram, svg_leverage_cooks, svg_missingness, svg_qq_plot,
    svg_residuals_vs_fitted, svg_roc_curve, svg_schoenfeld, svg_scree_plot, svg_sphericity_profile,
};
use crate::ui::charts::{
    bold_text, callout_box, kpi_card, plot_acf, plot_boxplot, plot_calibration, plot_clusters_2d,
    plot_corr_heatmap, plot_elbow_and_silhouette, plot_histogram, plot_leverage_cooks,
    plot_missingness_bars, plot_qq, plot_residuals_vs_fitted, plot_roc, plot_schoenfeld,
    plot_scree, plot_sphericity_profile, responsive_two_col, save_svg_button, secondary_button,
    show_card, show_table, status_badge, status_colors, table_header_row, SURFACE_CARD,
    TEXT_PRIMARY, TEXT_SECONDARY,
};
use crate::ui::NavSection;
use egui::{Color32, Margin, RichText, Stroke, Ui, Vec2};

pub struct ViewState {
    pub selected_normality_idx: usize,
    pub selected_schoenfeld_idx: usize,
    pub data_preview_limit: usize,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            selected_normality_idx: 0,
            selected_schoenfeld_idx: 0,
            data_preview_limit: 100,
        }
    }
}

fn page_header(ui: &mut Ui, title: &str, subtitle: &str) {
    ui.label(bold_text(title, 22.0, TEXT_PRIMARY));
    ui.label(RichText::new(subtitle).size(13.0).color(TEXT_SECONDARY));
    ui.add_space(10.0);
}

pub fn render_dashboard(
    ui: &mut Ui,
    rep: &AutoAnalysisReport,
    nav_target: &mut Option<NavSection>,
) {
    page_header(
        ui,
        "Automatisk Diagnostikk-oversikt",
        &format!(
            "Fil: {}  •  {} rader × {} kolonner  •  Generert {}",
            rep.filename, rep.row_count, rep.col_count, rep.generated_at
        ),
    );

    let ok_n = rep.summary_badges.iter().filter(|b| b.status == "ok").count();
    let warn_n = rep.summary_badges.iter().filter(|b| b.status == "warning").count();
    let fail_n = rep.summary_badges.iter().filter(|b| b.status == "fail").count();

    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Totalt sjekket", &format!("{}", rep.summary_badges.len()), Some("Forutsetninger & modell"));
        kpi_card(ui, "Oppfylt (OK)", &format!("{}", ok_n), Some("Ingen brudd"));
        kpi_card(ui, "Advarsler", &format!("{}", warn_n), Some("Bør vurderes"));
        kpi_card(ui, "Forutsetning brutt", &format!("{}", fail_n), Some("Bruk anbefalt tiltak"));
        kpi_card(ui, "Visualiseringer", &format!("{}", rep.charts.len()), Some("Interaktive + SVG"));
    });

    ui.add_space(14.0);
    if !rep.recommended_analyses.is_empty() {
        show_card(ui, |ui| {
            ui.label(bold_text("Automatisk valgte og utførte hovedanalyser (basert på forutsetningssjekk)", 14.5, TEXT_PRIMARY));
            ui.label(RichText::new("Programmet sjekker først normalitet, varianshomogenitet, sfærisitet og regresjonsforutsetninger — og utfører deretter riktig test (f.eks. Student t-test, Welchs t-test, Mann-Whitney, ANOVA, Welch ANOVA, Kruskal-Wallis eller robust OLS).").size(12.2).color(TEXT_SECONDARY));
            ui.add_space(6.0);
            show_table(ui, "rec_main_tbl", |ui| {
                table_header_row(ui, &["Analyseområde", "Utfall", "Valgt test", "Resultat", "Signifikant?"]);
                for ra in &rep.recommended_analyses {
                    ui.label(bold_text(&ra.domain, 12.5, TEXT_PRIMARY));
                    ui.label(RichText::new(&ra.outcome).color(TEXT_PRIMARY));
                    ui.label(bold_text(&ra.chosen_test, 12.5, Color32::from_rgb(29, 78, 216)));
                    ui.label(RichText::new(&ra.result_summary).color(TEXT_PRIMARY));
                    status_badge(ui, if ra.significant { "ok" } else { "warning" });
                    ui.end_row();
                }
            });
        });
        ui.add_space(14.0);
    }

    ui.label(bold_text("Alle sjekker — klikk «Åpne detaljer» på et kort for tabeller og interaktive plott", 14.5, TEXT_PRIMARY));
    ui.add_space(8.0);

    let avail = ui.available_width().max(260.0);
    let gap = 12.0_f32;
    let cols = if avail >= 960.0 {
        3
    } else if avail >= 620.0 {
        2
    } else {
        1
    };
    let card_w = ((avail - gap * (cols as f32 - 1.0)) / cols as f32).floor().max(240.0);

    for row_chunk in rep.summary_badges.chunks(cols) {
        ui.horizontal_top(|ui| {
            for (ci, b) in row_chunk.iter().enumerate() {
                if ci > 0 {
                    ui.add_space(gap);
                }
                let (_, _, border_col) = status_colors(&b.status);
                ui.allocate_ui_with_layout(
                    Vec2::new(card_w, 0.0),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        egui::Frame::new()
                            .fill(SURFACE_CARD)
                            .stroke(Stroke::new(1.4_f32, border_col))
                            .corner_radius(10.0_f32)
                            .inner_margin(Margin::same(13))
                            .show(ui, |ui| {
                                let inner_w = (card_w - 28.0).max(180.0);
                                ui.set_min_width(inner_w);
                                ui.set_max_width(inner_w);
                                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);

                                ui.horizontal_wrapped(|ui| {
                                    status_badge(ui, &b.status);
                                    ui.label(RichText::new(&b.category).size(11.0).color(TEXT_SECONDARY));
                                });
                                ui.add_space(4.0);
                                ui.label(bold_text(&b.check_name, 13.5, TEXT_PRIMARY));
                                ui.add_space(3.0);
                                ui.label(bold_text(&b.headline, 12.5, Color32::from_rgb(30, 58, 138)));
                                ui.add_space(3.0);
                                ui.label(RichText::new(&b.action).color(TEXT_SECONDARY).size(12.0));
                                ui.add_space(6.0);
                                if let Some(sec) = badge_to_section(&b.check_name) {
                                    if secondary_button(ui, "Åpne detaljer →").clicked() {
                                        *nav_target = Some(sec);
                                    }
                                }
                            });
                    },
                );
            }
        });
        ui.add_space(gap);
    }

    if !rep.notes.is_empty() {
        ui.add_space(4.0);
        show_card(ui, |ui| {
            ui.label(bold_text("Merknader fra automatisk kolonne-gjenkjenning", 14.0, TEXT_PRIMARY));
            ui.add_space(4.0);
            for n in &rep.notes {
                ui.label(RichText::new(format!("• {}", n)).color(TEXT_SECONDARY));
            }
        });
    }
}

fn badge_to_section(name: &str) -> Option<NavSection> {
    if name.contains("Normalfordeling") {
        Some(NavSection::Normality)
    } else if name.contains("Varianshomogenitet") {
        Some(NavSection::Variance)
    } else if name.contains("Sfærisitet") {
        Some(NavSection::Sphericity)
    } else if name.contains("Multikollinearitet") {
        Some(NavSection::Multicollinearity)
    } else if name.contains("Autokorrelasjon") {
        Some(NavSection::Autocorrelation)
    } else if name.contains("Heteroskedastisitet") {
        Some(NavSection::Heteroscedasticity)
    } else if name.contains("Uteliggere") {
        Some(NavSection::Outliers)
    } else if name.contains("manglende") {
        Some(NavSection::MissingData)
    } else if name.contains("faktoranalyse") {
        Some(NavSection::FactorSuitability)
    } else if name.contains("Lineær regresjon") {
        Some(NavSection::LinearRegression)
    } else if name.contains("Logistisk") || name.contains("ROC") {
        Some(NavSection::LogisticRegression)
    } else if name.contains("Cox") {
        Some(NavSection::Survival)
    } else if name.contains("Klyngeanalyse") {
        Some(NavSection::Clustering)
    } else {
        None
    }
}

pub fn render_normality(
    ui: &mut Ui,
    ds: &Dataset,
    rep: &AutoAnalysisReport,
    vs: &mut ViewState,
) {
    page_header(
        ui,
        "1. Normalfordeling",
        "Shapiro-Wilk (n < 50), Kolmogorov-Smirnov / Lilliefors (n >= 50), skjevhet, kurtose, Histogram og Q-Q-plott",
    );

    if rep.normality_checks.is_empty() {
        ui.label("Ingen numeriske variabler funnet for normalitetssjekk.");
        return;
    }

    show_card(ui, |ui| {
        ui.label(bold_text("Oversikt for alle numeriske variabler (klikk en variabel for å vise plott)", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "norm_tbl", |ui| {
            table_header_row(ui, &["Variabel", "n", "Anbefalt test", "Statistikk", "p-verdi", "Skjevhet", "Kurtose", "Status"]);
            for (idx, nc) in rep.normality_checks.iter().enumerate() {
                let selected = vs.selected_normality_idx == idx;
                if ui.selectable_label(selected, bold_text(&nc.variable, 13.0, TEXT_PRIMARY)).clicked() {
                    vs.selected_normality_idx = idx;
                }
                ui.label(format!("{}", nc.n));
                ui.label(&nc.recommended_test);
                ui.label(format!("{:.4}", nc.primary_statistic));
                ui.label(format!("{:.4}", nc.primary_p_value));
                ui.label(format!("{:.3}", nc.skewness));
                ui.label(format!("{:.3}", nc.excess_kurtosis));
                status_badge(ui, &nc.status);
                ui.end_row();
            }
        });
    });

    ui.add_space(12.0);
    let idx = vs.selected_normality_idx.min(rep.normality_checks.len() - 1);
    let nc = &rep.normality_checks[idx];

    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text(&format!("Detaljvisning & plott: {}", nc.variable), 15.0, TEXT_PRIMARY));
            if let Ok(vals) = ds.get_numeric_values(&nc.variable) {
                save_svg_button(ui, &format!("histogram_{}.svg", nc.variable), || {
                    svg_histogram(&format!("Histogram — {}", nc.variable), &nc.variable, &vals)
                });
                save_svg_button(ui, &format!("qq_{}.svg", nc.variable), || {
                    svg_qq_plot(&format!("Normal Q-Q-plott — {}", nc.variable), &vals)
                });
            }
        });
        ui.add_space(6.0);
        callout_box(ui, &nc.status, &nc.interpretation, &nc.recommendation);
        ui.add_space(8.0);

        ui.horizontal_wrapped(|ui| {
            if let Some(ref sw) = nc.shapiro_wilk {
                kpi_card(ui, "Shapiro-Wilk W", &format!("{:.4}", sw.w), Some(&format!("p = {:.4}", sw.p_value)));
            }
            kpi_card(
                ui,
                "KS-test D",
                &format!("{:.4}", nc.ks_test.d_stat),
                Some(&format!("Lilliefors p = {:.4}", nc.ks_test.p_value_lilliefors)),
            );
            kpi_card(ui, "Skjevhet", &format!("{:.3}", nc.skewness), Some("Nær 0 = symmetrisk"));
            kpi_card(ui, "Eksess-kurtose", &format!("{:.3}", nc.excess_kurtosis), Some("Nær 0 = normal"));
        });

        ui.add_space(10.0);
        if let Ok(vals) = ds.get_numeric_values(&nc.variable) {
            responsive_two_col(
                ui,
                |ui| {
                    ui.label(bold_text("Histogram med normalfordelingskurve", 13.5, TEXT_PRIMARY));
                    plot_histogram(ui, &format!("hist_{}", nc.variable), &vals, 270.0);
                },
                |ui| {
                    ui.label(bold_text("Normal Q-Q-plott", 13.5, TEXT_PRIMARY));
                    plot_qq(ui, &format!("qq_{}", nc.variable), &vals, 270.0);
                },
            );
        }
    });
}

pub fn render_variance(ui: &mut Ui, ds: &Dataset, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "2. Varianshomogenitet & Ikke-parametriske alternativer",
        "Levenes test, Bartletts test, Boksplott + Welchs t-test/ANOVA, Mann-Whitney U, Kruskal-Wallis og Bootstrap",
    );
    let Some(ref vh) = rep.variance_homogeneity else {
        ui.label("Ingen gruppekolonne ble gjenkjent automatisk. Bruk «Manuell testbenk» for å velge gruppekolonne manuelt.");
        return;
    };

    callout_box(ui, &vh.status, &vh.interpretation, &vh.recommendation);
    ui.add_space(10.0);

    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Levene (Median)", &format!("F = {:.3}", vh.levene_median.f_stat), Some(&format!("p = {:.4}", vh.levene_median.p_value)));
        kpi_card(ui, "Levene (Snitt)", &format!("F = {:.3}", vh.levene_mean.f_stat), Some(&format!("p = {:.4}", vh.levene_mean.p_value)));
        kpi_card(ui, "Bartletts test", &format!("χ² = {:.3}", vh.bartlett.chi2_stat), Some(&format!("p = {:.4}", vh.bartlett.p_value)));
        kpi_card(ui, "Welchs ANOVA", &format!("F = {:.3}", vh.welch_anova.f_stat), Some(&format!("p = {:.4}", vh.welch_anova.p_value)));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        let groups: Vec<(String, Vec<f64>)> = ds
            .get_grouped_values(&vh.value_col, &vh.group_col)
            .map(|m| m.into_iter().filter(|(_, v)| v.len() >= 2).collect())
            .unwrap_or_default();
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text(&format!("Boksplott: {} per {}", vh.value_col, vh.group_col), 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "boksplott_grupper.svg", || {
                svg_boxplot(&format!("Boksplott: {} per {}", vh.value_col, vh.group_col), &vh.value_col, &groups)
            });
        });
        ui.add_space(6.0);
        plot_boxplot(ui, &groups, 270.0);
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text("Gruppe-sammendrag og robuste/ikke-parametriske alternativer", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "grp_sum_tbl", |ui| {
            table_header_row(ui, &["Gruppe", "n", "Gjennomsnitt", "Median", "Std.avvik", "Varians"]);
            for g in &vh.group_summaries {
                ui.label(bold_text(&g.name, 13.0, TEXT_PRIMARY));
                ui.label(format!("{}", g.n));
                ui.label(format!("{:.3}", g.mean));
                ui.label(format!("{:.3}", g.median));
                ui.label(format!("{:.3}", g.std_dev));
                ui.label(format!("{:.3}", g.variance));
                ui.end_row();
            }
        });

        ui.add_space(10.0);
        if let Some(ref st) = vh.student_t_test {
            ui.label(bold_text(
                &format!(
                    "• Student uavhengig t-test ({} vs {}): t({}) = {:.3}, p = {:.4}, diff = {:.3}, Cohen's d = {:.3}",
                    st.group1, st.group2, st.df, st.t_stat, st.p_value, st.mean_diff, st.cohens_d
                ),
                13.0,
                TEXT_PRIMARY,
            ));
        }
        ui.label(format!(
            "• Standard enveis ANOVA: F({}, {}) = {:.3}, p = {:.4}, SSB = {:.2}, SSW = {:.2}, eta² = {:.3}",
            vh.standard_anova.df1, vh.standard_anova.df2, vh.standard_anova.f_stat, vh.standard_anova.p_value,
            vh.standard_anova.ss_between, vh.standard_anova.ss_within, vh.standard_anova.eta_squared
        ));
        if let Some(ref wt) = vh.welch_t_test {
            ui.label(format!(
                "• Welchs t-test ({} vs {}): t({:.1}) = {:.3}, p = {:.4}",
                wt.group1, wt.group2, wt.df, wt.t_stat, wt.p_value
            ));
        }
        if let Some(ref fb) = rep.group_fallbacks {
            if let Some(ref mw) = fb.mann_whitney {
                ui.label(format!(
                    "• Mann-Whitney U (ikke-parametrisk): U = {:.1}, Z = {:.3}, p = {:.4}",
                    mw.u_stat, mw.z_stat, mw.p_value
                ));
            }
            if let Some(ref kw) = fb.kruskal_wallis {
                ui.label(format!(
                    "• Kruskal-Wallis H (ikke-parametrisk): H({}) = {:.3}, p = {:.4}",
                    kw.df, kw.h_stat, kw.p_value
                ));
            }
            if !fb.bootstraps.is_empty() {
                ui.add_space(8.0);
                ui.label(bold_text("Bootstrapping (95 % persentil-konfidensintervall for gjennomsnitt):", 13.5, TEXT_PRIMARY));
                show_table(ui, "boot_tbl", |ui| {
                    table_header_row(ui, &["Mål", "Observert snitt", "Bootstrap SE", "95% KI nedre", "95% KI øvre"]);
                    for b in &fb.bootstraps {
                        ui.label(&b.target);
                        ui.label(format!("{:.3}", b.observed_mean));
                        ui.label(format!("{:.3}", b.bootstrap_se));
                        ui.label(format!("{:.3}", b.ci_low_95));
                        ui.label(format!("{:.3}", b.ci_high_95));
                        ui.end_row();
                    }
                });
            }
        }
    });
}

pub fn render_sphericity(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "3. Sfærisitet (Repeated Measures ANOVA)",
        "Mauchlys test for lik varians i differanser + Greenhouse-Geisser og Huynh-Feldt korreksjoner",
    );
    let Some(ref sp) = rep.sphericity else {
        ui.label("Ingen repeterte målinger (minst 3 numeriske kolonner) funnet. Kjør gjerne manuelt i testbenken.");
        return;
    };

    callout_box(ui, &sp.status, &sp.interpretation, &sp.recommendation);
    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Mauchlys W", &format!("{:.4}", sp.mauchly_w), Some(&format!("p = {:.4}", sp.mauchly_p_value)));
        kpi_card(ui, "RM-ANOVA F", &format!("{:.3}", sp.rm_anova_f), Some(&format!("Ukorr. p = {:.4}", sp.p_uncorrected)));
        kpi_card(ui, "Greenhouse-Geisser ε", &format!("{:.3}", sp.epsilon_gg), Some(&format!("Korr. p = {:.4}", sp.p_greenhouse_geisser)));
        kpi_card(ui, "Huynh-Feldt ε", &format!("{:.3}", sp.epsilon_hf), Some(&format!("Korr. p = {:.4}", sp.p_huynh_feldt)));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text(&format!("Profilplott ({})", sp.columns.join(" → ")), 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "sfaerisitet_profil.svg", || {
                svg_sphericity_profile("Repeterte målinger — Gjennomsnitt", &sp.columns, &sp.condition_means)
            });
        });
        ui.add_space(6.0);
        plot_sphericity_profile(ui, "spher_plot", &sp.condition_means, 260.0);
    });
}

pub fn render_multicollinearity(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "4. Multikollinearitet (VIF & Korrelasjonsmatrise)",
        "Variance Inflation Factor (< 5 anbefalt, > 10 alvorlig), korrelasjonsmatrise (|r| > 0.8), Ridge- og Lasso-regresjon",
    );
    let Some(ref mc) = rep.linear_regression.as_ref().and_then(|r| r.multicollinearity.as_ref()) else {
        ui.label("Multikollinearitet krever minst 2 forklaringsvariabler i regresjon.");
        return;
    };

    callout_box(ui, &mc.status, &mc.interpretation, &mc.recommendation);
    ui.add_space(10.0);

    show_card(ui, |ui| {
        ui.label(bold_text("VIF per forklaringsvariabel", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "vif_tbl", |ui| {
            table_header_row(ui, &["Variabel", "VIF", "Hjelpe-R²", "Vurdering"]);
            for v in &mc.vifs {
                ui.label(bold_text(&v.variable, 13.0, TEXT_PRIMARY));
                ui.label(format!("{:.2}", v.vif));
                ui.label(format!("{:.4}", v.r_squared_aux));
                status_badge(ui, &v.severity);
                ui.end_row();
            }
        });
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("Korrelasjonsmatrise (Heatmap)", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "korrelasjonsmatrise.svg", || {
                svg_corr_heatmap("Korrelasjonsmatrise", &mc.predictors, &mc.correlation_matrix)
            });
        });
        ui.add_space(6.0);
        plot_corr_heatmap(ui, "corr_mc_heat", &mc.predictors, &mc.correlation_matrix);
    });

    if mc.ridge_fallback.is_some() || mc.lasso_fallback.is_some() {
        ui.add_space(12.0);
        show_card(ui, |ui| {
            ui.label(bold_text("Regulariserte alternativer: Ridge (L2) og Lasso (L1)", 14.5, TEXT_PRIMARY));
            ui.add_space(6.0);
            responsive_two_col(
                ui,
                |ui| {
                    if let Some(ref rd) = mc.ridge_fallback {
                        ui.label(bold_text(&format!("Ridge (λ = {:.2}) — R² = {:.4}, RMSE = {:.3}", rd.lambda, rd.r_squared, rd.rmse), 13.0, TEXT_PRIMARY));
                        for (name, val) in &rd.coefficients {
                            ui.label(format!("  {}: {:.4}", name, val));
                        }
                    }
                },
                |ui| {
                    if let Some(ref la) = mc.lasso_fallback {
                        ui.label(bold_text(&format!("Lasso (λ = {:.2}) — R² = {:.4}, RMSE = {:.3}", la.lambda, la.r_squared, la.rmse), 13.0, TEXT_PRIMARY));
                        for (name, val) in &la.coefficients {
                            ui.label(format!("  {}: {:.4}", name, val));
                        }
                    }
                },
            );
        });
    }
}

pub fn render_autocorrelation(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "5. Autokorrelasjon (Avhengige feilledd)",
        "Durbin-Watson (nær 2.0 er bra), Breusch-Godfrey LM-test, ACF-plott, Newey-West HAC standardfeil og AR(1)",
    );
    let Some(ref reg) = rep.linear_regression else {
        ui.label("Ingen lineær regresjonsmodell tilgjengelig.");
        return;
    };
    let ac = &reg.autocorrelation;

    callout_box(ui, &ac.status, &ac.interpretation, &ac.recommendation);
    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Durbin-Watson", &format!("{:.3}", ac.durbin_watson), Some(&ac.dw_assessment));
        kpi_card(ui, &format!("Breusch-Godfrey ({})", ac.bg_lags), &format!("LM = {:.3}", ac.bg_lm_stat), Some(&format!("p = {:.4}", ac.bg_p_value)));
        kpi_card(ui, "AR(1) φ", &format!("{:.3}", ac.ar1_phi), Some(&format!("AR(1) RMSE = {:.3}", ac.ar1_rmse)));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("ACF-plott (Autokorrelasjonsfunksjon for residualer)", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "acf_plott.svg", || {
                svg_acf_plot("ACF av residualer", &ac.acf_lags, ac.acf_conf_bound)
            });
        });
        ui.add_space(6.0);
        plot_acf(ui, "acf_main", &ac.acf_lags, ac.acf_conf_bound, 260.0);
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text("Newey-West (HAC) robuste standardfeil vs. vanlig OLS", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "nw_tbl", |ui| {
            table_header_row(ui, &["Variabel", "Estimat β", "OLS SE", "OLS p", "Newey-West SE", "Newey-West p"]);
            for (i, c) in reg.coefficients.iter().enumerate() {
                let nw = &ac.newey_west_coefs[i];
                ui.label(bold_text(&c.name, 13.0, TEXT_PRIMARY));
                ui.label(format!("{:.4}", c.estimate));
                ui.label(format!("{:.4}", c.std_error));
                ui.label(format!("{:.4}", c.p_value));
                ui.label(format!("{:.4}", nw.std_error));
                ui.label(format!("{:.4}", nw.p_value));
                ui.end_row();
            }
        });
    });
}

pub fn render_heteroscedasticity(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "6. Heteroskedastisitet (Ulik spredning i residualer)",
        "Breusch-Pagan-test, Whites test, Residualplott + HC3 robuste standardfeil og Log-transformasjon av Y",
    );
    let Some(ref reg) = rep.linear_regression else {
        ui.label("Ingen lineær regresjonsmodell tilgjengelig.");
        return;
    };
    let ht = &reg.heteroscedasticity;

    callout_box(ui, &ht.status, &ht.interpretation, &ht.recommendation);
    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Breusch-Pagan", &format!("LM = {:.3}", ht.bp_lm_stat), Some(&format!("df={}, p = {:.4}", ht.bp_df, ht.bp_p_value)));
        kpi_card(ui, "Whites test", &format!("LM = {:.3}", ht.white_lm_stat), Some(&format!("df={}, p = {:.4}", ht.white_df, ht.white_p_value)));
        if ht.log_y_fallback.applicable {
            kpi_card(
                ui,
                "Log(Y) BP p-verdi",
                &format!("{:.4}", ht.log_y_fallback.bp_p_value.unwrap_or(0.0)),
                Some(&format!("Log(Y) R² = {:.3}", ht.log_y_fallback.r_squared.unwrap_or(0.0))),
            );
        }
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("Residualplott (Predikerte verdier vs. Residualer)", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "residualplott.svg", || {
                svg_residuals_vs_fitted("Residualer mot predikert verdi", &reg.fitted, &reg.residuals)
            });
        });
        ui.add_space(6.0);
        plot_residuals_vs_fitted(ui, "rvf_main", &reg.fitted, &reg.residuals, 280.0);
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text("Robuste standardfeil (HC3) vs. standard OLS", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "hc3_tbl", |ui| {
            table_header_row(ui, &["Variabel", "Estimat β", "OLS SE", "OLS p", "HC3 SE", "HC3 p"]);
            for (i, c) in reg.coefficients.iter().enumerate() {
                let h3 = &ht.hc3_coefs[i];
                ui.label(bold_text(&c.name, 13.0, TEXT_PRIMARY));
                ui.label(format!("{:.4}", c.estimate));
                ui.label(format!("{:.4}", c.std_error));
                ui.label(format!("{:.4}", c.p_value));
                ui.label(format!("{:.4}", h3.std_error));
                ui.label(format!("{:.4}", h3.p_value));
                ui.end_row();
            }
        });
    });
}

pub fn render_outliers(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "7. Uteliggere og innflytelse",
        "Cook's distance (D > 1 og D > 4/n), Mahalanobis-avstand, Leverage-plott og automatisk sensitivitetsanalyse",
    );
    let Some(ref reg) = rep.linear_regression else {
        ui.label("Ingen regresjonsmodell tilgjengelig.");
        return;
    };
    let out = &reg.outliers;

    callout_box(ui, &out.status, &out.interpretation, &out.recommendation);
    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Cook's D > 1", &format!("{}", out.count_cooks_gt_1), Some("Sterk uteligger"));
        kpi_card(ui, &format!("Cook's D > {:.4}", out.cooks_threshold_4_over_n), &format!("{}", out.count_cooks_gt_4_n), Some("4/n terskel"));
        kpi_card(ui, "Mahalanobis p<0.001", &format!("{}", out.count_mahalanobis_outliers), Some("Multivariat uteligger"));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("Leverage- og Cook's Distance-plott", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "leverage_cooks.svg", || {
                svg_leverage_cooks(
                    "Standardiserte residualer mot Leverage",
                    &out.leverage,
                    &out.standardized_residuals,
                    &out.cooks_d,
                    out.cooks_threshold_4_over_n,
                )
            });
        });
        ui.add_space(6.0);
        plot_leverage_cooks(
            ui,
            "lev_main",
            &out.leverage,
            &out.standardized_residuals,
            &out.cooks_d,
            out.cooks_threshold_4_over_n,
            280.0,
        );
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text("Mest innflytelsesrike observasjoner", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "out_tbl", |ui| {
            table_header_row(ui, &["Rad #", "Cook's D", "Leverage", "Residual", "Mahalanobis D²", "Mahal. p-verdi"]);
            for pt in out.top_influential.iter().take(8) {
                ui.label(format!("#{}", pt.row_index));
                ui.label(format!("{:.4}", pt.cooks_d));
                ui.label(format!("{:.4}", pt.leverage));
                ui.label(format!("{:.3}", pt.residual));
                ui.label(format!("{:.2}", pt.mahalanobis_d2));
                ui.label(format!("{:.4}", pt.mahalanobis_p));
                ui.end_row();
            }
        });

        if let Some(ref sens) = out.sensitivity {
            ui.add_space(10.0);
            ui.label(bold_text(
                &format!(
                    "Sensitivitetsanalyse (uten {} innflytelsesrike observasjoner): R² endres {:.4} → {:.4}, RMSE endres {:.3} → {:.3}",
                    sens.removed_count, sens.orig_r_squared, sens.clean_r_squared, sens.orig_rmse, sens.clean_rmse
                ),
                13.0,
                TEXT_PRIMARY,
            ));
        }
    });
}

pub fn render_missing(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "8. Tilfeldig manglende data (Littles MCAR, MICE & FIML)",
        "Mønsteranalyse av manglende verdier, Littles MCAR-test og multippel imputering (MICE) / EM-estimering (FIML)",
    );
    let m = &rep.missing_data;
    callout_box(ui, &m.status, &m.interpretation, &m.recommendation);
    ui.add_space(10.0);

    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Manglende celler", &format!("{}", m.total_missing_cells), Some(&format!("{:.2}% totalt", m.overall_missing_pct)));
        kpi_card(ui, "Komplette rader", &format!("{} / {}", m.complete_rows, m.total_rows), None);
        kpi_card(
            ui,
            "Littles MCAR χ²",
            &m.mcar_chi2.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "N/A".into()),
            Some(&m.mcar_p_value.map(|p| format!("p = {:.4}", p)).unwrap_or_else(|| "Ikke relevant".into())),
        );
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("Manglende verdier per kolonne", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "manglende_verdier.svg", || {
                svg_missingness("Manglende verdier per kolonne (%)", &m.columns)
            });
        });
        ui.add_space(8.0);
        plot_missingness_bars(ui, "miss_main", &m.columns);
    });

    if !m.mice_summary.is_empty() {
        ui.add_space(12.0);
        show_card(ui, |ui| {
            ui.label(bold_text(&format!("Multippel imputering (MICE, m={}) vs. FIML (EM) vs. komplette caser", m.mice_imputations), 14.5, TEXT_PRIMARY));
            ui.add_space(6.0);
            show_table(ui, "mice_tbl", |ui| {
                table_header_row(ui, &["Kolonne", "Komplett snitt", "FIML (EM) snitt", "MICE poolet snitt", "MICE SE (Rubin)"]);
                for x in &m.mice_summary {
                    ui.label(bold_text(&x.column, 13.0, TEXT_PRIMARY));
                    ui.label(format!("{:.4}", x.complete_case_mean));
                    ui.label(format!("{:.4}", x.fiml_em_mean));
                    ui.label(format!("{:.4}", x.mice_pooled_mean));
                    ui.label(format!("{:.4}", x.mice_pooled_se));
                    ui.end_row();
                }
            });
        });
    }
}

pub fn render_factor(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "9. Egnethet for faktoranalyse / PCA",
        "KMO-test (bør være > 0.6), individuell MSA, Bartletts sfærisitetstest (p < 0.05) og Scree-plott",
    );
    let Some(ref fac) = rep.factor_suitability else {
        ui.label("Trenger minst 3 numeriske kolonner for KMO og Bartletts sfærisitetstest.");
        return;
    };

    callout_box(ui, &fac.status, &fac.interpretation, &fac.recommendation);
    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Samlet KMO", &format!("{:.3}", fac.kmo_overall), Some(&fac.kmo_label));
        kpi_card(ui, "Bartlett χ²", &format!("{:.1}", fac.bartlett_chi2), Some(&format!("df={}, p = {:.4}", fac.bartlett_df, fac.bartlett_p_value)));
        kpi_card(ui, "Kaiser-faktorer", &format!("{}", fac.kaiser_num_factors), Some("Egenverdi > 1.0"));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("Scree-plott (PCA egenverdier)", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "faktor_scree.svg", || {
                svg_scree_plot("Scree-plott — Egenverdier", &fac.eigenvalues)
            });
        });
        ui.add_space(6.0);
        plot_scree(ui, "scree_main", &fac.eigenvalues, 260.0);
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text("MSA per variabel", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "msa_tbl", |ui| {
            table_header_row(ui, &["Variabel", "Individuell MSA", "Maks |r| mot andre"]);
            for v in &fac.per_variable_msa {
                ui.label(bold_text(&v.variable, 13.0, TEXT_PRIMARY));
                ui.label(format!("{:.3}", v.msa));
                ui.label(format!("{:.3}", v.max_abs_corr));
                ui.end_row();
            }
        });
    });
}

pub fn render_linear_regression(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "10. Lineær regresjon — Modellkvalitet & AIC / BIC",
        "Forklart varians (R², Justert R²), prediksjonsfeil (RMSE, MAE), koeffisienter og modellsammenligning (AIC/BIC)",
    );
    let Some(ref reg) = rep.linear_regression else {
        ui.label("Ingen lineær regresjonsmodell beregnet.");
        return;
    };

    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "R²", &format!("{:.4}", reg.r_squared), Some(&format!("Justert R² = {:.4}", reg.adj_r_squared)));
        kpi_card(ui, "RMSE", &format!("{:.4}", reg.rmse), Some(&format!("MAE = {:.4}", reg.mae)));
        kpi_card(ui, "AIC", &format!("{:.2}", reg.aic), Some("Lavere = bedre"));
        kpi_card(ui, "BIC", &format!("{:.2}", reg.bic), Some("Straffer kompleksitet"));
        kpi_card(ui, "F-statistikk", &format!("{:.2}", reg.f_stat), Some(&format!("p = {:.4}", reg.f_p_value)));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text(&format!("Modellsammenligning — AIC & BIC (Målvariabel: {})", reg.target), 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "aic_tbl", |ui| {
            table_header_row(ui, &["Modell", "Parametere", "R²", "Just. R²", "RMSE", "MAE", "AIC", "BIC"]);
            for m in &reg.model_comparison {
                ui.label(bold_text(&m.model_name, 13.0, TEXT_PRIMARY));
                ui.label(format!("{}", m.k_params));
                ui.label(format!("{:.4}", m.r_squared));
                ui.label(format!("{:.4}", m.adj_r_squared));
                ui.label(format!("{:.4}", m.rmse));
                ui.label(format!("{:.4}", m.mae));
                ui.label(bold_text(&format!("{:.2}", m.aic), 13.0, TEXT_PRIMARY));
                ui.label(bold_text(&format!("{:.2}", m.bic), 13.0, TEXT_PRIMARY));
                ui.end_row();
            }
        });
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text("Koeffisienter (OLS)", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "ols_tbl", |ui| {
            table_header_row(ui, &["Variabel", "Estimat β", "Std.feil", "t-verdi", "p-verdi"]);
            for c in &reg.coefficients {
                ui.label(bold_text(&c.name, 13.0, TEXT_PRIMARY));
                ui.label(format!("{:.4}", c.estimate));
                ui.label(format!("{:.4}", c.std_error));
                ui.label(format!("{:.3}", c.t_stat));
                ui.label(format!("{:.4}", c.p_value));
                ui.end_row();
            }
        });
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("Residualenes normalfordeling (Q-Q-plott)", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "regresjon_residual_qq.svg", || {
                svg_qq_plot("Residual Q-Q-plott", &reg.residuals)
            });
        });
        ui.add_space(6.0);
        callout_box(ui, &reg.residual_normality.status, &reg.residual_normality.interpretation, &reg.residual_normality.recommendation);
        ui.add_space(6.0);
        plot_qq(ui, "reg_qq_main", &reg.residuals, 250.0);
    });
}

pub fn render_logistic(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "11. Logistisk regresjon — Hosmer-Lemeshow & ROC / AUC",
        "Goodness-of-fit (Hosmer-Lemeshow p > 0.05), diskrimineringsevne (ROC-kurve & AUC) og AIC / BIC",
    );
    let Some(ref lr) = rep.logistic_regression else {
        ui.label("Ingen binær utfallskolonne ble funnet automatisk. Bruk «Manuell testbenk» for å velge målvariabel.");
        return;
    };

    callout_box(ui, &lr.auc_status, &lr.interpretation, &lr.recommendation);
    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "ROC AUC", &format!("{:.3}", lr.auc), Some(&lr.auc_label));
        kpi_card(ui, "Hosmer-Lemeshow", &format!("p = {:.4}", lr.hosmer_lemeshow.p_value), Some(&format!("χ²({}) = {:.2}", lr.hosmer_lemeshow.df, lr.hosmer_lemeshow.chi2_stat)));
        kpi_card(ui, "Nøyaktighet / F1", &format!("{:.1}%", lr.confusion_matrix.accuracy * 100.0), Some(&format!("F1 = {:.3}", lr.confusion_matrix.f1_score)));
        kpi_card(ui, "AIC / BIC", &format!("{:.1}", lr.aic), Some(&format!("BIC = {:.1}", lr.bic)));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("ROC-kurve og Hosmer-Lemeshow kalibrering", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "roc_kurve.svg", || {
                svg_roc_curve(&format!("ROC-kurve — {}", lr.target), &lr.roc_points, lr.auc)
            });
            save_svg_button(ui, "hosmer_lemeshow.svg", || {
                svg_calibration_plot("Hosmer-Lemeshow kalibrering", &lr.hosmer_lemeshow.bins)
            });
        });
        ui.add_space(6.0);
        responsive_two_col(
            ui,
            |ui| plot_roc(ui, "roc_main", &lr.roc_points, lr.auc, 270.0),
            |ui| plot_calibration(ui, "cal_main", &lr.hosmer_lemeshow.bins, 270.0),
        );
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text("Koeffisienter og Odds Ratio (OR)", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "log_tbl", |ui| {
            table_header_row(ui, &["Variabel", "β (log-odds)", "Odds Ratio", "SE", "z", "p-verdi"]);
            for c in &lr.coefficients {
                ui.label(bold_text(&c.name, 13.0, TEXT_PRIMARY));
                ui.label(format!("{:.4}", c.estimate));
                ui.label(format!("{:.3}", c.odds_ratio));
                ui.label(format!("{:.4}", c.std_error));
                ui.label(format!("{:.2}", c.z_stat));
                ui.label(format!("{:.4}", c.p_value));
                ui.end_row();
            }
        });
    });
}

pub fn render_survival(ui: &mut Ui, rep: &AutoAnalysisReport, vs: &mut ViewState) {
    page_header(
        ui,
        "12. Overlevelsesanalyse — Cox PH & Schoenfeld-residualer",
        "Proporsjonal hasard-forutsetning (Schoenfeld-residualer p > 0.05 indikerer konstant risikoforhold over tid)",
    );
    let Some(ref cox) = rep.cox_survival else {
        ui.label("Ingen overlevelseskolonner (tid + hendelse) ble gjenkjent automatisk. Bruk «Manuell testbenk» for å velge kolonner.");
        return;
    };

    callout_box(ui, &cox.status, &cox.interpretation, &cox.recommendation);
    ui.add_space(10.0);
    ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Schoenfeld global p", &format!("{:.4}", cox.schoenfeld_global_p), Some(&format!("χ²({}) = {:.2}", cox.schoenfeld_global_df, cox.schoenfeld_global_chi2)));
        kpi_card(ui, "Harrells C-index", &format!("{:.3}", cox.concordance_c), Some("Diskrimineringsevne"));
        kpi_card(ui, "Hendelser / N", &format!("{} / {}", cox.n_events, cox.n), Some(&format!("{} / {}", cox.time_col, cox.event_col)));
        kpi_card(ui, "AIC / BIC", &format!("{:.1}", cox.aic), Some(&format!("BIC = {:.1}", cox.bic)));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.label(bold_text("Cox Hazard Ratios & Schoenfeld-test per kovariat (klikk en rad for plott)", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "cox_tbl", |ui| {
            table_header_row(ui, &["Kovariat", "β", "Hazard Ratio", "HR(tidlig)", "HR(sen)", "Wald p", "Schoenfeld p", "PH oppfylt?"]);
            for (i, c) in cox.coefficients.iter().enumerate() {
                let st = &cox.schoenfeld_tests[i];
                if ui.selectable_label(vs.selected_schoenfeld_idx == i, bold_text(&c.name, 13.0, TEXT_PRIMARY)).clicked() {
                    vs.selected_schoenfeld_idx = i;
                }
                ui.label(format!("{:.4}", c.coef));
                ui.label(format!("{:.3}", c.hazard_ratio));
                ui.label(format!("{:.3}", st.hr_early_t25));
                ui.label(format!("{:.3}", st.hr_late_t75));
                ui.label(format!("{:.4}", c.p_value));
                ui.label(format!("{:.4}", st.p_value));
                status_badge(ui, if st.ph_met { "ok" } else { "fail" });
                ui.end_row();
            }
        });
    });

    if !cox.schoenfeld_tests.is_empty() {
        let idx = vs.selected_schoenfeld_idx.min(cox.schoenfeld_tests.len() - 1);
        let st = &cox.schoenfeld_tests[idx];
        let beta = cox.coefficients[idx].coef;
        ui.add_space(12.0);
        show_card(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(bold_text(&format!("Schoenfeld-residualer mot tid — {}", st.variable), 14.5, TEXT_PRIMARY));
                save_svg_button(ui, &format!("schoenfeld_{}.svg", st.variable), || {
                    svg_schoenfeld(&format!("Schoenfeld — {}", st.variable), &st.variable, &st.points, beta)
                });
            });
            ui.add_space(6.0);
            plot_schoenfeld(ui, &format!("sch_{}", st.variable), &st.points, beta, 270.0);
        });
    }
}

pub fn render_clustering(ui: &mut Ui, rep: &AutoAnalysisReport) {
    page_header(
        ui,
        "13. Klyngeanalyse — Silhouette-skår & Elbow-metoden",
        "K-means++ klyngeseparasjon (Silhouette nær +1 er tydelig adskilt) og Elbow-metoden (WCSS)",
    );
    let Some(ref cl) = rep.clustering else {
        ui.label("Trenger minst 2 numeriske variabler for klyngeanalyse.");
        return;
    };

    callout_box(ui, &cl.status, &cl.interpretation, &cl.recommendation);
    ui.add_space(10.0);
        ui.horizontal_wrapped(|ui| {
        kpi_card(ui, "Beste Silhouette", &format!("{:.3}", cl.best_silhouette), Some(&format!("Ved k = {}", cl.best_k_silhouette)));
        kpi_card(ui, "Elbow anbefaler", &format!("k = {}", cl.elbow_k), Some("Knekkpunkt på WCSS"));
        kpi_card(
            ui,
            "DBSCAN-alternativ",
            &format!("{} klynger", cl.dbscan_fallback.n_clusters),
            Some(&format!("eps={:.2}, {} støy", cl.dbscan_fallback.eps, cl.dbscan_fallback.noise_points)),
        );
        kpi_card(ui, "Observasjoner", &format!("{}", cl.n), Some(&format!("{} variabler", cl.variables.len())));
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text("Elbow- og Silhouette-kurver", 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "klynge_elbow_silhouette.svg", || {
                svg_elbow_silhouette("Elbow og Silhouette", &cl.k_evals, cl.elbow_k, cl.best_k_silhouette)
            });
        });
        ui.add_space(6.0);
        plot_elbow_and_silhouette(ui, "cl_main", &cl.k_evals, 250.0);
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text(&format!("2D PCA-projeksjon av klynger (k = {})", cl.best_k_silhouette), 14.5, TEXT_PRIMARY));
            save_svg_button(ui, "klynger_pca2d.svg", || {
                svg_clusters_2d(&format!("Klynger (k={})", cl.best_k_silhouette), &cl.points_2d)
            });
        });
        ui.add_space(6.0);
        plot_clusters_2d(ui, "cl_2d_main", &cl.points_2d, 280.0);
    });
}

pub fn render_data_explorer(ui: &mut Ui, ds: &Dataset, vs: &mut ViewState) {
    page_header(
        ui,
        "Datautforsker & Kolonnesammendrag",
        &format!("Fil: {} • Skilletegn: '{}' • {} rader × {} kolonner", ds.filename, ds.delimiter, ds.row_count(), ds.headers.len()),
    );

    show_card(ui, |ui| {
        ui.label(bold_text("Kolonneoversikt og deskriptiv statistikk", 14.5, TEXT_PRIMARY));
        ui.add_space(6.0);
        show_table(ui, "cols_tbl", |ui| {
            table_header_row(ui, &["Kolonne", "Type", "Gyldige", "Mangler", "Unike", "Snitt", "SD", "Min", "Maks"]);
            for c in &ds.columns {
                ui.label(bold_text(&c.name, 13.0, TEXT_PRIMARY));
                ui.label(format!("{:?}", c.kind));
                ui.label(format!("{}", c.valid_count));
                ui.label(format!("{}", c.missing_count));
                ui.label(format!("{}", c.unique_count));
                ui.label(c.mean.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "—".into()));
                ui.label(c.std_dev.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "—".into()));
                ui.label(c.min.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "—".into()));
                ui.label(c.max.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "—".into()));
                ui.end_row();
            }
        });
    });

    ui.add_space(12.0);
    show_card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(bold_text(&format!("Datatabell (viser første {} rader)", vs.data_preview_limit.min(ds.row_count())), 14.5, TEXT_PRIMARY));
            ui.add(egui::DragValue::new(&mut vs.data_preview_limit).range(10..=500).prefix("Rader: "));
        });
        ui.add_space(6.0);
        show_table(ui, "raw_tbl", |ui| {
            ui.label(bold_text("#", 12.5, TEXT_PRIMARY));
            for h in &ds.headers {
                ui.label(bold_text(h, 12.5, TEXT_PRIMARY));
            }
            ui.end_row();
            for (ri, row) in ds.raw_rows.iter().take(vs.data_preview_limit).enumerate() {
                ui.label(RichText::new(format!("{}", ri + 1)).color(TEXT_SECONDARY));
                for cell in row {
                    match cell {
                        Some(s) => ui.label(RichText::new(s).color(TEXT_PRIMARY)),
                        None => ui.label(bold_text("NA", 12.5, Color32::from_rgb(185, 28, 28))),
                    };
                }
                ui.end_row();
            }
        });
    });
}
