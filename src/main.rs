mod auto;
mod data;
mod plots;
mod report;
mod stats;
#[cfg(test)]
mod tests;
mod ui;

use crate::auto::run_automatic_analysis;
use crate::data::Dataset;
use crate::report::{build_zip_bundle, render_markdown_report, render_pdf_report};
use clap::Parser;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "stat-analysis",
    version,
    about = "stat-analysis — Profesjonelt desktop-program i ren Rust for automatisk og manuell statistisk forutsetningssjekk, modellkvalitet og visualisering"
)]
struct Cli {
    /// Kjør automatisk analyse direkte på en CSV-fil uten GUI (CLI-modus)
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Utmappe for rapport og visualiseringer i CLI-modus
    #[arg(long, default_value = "stat_rapport")]
    out: PathBuf,

    /// Signifikansnivå (alpha)
    #[arg(long, default_value_t = 0.05)]
    alpha: f64,
}

fn main() {
    let cli = Cli::parse();

    if let Some(csv_path) = cli.csv {
        println!("Leser CSV-fil: {}", csv_path.display());
        let bytes = fs::read(&csv_path).unwrap_or_else(|e| {
            eprintln!("Feil ved lesing av fil: {}", e);
            std::process::exit(1);
        });
        let filename = csv_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("data.csv");
        let ds = Dataset::from_csv_bytes(filename, &bytes).unwrap_or_else(|e| {
            eprintln!("CSV-feil: {}", e);
            std::process::exit(1);
        });

        println!(
            "Analyserer {} rader x {} kolonner automatisk (alpha = {})...",
            ds.row_count(),
            ds.headers.len(),
            cli.alpha
        );
        let rep = run_automatic_analysis(&ds, cli.alpha);

        let vis_dir = cli.out.join("visualiseringer");
        fs::create_dir_all(&vis_dir).expect("Kunne ikke opprette utmappe");

        fs::write(cli.out.join("rapport.pdf"), render_pdf_report(&rep))
            .expect("Kunne ikke skrive rapport.pdf");
        fs::write(cli.out.join("rapport.md"), render_markdown_report(&rep))
            .expect("Kunne ikke skrive rapport.md");
        fs::write(
            cli.out.join("resultater.json"),
            serde_json::to_string_pretty(&rep).unwrap(),
        )
        .expect("Kunne ikke skrive resultater.json");

        for ch in &rep.charts {
            fs::write(vis_dir.join(&ch.filename), &ch.svg).expect("Kunne ikke skrive SVG");
        }
        if let Ok(zip_bytes) = build_zip_bundle(&rep) {
            fs::write(cli.out.join("rapport_og_visualiseringer.zip"), zip_bytes)
                .expect("Kunne ikke skrive ZIP");
        }

        println!("\n=== OPPSUMMERING ===");
        for b in &rep.summary_badges {
            println!(
                "[{:>7}] {} -> {}",
                b.status.to_uppercase(),
                b.check_name,
                b.headline
            );
        }
        println!(
            "\nFerdig! Lagret rapport og {} SVG-visualiseringer i: {}",
            rep.charts.len(),
            cli.out.display()
        );
        return;
    }

    // Native Rust Desktop GUI (eframe / egui)
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("stat-analysis — Statistisk Diagnostikk & Modellkvalitet")
            .with_inner_size([1360.0, 860.0])
            .with_min_inner_size([900.0, 620.0]),
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "stat-analysis",
        options,
        Box::new(|cc| Ok(Box::new(ui::StatApp::new(cc)))),
    ) {
        eprintln!("Klarte ikke starte desktop-grensesnittet: {}", e);
        std::process::exit(1);
    }
}
