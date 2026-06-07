import shutil
import subprocess
import sys
from pathlib import Path


def find_pdflatex():
    exe = shutil.which("pdflatex")
    if exe:
        return exe

    candidates = [
        Path(r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"),
        Path(r"C:\Program Files (x86)\MiKTeX\miktex\bin\pdflatex.exe"),
        Path(r"C:\texlive\2026\bin\windows\pdflatex.exe"),
        Path(r"C:\texlive\2025\bin\windows\pdflatex.exe"),
        Path(r"C:\texlive\2024\bin\windows\pdflatex.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def build_figure_check_pdf(figures, output_path):
    import fitz

    doc = fitz.open()
    page_w, page_h = 842, 595  # A4 landscape in points.
    margin = 36

    for title, image_path in figures:
        page = doc.new_page(width=page_w, height=page_h)
        page.insert_text(
            (margin, 28),
            title,
            fontsize=16,
            fontname="helv",
        )
        pix = fitz.Pixmap(str(image_path))
        img_w, img_h = pix.width, pix.height
        max_w = page_w - 2 * margin
        max_h = page_h - 80
        scale = min(max_w / img_w, max_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        x0 = (page_w - draw_w) / 2
        y0 = 60
        page.insert_image(
            fitz.Rect(x0, y0, x0 + draw_w, y0 + draw_h),
            filename=str(image_path),
        )

    doc.save(output_path)
    print(f"Built fallback figure-check PDF: {output_path}")


def main():
    root = Path(__file__).resolve().parent
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)

    copies = {
        root
        / "results"
        / "diffusion"
        / "tau_sweep_s4"
        / "cmmd_vs_time.png": figures / "diffusion_cmmd_vs_time.png",
        root
        / "results"
        / "diffusion"
        / "final_s4_sample_grid.png": figures
        / "diffusion_final_s4_sample_grid.png",
    }

    for src, dst in copies.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    pdflatex = find_pdflatex()
    if pdflatex is None:
        build_figure_check_pdf(
            [
                ("CMMD vs. generation time", figures / "diffusion_cmmd_vs_time.png"),
                (
                    "Generated samples: baseline, global cyclic, RACD",
                    figures / "diffusion_final_s4_sample_grid.png",
                ),
            ],
            reports / "diffusion_figures_check.pdf",
        )
        print()
        print("Could not find pdflatex.exe on PATH or in common MiKTeX/TeX Live paths.")
        print("Install MiKTeX/TeX Live or add pdflatex.exe to PATH to build diffusion_report.pdf.")
        print("The figure files were copied successfully into reports/figures/.")
        sys.exit(1)

    for pass_idx in range(2):
        print(f"pdflatex pass {pass_idx + 1}")
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "diffusion_report.tex"],
            cwd=reports,
            check=True,
        )

    pdf = reports / "diffusion_report.pdf"
    if not pdf.exists():
        raise FileNotFoundError(f"PDF was not created: {pdf}")
    print(f"Built {pdf}")


if __name__ == "__main__":
    main()
