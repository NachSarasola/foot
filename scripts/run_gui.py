"""Tkinter GUI for generating Ush Analytics pro reports.

Allows selecting event and match CSV files, choosing a match,
selecting an output folder and running
:func:`run_all_pro.run_pipeline`.
"""

from tkinter import (
    Tk,
    StringVar,
    PhotoImage,
    ttk,
    filedialog,
    messagebox,
)
from pathlib import Path
import pandas as pd

from ush_style import COLORS
import run_all_pro


data_dir = Path(__file__).resolve().parents[1] / "data"
default_out = Path(__file__).resolve().parents[1] / "report"


def main():
    """Launch GUI for report generation."""
    root = Tk()
    root.title("Ush Analytics – Reporte Pro")
    root.geometry("520x180")
    root.configure(background=COLORS["paper"])

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "TButton",
        font=("Rajdhani", 12, "bold"),
        foreground=COLORS["paper"],
        background=COLORS["blue"],
    )
    style.configure(
        "TLabel",
        font=("Inter", 11),
        background=COLORS["paper"],
        foreground=COLORS["ink"],
    )
    style.configure(
        "TCombobox",
        font=("Inter", 11),
        fieldbackground=COLORS["paper"],
        foreground=COLORS["ink"],
    )

    icon_path = Path(__file__).resolve().parents[1] / "brand" / "ush_avatar_on_fog.png"
    if icon_path.exists():
        icon_img = PhotoImage(file=str(icon_path))
        root.iconphoto(False, icon_img)

    events_path = StringVar(value=str(data_dir / "events.csv"))
    matches_path = StringVar(value=str(data_dir / "matches.csv"))
    output_dir = StringVar(value=str(default_out))
    match_var = StringVar()
    status_var = StringVar()

    matches_df = None  # DataFrame con los partidos cargados
    match_map = {}  # Relaciona el texto mostrado con ``match_id``

    def select_events():
        """Pick events CSV using a file dialog."""
        path = filedialog.askopenfilename(
            title="Seleccionar events.csv", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            events_path.set(path)
            events_btn.config(text=f"Eventos: {Path(path).name}")

    def load_matches(path: str):
        """Load matches from CSV and populate dropdown."""
        nonlocal matches_df
        if not path:
            return
        matches_df = pd.read_csv(path)
        teams = pd.read_csv(data_dir / "teams.csv").set_index("team_id")["name"]
        matches_df["home_team"] = matches_df["home_team_id"].map(teams)
        matches_df["away_team"] = matches_df["away_team_id"].map(teams)
        match_map.clear()
        options = []
        for _, row in matches_df.iterrows():
            display = f"{row['match_id']}: {row['home_team']} vs {row['away_team']}"
            match_map[display] = row['match_id']
            options.append(display)
        if options:
            match_var.set(options[0])
            match_cb["values"] = options
            match_cb.current(0)
        matches_btn.config(text=f"Partidos: {Path(path).name}")

    def select_matches():
        """Pick matches CSV, read it and populate dropdown."""
        path = filedialog.askopenfilename(
            title="Seleccionar matches.csv", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            matches_path.set(path)
            load_matches(path)

    def choose_output():
        """Select an output directory for generated files."""
        path = filedialog.askdirectory(title="Cambiar salida...")
        if path:
            output_dir.set(path)
            out_btn.config(text=f"Salida: {Path(path).name}")

    def generate_report():
        """Run report generation and show status."""
        if not (
            events_path.get()
            and matches_path.get()
            and matches_df is not None
            and output_dir.get()
        ):
            messagebox.showerror("Error", "Seleccione archivos y carpeta de salida")
            return

        controls = [events_btn, matches_btn, match_cb, out_btn, run_btn]
        for widget in controls:
            widget.config(state="disabled")
        progress.grid()
        progress.start()
        root.update_idletasks()

        try:
            result = run_all_pro.run_pipeline(
                events_path.get(),
                matches_path.get(),
                output_dir.get(),
                match_id=match_map.get(match_var.get()),
            )
            report_path = result[0].get("report") if result else ""
            status_var.set(f"Reporte generado en {report_path}")
            messagebox.showinfo("Éxito", status_var.get())
        except Exception as exc:
            status_var.set(f"Error: {exc}")
            messagebox.showerror("Error", status_var.get())
        finally:
            progress.stop()
            progress.grid_remove()
            for widget in controls:
                widget.config(state="normal")
            root.update_idletasks()

    # --- layout ---
    events_btn = ttk.Button(root, text="Eventos", command=select_events)
    events_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    matches_btn = ttk.Button(root, text="Partidos", command=select_matches)
    matches_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    match_cb = ttk.Combobox(root, textvariable=match_var, state="readonly", width=40)
    match_cb.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    out_btn = ttk.Button(
        root, text=f"Salida: {default_out.name}", command=choose_output
    )
    out_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

    run_btn = ttk.Button(root, text="Generar reporte", command=generate_report)
    run_btn.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    status_label = ttk.Label(root, textvariable=status_var, anchor="w")
    status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    progress = ttk.Progressbar(root, mode="indeterminate")
    progress.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    progress.grid_remove()

    load_matches(matches_path.get())

    root.mainloop()


if __name__ == "__main__":
    main()
