"""Tkinter GUI for generating Ush Analytics pro reports.

Allows selecting event and match CSV files, choosing a match, 
selecting an output folder and running
:func:`run_all_pro.generate_report_for_match`.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd

import run_all_pro


def main():
    """Launch GUI for report generation."""
    root = tk.Tk()
    root.title("Ush Analytics – Reporte Pro")

    events_path = tk.StringVar()
    matches_path = tk.StringVar()
    output_dir = tk.StringVar()
    match_var = tk.StringVar()
    status_var = tk.StringVar()

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

    def select_matches():
        """Pick matches CSV, read it and populate dropdown."""
        nonlocal matches_df
        path = filedialog.askopenfilename(
            title="Seleccionar matches.csv", filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        matches_path.set(path)
        matches_df = pd.read_csv(path)
        match_map.clear()
        options = []
        for _, row in matches_df.iterrows():
            display = f"{row['match_id']}: {row['home_team']} vs {row['away_team']}"
            match_map[display] = row['match_id']
            options.append(display)
        if options:
            match_var.set(options[0])
            menu = match_menu["menu"]
            menu.delete(0, "end")
            for opt in options:
                menu.add_command(label=opt, command=tk._setit(match_var, opt))
        matches_btn.config(text=f"Partidos: {Path(path).name}")

    def choose_output():
        """Select an output directory for generated files."""
        path = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if path:
            output_dir.set(path)
            out_btn.config(text=f"Salida: {Path(path).name}")

    def generate_report():
        """Run report generation and show status."""
        if not (events_path.get() and matches_df is not None and output_dir.get()):
            messagebox.showerror("Error", "Seleccione archivos y carpeta de salida")
            return
        try:
            events_df = pd.read_csv(events_path.get())
            result = run_all_pro.generate_report_for_match(
                events_df, matches_df, output_dir.get(), match_id=match_map.get(match_var.get())
            )
            report_path = result[0].get("report") if result else ""
            status_var.set(f"Reporte generado en {report_path}")
            messagebox.showinfo("Éxito", status_var.get())
        except Exception as exc:
            status_var.set(f"Error: {exc}")
            messagebox.showerror("Error", status_var.get())

    # --- layout ---
    events_btn = tk.Button(root, text="Eventos...", command=select_events)
    events_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    matches_btn = tk.Button(root, text="Partidos...", command=select_matches)
    matches_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    match_menu = tk.OptionMenu(root, match_var, "")
    match_menu.config(width=40)
    match_menu.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    out_btn = tk.Button(root, text="Elegir salida...", command=choose_output)
    out_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

    run_btn = tk.Button(root, text="Generar reporte", command=generate_report)
    run_btn.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    status_label = tk.Label(root, textvariable=status_var, anchor="w")
    status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    root.mainloop()


if __name__ == "__main__":
    main()
