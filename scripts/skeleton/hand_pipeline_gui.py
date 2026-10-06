#!/usr/bin/env python3
"""
Haptica Hand Tracking Pipeline - GUI

Simple graphical interface for the hand tracking pipeline.
Wraps the CLI tool with a user-friendly interface.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Add the script directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


class HandPipelineGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Haptica Hand Tracking Pipeline")
        self.root.geometry("700x600")
        self.root.configure(bg="#1a1a2e")

        # Variables
        self.video_path = tk.StringVar()
        self.params_path = tk.StringVar()
        self.cameras_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.home() / "hand_pipeline_output"))

        self.render_skeleton = tk.BooleanVar(value=True)
        self.render_mesh = tk.BooleanVar(value=False)
        self.export_json = tk.BooleanVar(value=True)

        self.device = tk.StringVar(value="cuda")
        self.is_running = False

        self._build_ui()

    def _build_ui(self):
        # Style
        style = ttk.Style()
        style.theme_use("clam")

        # Colors
        bg = "#1a1a2e"
        fg = "#eaeaea"
        accent = "#00d4aa"  # Haptica teal
        entry_bg = "#16213e"

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Helvetica", 11))
        style.configure("TButton", font=("Helvetica", 11), padding=8)
        style.configure("Accent.TButton", background=accent, foreground="#000")
        style.configure("TCheckbutton", background=bg, foreground=fg, font=("Helvetica", 11))
        style.configure("TEntry", fieldbackground=entry_bg, foreground=fg)
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), foreground=accent)

        self.root.configure(bg=bg)

        # Main container
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main, text="🖐️ Haptica Hand Tracking Pipeline", style="Header.TLabel")
        header.pack(pady=(0, 20))

        # === Input Section ===
        input_frame = ttk.LabelFrame(main, text="Input Files", padding=15)
        input_frame.pack(fill=tk.X, pady=10)

        # Video file
        self._add_file_row(input_frame, "Video File:", self.video_path,
                          [("MP4 files", "*.mp4"), ("All files", "*.*")], 0)

        # Parameters file
        self._add_file_row(input_frame, "Parameters (.pth):", self.params_path,
                          [("PyTorch files", "*.pth"), ("All files", "*.*")], 1)

        # Cameras file (optional)
        self._add_file_row(input_frame, "Cameras (optional):", self.cameras_path,
                          [("JSON files", "*.json"), ("All files", "*.*")], 2)

        # === Output Section ===
        output_frame = ttk.LabelFrame(main, text="Output", padding=15)
        output_frame.pack(fill=tk.X, pady=10)

        # Output directory
        row = ttk.Frame(output_frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Output Directory:", width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.output_dir, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(row, text="Browse", command=self._browse_output_dir).pack(side=tk.LEFT)

        # === Operations Section ===
        ops_frame = ttk.LabelFrame(main, text="Operations", padding=15)
        ops_frame.pack(fill=tk.X, pady=10)

        ops_row = ttk.Frame(ops_frame)
        ops_row.pack(fill=tk.X)

        ttk.Checkbutton(ops_row, text="🦴 Skeleton Overlay",
                        variable=self.render_skeleton).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(ops_row, text="🖐️ Mesh Overlay",
                        variable=self.render_mesh).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(ops_row, text="📊 Export JSON",
                        variable=self.export_json).pack(side=tk.LEFT, padx=10)

        # Device selection
        device_row = ttk.Frame(ops_frame)
        device_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(device_row, text="Device:").pack(side=tk.LEFT)
        ttk.Radiobutton(device_row, text="CUDA (GPU)", variable=self.device,
                        value="cuda").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(device_row, text="CPU", variable=self.device,
                        value="cpu").pack(side=tk.LEFT, padx=10)

        # === Progress Section ===
        progress_frame = ttk.Frame(main)
        progress_frame.pack(fill=tk.X, pady=20)

        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X)

        self.status_label = ttk.Label(progress_frame, text="Ready", foreground="#888")
        self.status_label.pack(pady=5)

        # === Buttons ===
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)

        self.run_btn = ttk.Button(btn_frame, text="▶️ Run Pipeline", command=self._run_pipeline)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="📂 Open Output", command=self._open_output).pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="❓ Help", command=self._show_help).pack(side=tk.RIGHT, padx=5)

        # === Log Section ===
        log_frame = ttk.LabelFrame(main, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.log_text = tk.Text(log_frame, height=8, bg="#0d1117", fg="#c9d1d9",
                                font=("Consolas", 10), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self._log("Haptica Hand Tracking Pipeline ready.")
        self._log("Select input files and click 'Run Pipeline' to start.")

    def _add_file_row(self, parent, label, var, filetypes, row):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label, width=18).pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Browse",
                   command=lambda: self._browse_file(var, filetypes)).pack(side=tk.LEFT)

    def _browse_file(self, var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _set_status(self, status):
        self.status_label.configure(text=status)

    def _run_pipeline(self):
        if self.is_running:
            return

        # Validate inputs
        if not self.params_path.get():
            messagebox.showerror("Error", "Please select a parameters file (.pth)")
            return

        if not any([self.render_skeleton.get(), self.render_mesh.get(), self.export_json.get()]):
            messagebox.showerror("Error", "Please select at least one operation")
            return

        if (self.render_skeleton.get() or self.render_mesh.get()) and not self.video_path.get():
            messagebox.showerror("Error", "Video file required for rendering")
            return

        # Start pipeline in background thread
        self.is_running = True
        self.run_btn.configure(state=tk.DISABLED)
        self.progress.start(10)
        self._set_status("Running pipeline...")

        thread = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        thread.start()

    def _run_pipeline_thread(self):
        try:
            from hand_pipeline import HandPipeline

            output_dir = Path(self.output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            self._log(f"\n{'='*50}")
            self._log(f"Starting pipeline...")
            self._log(f"Output: {output_dir}")

            # Initialize pipeline
            cameras = self.cameras_path.get() if self.cameras_path.get() else None
            pipeline = HandPipeline(
                self.params_path.get(),
                cameras,
                self.device.get()
            )

            # Export JSON
            if self.export_json.get():
                self._set_status("Exporting joint data...")
                self._log("\n📊 Exporting joint data...")
                pipeline.export_joints(str(output_dir / "joints.jsonl"))
                self._log("  ✓ joints.jsonl created")

            # Render skeleton
            if self.render_skeleton.get():
                self._set_status("Rendering skeleton...")
                self._log("\n🦴 Rendering skeleton overlay...")
                pipeline.render_skeleton(
                    self.video_path.get(),
                    str(output_dir / "skeleton.mp4")
                )
                self._log("  ✓ skeleton.mp4 created")

            # Render mesh
            if self.render_mesh.get():
                self._set_status("Rendering mesh...")
                self._log("\n🖐️ Rendering mesh overlay...")
                pipeline.render_mesh(
                    self.video_path.get(),
                    str(output_dir / "mesh.mp4")
                )
                self._log("  ✓ mesh.mp4 created")

            self._log(f"\n{'='*50}")
            self._log(f"✅ Pipeline complete!")
            self._set_status("Complete!")

        except Exception as e:
            self._log(f"\n❌ Error: {str(e)}")
            self._set_status("Error!")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.is_running = False
            self.root.after(0, lambda: self.run_btn.configure(state=tk.NORMAL))
            self.root.after(0, self.progress.stop)

    def _open_output(self):
        output_dir = self.output_dir.get()
        if os.path.exists(output_dir):
            if sys.platform == 'darwin':
                os.system(f'open "{output_dir}"')
            elif sys.platform == 'win32':
                os.startfile(output_dir)
            else:
                os.system(f'xdg-open "{output_dir}"')
        else:
            messagebox.showinfo("Info", f"Output directory doesn't exist yet:\n{output_dir}")

    def _show_help(self):
        help_text = """
Haptica Hand Tracking Pipeline

USAGE:
1. Select your input video (.mp4)
2. Select parameters file (smooth_fit_params.pth from Dyn-HaMR)
3. Optionally select cameras.json for intrinsics
4. Choose output directory
5. Select operations to run
6. Click 'Run Pipeline'

OPERATIONS:
• Skeleton Overlay: Renders 21-joint skeleton in Haptica teal
• Mesh Overlay: Renders full MANO hand mesh (slower)
• Export JSON: Outputs per-frame joint positions as JSONL

OUTPUT FILES:
• joints.jsonl - Per-frame 3D/2D joint positions
• skeleton.mp4 - Video with skeleton overlay
• mesh.mp4 - Video with mesh overlay

TIPS:
• Use CUDA (GPU) for faster processing
• Export JSON is fastest, mesh is slowest
• Check the log for progress and errors
        """
        messagebox.showinfo("Help", help_text)

    def run(self):
        self.root.mainloop()


def main():
    app = HandPipelineGUI()
    app.run()


if __name__ == '__main__':
    main()
