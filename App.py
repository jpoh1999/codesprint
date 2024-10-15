import os
import sys
import pandas as pd
import multiprocessing
from libs.utils import load_config_file, configure_logger
from libs.containers_generator import ContainerGenerator
from libs.slot_profile_reader import SlotProfileReader
from model.model import Model
from libs.constants import CONFIG_FILE_PATH
import customtkinter as ctk
from tkinter import filedialog, Toplevel

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Container Data Generator & Model Solver")

        # Make window resizeable
        self.root.geometry("800x600")  # Starting size
        self.root.minsize(600, 400)    # Minimum size

        # Set blue-black theme like VSCode
        ctk.set_appearance_mode("dark")  # dark mode
        ctk.set_default_color_theme("blue")  # blue-black accent

        # Create header
        self.create_header()

        # Create buttons for functionality
        self.create_functionality_buttons()

    def create_header(self):
        """Create and place the header with logo and label side by side, centered."""
        header_frame = ctk.CTkFrame(self.root, fg_color='#1e1e1e')
        header_frame.pack(fill="x", pady=30)

        # Centering frame for the logo and header
        center_frame = ctk.CTkFrame(header_frame, fg_color='#1e1e1e')
        center_frame.pack(expand=True, fill="both")

        # Frame to hold logo and header title side by side
        logo_and_title_frame = ctk.CTkFrame(center_frame, fg_color='#1e1e1e')
        logo_and_title_frame.pack(side="top")  # Keep it at the top for centering

        # Logo and Header label side by side
        logo = ctk.CTkLabel(logo_and_title_frame, text="🛠️", font=ctk.CTkFont("Arial", 30))
        logo.pack(side="left", padx=10)  # Use side="left" to keep them together horizontally

        header_label = ctk.CTkLabel(logo_and_title_frame, text="Container Data & Model Solver", font=ctk.CTkFont("Arial", 24, "bold"))
        header_label.pack(side="left")

        # Optional subheader below the logo and header label
        subheader_label = ctk.CTkLabel(center_frame, text="Solve & Generate with Ease", font=ctk.CTkFont("Arial", 16, "normal"))
        subheader_label.pack(pady=(5, 20))  # Add padding for separation

    def create_functionality_buttons(self):
        """Create buttons for the 4 main functionalities, making them responsive to window size changes."""
        button_frame = ctk.CTkFrame(self.root)
        button_frame.pack(pady=50, fill="both", expand=True)  # Allow the frame to grow/shrink

        buttons = [
            ("Generate Containers", self.open_generate_dialog),
            ("Preprocess Containers", self.open_preprocess_dialog),
            ("Run Greedy Model", self.open_greedy_dialog),
            ("Run Random Model", self.open_random_dialog)
        ]

        # Helper function to create a responsive button
        def create_responsive_button(text, command, row, col):
            button = ctk.CTkButton(
                button_frame,
                text=text,
                command=command,
                font=ctk.CTkFont("Arial", 14),
                height=60,
                width=250,
                corner_radius=15,
                fg_color="#00509E",
                hover_color="#003D73",
                text_color="white",
                border_width=2,
                border_color="black"
            )
            button.grid(row=row, column=col, padx=20, pady=20, sticky="nsew")

        # Configure buttons in a 2x2 grid with padding and rounded corners
        for i, (text, command) in enumerate(buttons):
            create_responsive_button(text, command, row=i // 2, col=i % 2)

        # Make sure columns and rows expand when window is resized
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_rowconfigure(0, weight=1)
        button_frame.grid_rowconfigure(1, weight=1)

    def open_generate_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Generate Containers")
        dialog.geometry("400x300")

        # Add description at the top
        description = ("This feature allows you to generate a specified number "
                       "of containers for a model. You can input the number of "
                       "slots, rows, and levels you need.")
        ctk.CTkLabel(dialog, text=description, wraplength=350, font=ctk.CTkFont("Arial", 12, "normal")).pack(pady=10)

        ctk.CTkLabel(dialog, text="Number of Slots:").pack(pady=10)
        self.num_slots_var = ctk.IntVar(value=5)
        ctk.CTkEntry(dialog, textvariable=self.num_slots_var, width=100).pack()

        ctk.CTkLabel(dialog, text="Number of Rows:").pack(pady=10)
        self.num_rows_var = ctk.IntVar(value=10)
        ctk.CTkEntry(dialog, textvariable=self.num_rows_var, width=100).pack()

        ctk.CTkLabel(dialog, text="Number of Levels:").pack(pady=10)
        self.num_levels_var = ctk.IntVar(value=6)
        ctk.CTkEntry(dialog, textvariable=self.num_levels_var, width=100).pack()

        ctk.CTkButton(dialog, text="Generate", command=self.generate_containers).pack(pady=20)

    def open_preprocess_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Preprocess Data")
        dialog.geometry("400x200")

        # Add description at the top
        description = ("Preprocess existing container data by selecting the "
                       "directory containing slot profiles. This is necessary "
                       "before running the models.")
        ctk.CTkLabel(dialog, text=description, wraplength=350, font=ctk.CTkFont("Arial", 12, "normal")).pack(pady=10)

        ctk.CTkLabel(dialog, text="Select Slot Profiles Directory:").pack(pady=10)
        self.slot_profiles_var = ctk.StringVar()
        ctk.CTkEntry(dialog, textvariable=self.slot_profiles_var).pack()

        ctk.CTkButton(dialog, text="Browse", command=self.browse_slot_profiles).pack(pady=20)

    def open_greedy_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Run Greedy Model")
        dialog.geometry("400x200")

        # Add description at the top
        description = ("Run the Greedy Model on the preprocessed container data. "
                       "Select the output directory to save the results.")
        ctk.CTkLabel(dialog, text=description, wraplength=350, font=ctk.CTkFont("Arial", 12, "normal")).pack(pady=10)

        ctk.CTkLabel(dialog, text="Select Output Directory:").pack(pady=10)
        self.greedy_output_var = ctk.StringVar()
        ctk.CTkEntry(dialog, textvariable=self.greedy_output_var).pack()

        ctk.CTkButton(dialog, text="Browse", command=lambda: self.browse_dir(self.greedy_output_var)).pack(pady=20)

    def open_random_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Run Random Model")
        dialog.geometry("400x200")

        # Add description at the top
        description = ("Run the Random Model on the preprocessed container data. "
                       "Select the output directory to save the results.")
        ctk.CTkLabel(dialog, text=description, wraplength=350, font=ctk.CTkFont("Arial", 12, "normal")).pack(pady=10)

        ctk.CTkLabel(dialog, text="Select Output Directory:").pack(pady=10)
        self.random_output_var = ctk.StringVar()
        ctk.CTkEntry(dialog, textvariable=self.random_output_var).pack()

        ctk.CTkButton(dialog, text="Browse", command=lambda: self.browse_dir(self.random_output_var)).pack(pady=20)

    def generate_containers(self):
        # Container generation logic...
        pass

    def browse_slot_profiles(self):
        # Logic to browse directory for slot profiles
        pass

    def browse_dir(self, var):
        dir_path = filedialog.askdirectory()
        if dir_path:
            var.set(dir_path)