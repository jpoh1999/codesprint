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
        self.reset_model_form()    
        self.entry_width = 200  # Width for entries
        self.label_width = 200  # Width for labels
        self.logger = configure_logger("App_Logger", "app.log")

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

    def reset_model_form(self) :
        
        self.model_inputs = {}

        self.model_inputs =  {
            "stop_factor": ctk.IntVar(value=10000),
            "max_moves": ctk.IntVar(value=10),
            "depth": ctk.IntVar(value=5),
            "epochs": ctk.IntVar(value=1000)
        }
    
    def reset_container_form(self) :
        
        self.container_inputs = {}
        
        self.container_inputs = {
            "slots" : ctk.IntVar(value=5),
            "rows" : ctk.IntVar(value=10),
            "levels" : ctk.IntVar(value=6),
        }

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
        """ Function to open up generate dialog """
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Generate Containers")
        dialog.geometry("600x400")

        # Reset container inputs
        self.reset_container_form()

        # Add description at the top
        description = ("This feature allows you to generate a specified number "
                       "of containers for a model. You can input the number of "
                       "slots, rows, and levels you need.")
        ctk.CTkLabel(dialog, text=description, wraplength=350, font=ctk.CTkFont("Arial", 12, "normal")).pack(pady=10)
        
        for label, label_var in self.container_inputs.items() :
            self.create_label_entry(dialog, f"Number of {label}", label_var)
            
        # Create a frame for the containers row with fill and centered alignment
        containers_frame = ctk.CTkFrame(dialog)
        containers_frame.pack(pady=10, fill="x")
        
        self.containers_path = ctk.StringVar()
        container_row = ctk.CTkFrame(containers_frame)
        container_row.pack(fill="x")  # Fill horizontally and provide padding
        
        # Containers directory : 
        ctk.CTkLabel(container_row, text="Containers Directory: ", width=self.label_width, anchor="e").pack(side="left", padx=(10,10))
        ctk.CTkEntry(container_row, textvariable=self.containers_path, width=self.entry_width).pack(side="left", padx=(0, 10), pady=(5, 5))
        ctk.CTkButton(container_row, text="🔍", command=lambda: self.browse_dir(self.containers_path), width=40).pack(side="left")
       
        # Add the button beside the entry
        ctk.CTkButton(dialog, text="Generate", command=self.generate_containers).pack(pady=20)


    def open_preprocess_dialog(self):
        """ Function to open up preprocessing dialog """

        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Preprocess Data")
        dialog.geometry("600x600")
        
        # Add description at the top
        description = ("Preprocess existing container data by selecting the "
                       "directory containing slot profiles. This is necessary "
                       "before running the models.")
        ctk.CTkLabel(dialog, text=description, wraplength=350, font=ctk.CTkFont("Arial", 12, "normal")).pack(pady=10)

        preprocess_frame = ctk.CTkFrame(dialog)
        preprocess_frame.pack(pady=10, fill="x")

        # Containers parquet file
        container_row = ctk.CTkFrame(preprocess_frame)
        container_row.pack(fill="x")
        self.containers_file_path = ctk.StringVar()
        ctk.CTkLabel(container_row, text="Containers Data:", width=self.label_width, anchor="e").pack(side="left", pady=(10,10), padx=(10,10))
        ctk.CTkEntry(container_row, textvariable=self.containers_file_path, width=self.entry_width).pack(side="left", padx=(0, 10), pady=(5,5))
        ctk.CTkButton(container_row, text="🔍", command=lambda: self.browse_file(self.containers_file_path), width=40).pack(side="left")

        # Slot Layouts Directory
        slots_row = ctk.CTkFrame(preprocess_frame)
        slots_row.pack(fill="x")
        self.slot_profiles_var = ctk.StringVar()
        ctk.CTkLabel(slots_row, text="Select Slot Profiles Directory:", width=self.label_width, anchor="e").pack(side="left", pady=(10,10), padx=(10,10))
        ctk.CTkEntry(slots_row, textvariable=self.slot_profiles_var, width=self.entry_width).pack(side="left", padx=(0, 10), pady=(5,5))
        ctk.CTkButton(slots_row, text="🔍", command=lambda: self.browse_dir(self.slot_profiles_var), width=40).pack(side="left")

        ctk.CTkButton(dialog, text="Generate Slot Profiles", command=self.preprocess_data).pack(pady=20)

    def create_label_entry(self, frame, label_text, variable) -> None:
        """ 
            Helper function to create label entries 
            
            Args :
                frame (Frame) : the main frame
                label_text (str) : the str of the label
                variable (any) : tk variable
            
            Returns :
                None
        
        """
        entry_frame = ctk.CTkFrame(frame, fg_color=None)  # Frame for holding label and entry side by side
        entry_frame.pack(pady=5, fill="x")  # Pack with left alignment

        # Create label
        ctk.CTkLabel(entry_frame, text=label_text + ":", width=self.label_width, anchor="e").pack(side="left", padx=(10, 10))

        # Create entry
        ctk.CTkEntry(entry_frame, textvariable=variable, width=self.entry_width).pack(side="left", padx=(0,10))
        

    def model_input_form(self, dialog) :
        """
            Generic Model Input Form Template

            Args :
                - dialog : the dialog pop-up element
        """
        # Set it to a default one
        self.reset_model_form()

        # The model parameters to be decided by user
        input_frame = ctk.CTkFrame(dialog)
        input_frame.pack(pady=10, fill="x")
        self.input_path = ctk.StringVar()

        # Frame for input label and entry
        input_row = ctk.CTkFrame(input_frame)
        input_row.pack(fill="x")
        ctk.CTkLabel(input_row, text="Select Input Directory:", width=self.label_width, anchor="e").pack(side="left", padx=(10, 10))
        ctk.CTkEntry(input_row, textvariable=self.input_path, width=self.entry_width).pack(side="left", padx=(0, 10), pady=(5,5))
        ctk.CTkButton(input_row, text="🔍", command=lambda: self.browse_dir(self.input_path), width=40).pack(side="left")

        # Create the inputs using a loop
        for param_name, param_var in self.model_inputs.items():
            self.create_label_entry(dialog, param_name, param_var)
           
        # Set the output_path
        output_frame = ctk.CTkFrame(dialog)
        output_frame.pack(pady=10, fill="x")
        self.output_path = ctk.StringVar()

        # Frame for output label and entry
        output_row = ctk.CTkFrame(output_frame)
        output_row.pack(fill="x")
        ctk.CTkLabel(output_row, text="Select Output Directory:", width=self.label_width, anchor="e").pack(side="left", padx=(10, 10))
        ctk.CTkEntry(output_row, textvariable=self.output_path, width=self.entry_width).pack(side="left", padx=(0, 10), pady=(5,5))
        ctk.CTkButton(output_row, text="🔍", command=lambda: self.browse_dir(self.output_path), width=40).pack(side="left")

    def open_greedy_dialog(self):
        """ Function to open up dialog for greedy model """

        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Run Greedy Model")
        dialog.geometry("500x600")
        

        # Add description at the top
        description = ("Run the Greedy Model on the preprocessed container data. "
                       "Select the output directory to save the results.")
        ctk.CTkLabel(dialog, text=description, wraplength=350, font=ctk.CTkFont("Arial", 12, "normal")).pack(pady=10)

        self.model_input_form(dialog)

        # Button on submit : run the model() : validates all inputs first
        ctk.CTkButton(dialog, text="Generate", command=self.run_model(random=False)).pack(pady=20)


    def open_random_dialog(self):
        """ Function to open up dialog for random model """

        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Run Random Model")
        dialog.geometry("500x600")
    
        # Add description at the top
        description = ("Run the Random Model on the preprocessed container data. "
                       "Select the output directory to save the results.")
        ctk.CTkLabel(dialog, text=description, wraplength=350, font=ctk.CTkFont("Arial", 12, "normal")).pack(pady=10)

        self.model_input_form(dialog)

        # Button on submit : run the model() : validates all inputs first
        ctk.CTkButton(dialog, text="Generate", command=lambda : self.run_model(random=True)).pack(pady=20)

    def generate_containers(self):

        print("Start generating")
        container_input_vals = {k : v.get() for k,v in self.container_inputs.items()}

        df = ContainerGenerator().generate(**container_input_vals)
        df.to_parquet(f"{self.containers_path.get()}/containers.parquet")

        print("End generation")
    
    def preprocess_data(self): 
        
        print("Preprocessing...")
        reader = SlotProfileReader()

        reader.read_and_process(self.containers_file_path.get(),
                                self.output_path.get())
        print("End Preprocessing...")

    def browse_slot_profiles(self):
        
        pass
    
    def browse_dir(self, var):
        """
            Helper function to browse directory

            Args :
                var : tkinter_input var
        """
        dir_path = filedialog.askdirectory()
        if dir_path:
            var.set(dir_path)

    def browse_file(self, var):
        """
            Helper function to browse file 
            
            Args :
                var : tkinter_input var
        """
        filename = filedialog.askopenfilename(
            title="Select a File",
            filetypes=(("Text Files", "*.txt"), ("Parquet Files", "*.parquet"), ("Csv Files", "*.csv"), ("All Files", "*.*"))  # Adjust file types as needed
        )
        if filename:
            var.set(filename)

    def run_model(self, random) :
        """
            Helper function to run model

            Args :
                random (bool) : to indicate greedy/random model

        """
        os.makedirs(self.input_path.get(), exist_ok=True)
        os.makedirs(f"{self.output_path.get()}/scores/", exist_ok=True)

        model_name = "Random model" if random else "Greedy Model"
        log_file_name = "random_model.log" if random else "greedy_model.log"
        logger = configure_logger(model_name, log_file_name)

        model_input_values = {k : v.get() for k,v in self.model_inputs.items()}
        
        model = Model(logger=logger, random=random, **model_input_values)
        scores_path = f"{self.output_path.get()}/scores/sol.csv"
        
        # Read the CSV file
        file_list = [f for f in os.listdir(self.input_path.get()) if f.endswith('.csv')]
        scores_list = []
        
        for file in file_list:
            file_path = os.path.join(self.input_path.get(), file)
            df = pd.read_csv(file_path, header=0, index_col=0, na_filter=False)

            # Convert the index to int
            df.index = df.index.astype(int)

            # Convert the columns to int
            df.columns = df.columns.astype(int)

            slot_name = file.split(".")[0]
            
            initial_score, final_score, reduction = model.solve(df, slot_name, self.output_path.get())
        
        scores_list.append([slot_name, initial_score, final_score, reduction])
        
        scores_df = pd.DataFrame(scores_list, columns=["slot", "initial_score", "final_score", "reduction"])
        scores_df.to_csv(scores_path)
        
