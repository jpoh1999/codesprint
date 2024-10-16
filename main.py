import sys
from App import *

sys.setrecursionlimit(5000)

# Run the app
if __name__ == "__main__":
    root = ctk.CTk()
    app = App(root)
    root.mainloop()



    
    