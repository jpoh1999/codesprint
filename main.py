import sys
from App import *

sys.setrecursionlimit(5000)

# Run the app
if __name__ == "__main__":
    root = ctk.CTk()
    app = App(root)
    root.mainloop()

# if __name__ == "__main__":

#     # These two processes need to be done in order.
#     generate_containers()    
#     preprocess_containers_data()
    
#     # These other two processes can be done in parallel execution.
#     greedy_process = multiprocessing.Process(target=run_greedy_model)
#     random_process = multiprocessing.Process(target=run_random_model)
    
#     # Start both processes
#     greedy_process.start()
#     random_process.start()

#     # Wait for both processes to complete
#     greedy_process.join()
#     random_process.join()

#     print("Finished running models")
    
    # test_solution()


    
    