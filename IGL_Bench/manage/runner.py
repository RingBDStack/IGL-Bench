import importlib
import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
class Manager:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.solver = self.initialize_solver()

    def initialize_solver(self):
        solver_name = f"{self.config.algorithm}_{self.config.task}_solver"
        module_path = f"IGL_Bench.algorithm.{self.config.algorithm}.solver"
        try:
            module = importlib.import_module(module_path)
            solver_class = getattr(module, solver_name)
            return solver_class(self.config, self.dataset)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Failed to import solver {solver_name} from {module_path}: {e}")

    def run(self, num_runs=1, random_seed=1):
        all_acc = []
        all_bacc = []
        all_mf1 = []
        all_roc = []
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs} for algorithm {self.solver.__class__.__name__}")
            set_seed(random_seed+run)
            self.solver.train()
            acc, bacc, mf1, roc = self.solver.test()
            
            all_acc.append(acc)
            all_bacc.append(bacc)
            all_mf1.append(mf1)
            all_roc.append(roc)
        
        avg_acc = np.mean(all_acc) * 100
        std_acc = np.std(all_acc) * 100
        avg_bacc = np.mean(all_bacc) * 100
        std_bacc = np.std(all_bacc) * 100
        avg_mf1 = np.mean(all_mf1) * 100
        std_mf1 = np.std(all_mf1) * 100
        avg_roc = np.mean(all_roc) * 100
        std_roc = np.std(all_roc) * 100
        
        self.print_results(avg_acc, std_acc, avg_bacc, std_bacc, avg_mf1, std_mf1, avg_roc, std_roc)
    
    def print_results(self, avg_acc, std_acc, avg_bacc, std_bacc, avg_mf1, std_mf1, avg_roc, std_roc):
        print(f"\nTest results for {self.config.algorithm} (averaged across runs):")
        print("+----------------------+------------------+------------------+")
        print("| {:<20} | {:>8.2f} ± {:>8.2f} |".format("Accuracy", avg_acc, std_acc))
        print("| {:<20} | {:>8.2f} ± {:>8.2f} |".format("Balanced Accuracy", avg_bacc, std_bacc))
        print("| {:<20} | {:>8.2f} ± {:>8.2f} |".format("Macro F1", avg_mf1, std_mf1))
        print("| {:<20} | {:>8.2f} ± {:>8.2f} |".format("ROC-AUC", avg_roc, std_roc))
        print("+----------------------+------------------+------------------+")
