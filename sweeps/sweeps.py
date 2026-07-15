

# --- Small Datasets (Full-Batch Training) ---
# batch_size = -1 signals the loader to use the full graph.
metadata_chameleon = {
    "in_channels": {"values": [2325]}, 
    "out_channels": {"values": [3]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [-1]},
}
metadata_squirrel  = {
    "in_channels": {"values": [2089]}, 
    "out_channels": {"values": [3]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [-1]},
}
metadata_cora      = {
    "in_channels": {"values": [1433]}, 
    "out_channels": {"values": [3]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [-1]},
}


metadata_patents   = {
    "in_channels": {"values": [269]},  
    "out_channels": {"values": [3]},
    "batch_size": {"values": [16384]},
    "num_neighbors": {"values": [8]},
}
metadata_arxiv     = {
    "in_channels": {"values": [128]},  
    "out_channels": {"values": [3]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [10]},
}
metadata_reddit2   = {
    "in_channels": {"values": [602]},  
    "out_channels": {"values": [30]},
    "batch_size": {"values": [2**10]},
    "num_neighbors": {"values": [8]},
}
metadata_coauthor  = {
    "in_channels": {"values": [6805]}, 
    "out_channels": {"values": [11]},
    "batch_size": {"values": [-1]},
    "num_neighbors": {"values": [10]},
}
metadata_amazon_ratings = {
    "in_channels":  {"values": [300]},  
    "out_channels": {"values": [3]},    
    "batch_size":   {"values": [-1]},   
    "num_neighbors":{"values": [10]},
}
metadata_roman_empire = {
    "in_channels":  {"values": [300]},  
    "out_channels": {"values": [13]},   # ID classes = {5..17}
    "batch_size":   {"values": [-1]},   
    "num_neighbors":{"values": [10]},
}


sweep_vanilla = {
    "method": "bayes",
    "metric": {"name": "val_auroc", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128]},
        "num_layers": {"values": [2, 3]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [30]},
        "monitor": {"values": ["val_auroc"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    }
}

sweep_ensemble = {
    "method": "grid", 
    "metric": {
        "name": "val_auroc_EU_credal", 
        "goal": "maximize"
    },
    "parameters": {
        "M": {"values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
    },
}


sweep_credal = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128]},
        "num_layers": {"values": [2, 3]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [10]},
        "monitor": {"values": ["val_auroc_EU"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    },
}


sweep_credal_LJ = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU", 
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64, 128]},
        "num_layers": {"values": [2, 3]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [10]},
        "monitor": {"values": ["val_auroc_EU"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    },
}


sweep_credal_LJ_dual_head = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU",
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
        "hidden_channels": {"values": [64]},
        "num_layers": {"values": [2]},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "lambda_cls": {"distribution": "uniform", "min": 0.1, "max": 2.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [20]},
        "monitor": {"values": ["val_auroc_EU"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    },
}


sweep_credal_LJ_dual_head_detached = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU",
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
        "hidden_channels": {"values": [64]},
        "num_layers": {"values": [2]},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "lambda_cls": {"distribution": "uniform", "min": 0.1, "max": 2.0},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [20]},
        "monitor": {"values": ["val_auroc_EU"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    },
}


sweep_credal_LJ_dual_head_constrained = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU",
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "hidden_channels": {"values": [64]},
        "num_layers": {"values": [2]},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "lambda_cls": {"distribution": "uniform", "min": 0.5, "max": 2.0},
        "lambda_cons": {"values": [0.01, 0.05, 0.1, 0.25]},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [10]},
        "monitor": {"values": ["val_auroc_EU"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    },
}


sweep_credal_LJ_dual_head_constrained_detached = {
    "method": "bayes",
    "metric": {
        "name": "val_auroc_EU",
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
        "hidden_channels": {"values": [64]},
        "num_layers": {"values": [2]},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "lambda_cls": {"distribution": "uniform", "min": 0.1, "max": 2.0},
        "lambda_cons": {"values": [0.01, 0.05, 0.1, 0.25]},
        "gnn_type": {"values": ["GCN", "SAGE"]},
        "patience": {"values": [20]},
        "monitor": {"values": ["val_auroc_EU"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    },
}


sweep_mahalanobis = {
    "method": "grid",  
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "noise_magnitude": {
            "values": [0.0, 0.001, 0.005, 0.01, 0.05]
        }
    },
}



sweep_knn = {
    "method": "grid",
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "k": {
            "values": [5, 10, 20, 50, 100, 200]
        }
    },
}

sweep_energy = {
    "method": "grid",
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "dummy_run_id": {"values": [1]} # A dummy parameter to ensure the agent runs exactly once.
    },
}

sweep_odin = {
    "method": "grid",
    "metric": {
        "name": "val_auroc",
        "goal": "maximize",
    },
    "parameters": {
        "temperature": {"values": [1.0, 10.0, 100.0, 1000.0]},
        "noise_magnitude": {"values": [0.0, 0.001, 0.005, 0.01, 0.05]},
    },
}

sweep_knn_LJ = {
    "method": "grid",
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "k": {
            "values": [5, 10, 20, 50, 100, 200]
        }
    },
}

sweep_gnnsafe = {
    "method": "grid",
    "metric": {
        "name": "val_auroc", 
        "goal": "maximize"
    },
    "parameters": {
        "K": {"values": [1, 2, 4, 8, 16]},
        "alpha": {"values": [0.1, 0.3, 0.5, 0.7, 0.9]}
    },
}

sweep_gebm = {
    "method": "grid",
    "metric": {"name": "val_auroc", "goal": "maximize"},
    "parameters": {
        # keep consistent with your other sweeps
        "seed": {"values": [0, 1, 2, 3, 4]},
    },
}

sweep_frozen = {
    "method": "bayes",
    "metric": {"name": "val_auroc_EU", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "seed": {"values": [0, 1, 2, 3, 4]},
        "delta": {"distribution": "uniform", "min": 0.5, "max": 1.0},
        "patience": {"values": [10]},
        "monitor": {"values": ["val_auroc_EU"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    },
}

sweep_cagcn = {
    "method": "bayes",
    "metric": {"name": "val_auroc", "goal": "maximize"},
    "parameters": {
        "seed": {"values": [0,1,2,3,4]},
        "lambda_cal": {"values": [0.25, 0.5, 0.75]},
        "calib_hidden": {"values": [8,16,32]},
        "calib_layers": {"values": [1,2]},
        "lr": {"distribution": "uniform", "min": 1e-5, "max": 1e-1},
        "weight_decay": {"distribution": "uniform", "min": 1e-7, "max": 1e-1},
        "max_epochs": {"values": [200]},
        "patience": {"values": [10]},
        "monitor": {"values": ["val_auroc"]},
        "mode": {"values": ["max"]},
        "num_sanity_val_steps": {"values": [0]},
    },
}

sweep_graph_esn = {
    "method": "bayes",
    "metric": {"name": "val_auroc_TU_credal", "goal": "maximize"},
    "parameters": {
        "hidden_channels": {"distribution": "int_uniform", "min": 64, "max": 256},
        "num_layers": {"distribution": "int_uniform", "min": 1, "max": 3},
        "spectral_radius": {"distribution": "uniform", "min": 0.5, "max": 0.9},
        "input_scaling": {"distribution": "log_uniform_values", "min": 1e-3, "max": 1.0},
        "leakage": {"distribution": "uniform", "min": 0.3, "max": 1.0},
        "num_reservoirs": {"distribution": "int_uniform", "min": 10, "max": 100},
        "readout_regularization": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e2},
        "bias": {"values": [False, True]},
        "pooling": {"values": ["none"]},
        "fully": {"values": [False, True]},
        "max_iterations": {"distribution": "int_uniform", "min": 50, "max": 150},
        "epsilon": {"distribution": "log_uniform_values", "min": 1e-7, "max": 1e-4},
    },
}
