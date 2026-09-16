# -*- coding: utf-8 -*-
"""
Hotspot residue configuration
"""

# Internal Contact Configuration of the Binder Service
CON_OPT = {
    "num": 4,
    "num_pos": float("inf"),
    "cutoff": 14.0,
    "seqsep": 9,
    "binary": False
}


# Interface contact configuration
I_CON_OPT = {
    "num": 2,
    "num_pos": float("inf"),
    "cutoff": 10.0,
    "seqsep": 0,
    "binary": False
} 

# Negative Hotspot residue configuration
NEG_CON_OPT = {
    "num": 2,
    "num_pos": float("inf"),
    "cutoff": 10.0,
    "seqsep": 0,
    "binary": False
} 
 
# Negative framework residue configuration
NEG_FRAMEWORK_CON_OPT = {
    "num": 4,
 	"num_pos": 15, 
 	"cutoff": 10.0,
 	"seqsep": 0,
 	"binary": False
}

PARATOPE_CDR_OPT = {
    "num": 10,
    "num_pos": float("inf"),# Average all residues that meet the mask_1d condition.
    "cutoff": 15,          # Distance threshold
    "seqsep": 0,
    "binary": False
}

# Framework, the same configuration is usually used, but independent configuration is also possible
PARATOPE_FW_OPT = {
    "num": 10,
    "num_pos": float("inf"),
    "cutoff": 20,
    "seqsep": 0,
    "binary": False
}

# Paratope Loss Hyperparameters
PARATOPE_SCALAR_OPT = {
    "lambda_offset": 2.0,   # λ Parameters
    "eps": 1e-8,             # Numerical stability term
    "rebalance_val": 1.0,    # Rebalancing factor
}