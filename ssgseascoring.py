import pandas as pd
import numpy as np
import pathlib as path
import gseapy as gp 
df = pd.read_csv("TCGA-LUAD_expression_matrix_log2.tsv", sep = '\t').set_index("gene")

MATERIAL_MODULES = {
    "fibrillar_collagen": [
        "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2",
        "COL11A1", "COL12A1", "COL14A1",
        "ITGA1", "ITGA2", "ITGA10", "ITGA11", "ITGB1"
    ],

    "collagen_IV": [
        "COL4A1", "COL4A2", "COL4A3",
        "COL4A4", "COL4A5", "COL4A6",
        "ITGA1", "ITGA2", "ITGA3", "ITGB1"
    ],

    "collagen_VI": [
        "COL6A1", "COL6A2", "COL6A3",
        "ITGA1", "ITGA2", "ITGA3", "ITGB1"
    ],

    "fibronectin": [
        "FN1",
        "ITGA5", "ITGAV",
        "ITGB1", "ITGB3", "ITGB5"
    ],

    "laminin": [
        "LAMA3", "LAMA4", "LAMA5",
        "LAMB1", "LAMB2", "LAMB3",
        "LAMC1", "LAMC2",
        "ITGA3", "ITGA6", "ITGB1", "ITGB4"
    ],

    "vitronectin": [
        "VTN",
        "ITGAV", "ITGB1", "ITGB3", "ITGB5"
    ],

    "osteopontin": [
        "SPP1",
        "ITGAV", "ITGA9",
        "ITGB1", "ITGB3", "ITGB5"
    ],

    "tenascin_C": [
        "TNC",
        "ITGA9", "ITGAV",
        "ITGB1", "ITGB3", "ITGB6"
    ],

    "thrombospondin": [
        "THBS1", "THBS2",
        "ITGAV", "ITGB1", "ITGB3"
    ],
}

ssgsea = [] 

ssgsea_results = gp.ssgsea(
    data=df,
    gene_sets=MATERIAL_MODULES,
    outdir=None,             # Set a path string here if you want to save csv/plots to disk
    sample_norm_method='rank', # Customizes enrichment statistic weighting ('rank' is standard)
    permutation_num=0,  
    min_size = 3, # ssGSEA does not require permutations for single sample scores
    no_plot=True             # Skips saving plot files to save memory/time
)
ssgsea_long = ssgsea_results.res2d
ssgsea_long.to_csv("ssgsea_long.csv", index=False) #original format.
ssgsea_wide = ssgsea_long.pivot(
    index="Name",
    columns="Term",
    values="NES",
).reset_index()

ssgsea_wide.to_csv("ssgsea_scores.csv", index=False) #need to convert it to wide
