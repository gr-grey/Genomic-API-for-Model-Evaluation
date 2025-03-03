## Variant Scoring

This tutorial describes how to predict variant effect scores for a small set of SNVs defined in a .vcf file. For examples showcasing variant effect prediction at a larger scale (e.g. fine-mapped eQTL classification benchmarks), we refer the user to the [borzoi-paper respository](https://github.com/calico/borzoi-paper/tree/main). This example uses the pre-trained, published Borzoi model to predict variant effects. To download this model, run the script 'download_models.sh' in the 'borzoi' root folder.

First, to calculate **gene-specific expression** scores, run the script 'score_expr_sed.sh'. Two different statistics are computed: (1) logSED (gene expression log fold change), and (2) logD2 (bin-level L2 norm across the coverage profile intersecting the exons of the gene).
```sh
conda activate borzoi_py310
cd ~/borzoi/tutorials/legacy/score_variants
./score_expr_sed.sh
```

To calculate **gene-agnostic expression** scores, run the script 'score_expr_sad.sh'. One statistic is computed: logD2 (bin-level L2 norm across the entire predicted coverage track).
```sh
./score_expr_sad.sh
```

To calculate **gene-specific polyadenylation** scores, run the script 'score_polya.sh'. One statistic is computed: COVR (3' coverage ratio across pA junctions of the target gene).
```sh
./score_polya.sh
```

To calculate **gene-specific splicing** scores, run the script 'score_splice.sh'. One statistic is computed: nDi (normalized maximum absolute difference in coverage bins across the target gene span).
```sh
./score_splice.sh
```

Finally, the jupyter notebook 'run_variant_scripts.ipynb' is provided for convenience to execute all above scripts. The notebook also exemplifies how to navigate the variant prediction hdf5 files and print some example scores.

*Notes*:
- The legacy data transforms are activated in all above .sh scripts with the flag '-u'.

