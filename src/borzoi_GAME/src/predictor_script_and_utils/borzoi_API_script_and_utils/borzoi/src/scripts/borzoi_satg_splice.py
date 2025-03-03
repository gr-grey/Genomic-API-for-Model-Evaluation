#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser

import gc
import json
import os

import h5py
import numpy as np
import pandas as pd
import pysam

from baskerville.dataset import targets_prep_strand
from baskerville import dna as dna_io
from baskerville import gene as bgene
from baskerville import seqnn

"""
borzoi_satg_splice.py

Perform a gradient saliency analysis for genes specified in a GTF file (splice-centric).
"""

################################################################################
# main
# ###############################################################################
def main():
    usage = "usage: %prog [options] <params> <model> <gene_gtf>"
    parser = OptionParser(usage)
    parser.add_option(
        "--fa",
        dest="genome_fasta",
        default="%s/assembly/ucsc/hg38.fa" % os.environ.get('BORZOI_HG38', 'hg38'),
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="satg_out",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "-f",
        dest="folds",
        default="0",
        type="str",
        help="Model folds to use in ensemble (comma-separated list) [Default: %default]",
    )
    parser.add_option(
        '-c',
        dest='crosses',
        default="0",
        type="str",
        help='Model crosses (replicates) to use in ensemble (comma-separated list) [Default:%default]',
    )
    parser.add_option(
        "--head",
        dest="head_i",
        default=0,
        type="int",
        help="Model head index [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--span",
        dest="span",
        default=False,
        action="store_true",
        help="Aggregate entire gene span [Default: %default]",
    )
    parser.add_option(
        "--clip_soft",
        dest="clip_soft",
        default=None,
        type="float",
        help="Model clip_soft setting [Default: %default]",
    )
    parser.add_option(
        "--track_scale",
        dest="track_scale",
        default=0.02,
        type="float",
        help="Target transform scale [Default: %default]",
    )
    parser.add_option(
        "--track_transform",
        dest="track_transform",
        default=0.75,
        type="float",
        help="Target transform exponent [Default: %default]",
    )
    parser.add_option(
        "--untransform_old",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Run gradients with old version of inverse transforms [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "-s",
        dest="splice_gff",
        default="%s/genes/gencode41/gencode41_basic_protein_splice.gff"
        % os.environ.get('BORZOI_HG38', 'hg38'),
        help="Splice site annotation [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_folder = args[1]
        genes_gtf_file = args[2]
    else:
        parser.error("Must provide parameter file, model folder and GTF file")

    if not os.path.isdir(options.out_dir):
        os.makedirs(options.out_dir, exist_ok=True)

    options.folds = [int(fold) for fold in options.folds.split(",")]
    options.crosses = [int(cross) for cross in options.crosses.split(",")]
    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    seq_len = params_model["seq_length"]

    if options.targets_file is None:
        parser.error("Must provide targets table to properly handle strands.")
    else:
        targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # prep strand
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = np.array(
        [orig_new_index[ti] for ti in targets_df.strand_pair]
    )
    targets_strand_df = targets_prep_strand(targets_df)
    num_targets = 1

    #################################################################
    # load first model fold to get parameters

    seqnn_model = seqnn.SeqNN(params_model)
    
    model_path = model_folder + "/f" + str(options.folds[0]) + "c0/train/model" + str(options.head_i) + "_best.h5"
    if not os.path.isfile(model_path) :
        model_path = model_folder + "/f" + str(options.folds[0]) + "c0/train/model_best.h5"
    
    seqnn_model.restore(
        model_path,
        options.head_i
    )
    seqnn_model.build_slice(targets_df.index, False)
    # seqnn_model.build_ensemble(options.rc, options.shifts)

    model_stride = seqnn_model.model_strides[0]
    model_crop = seqnn_model.target_crops[0]
    target_length = seqnn_model.target_lengths[0]

    #################################################################
    # read genes

    # parse GTF
    transcriptome = bgene.Transcriptome(genes_gtf_file)

    # order valid genes
    genome_open = pysam.Fastafile(options.genome_fasta)
    gene_list = sorted(transcriptome.genes.keys())
    num_genes = len(gene_list)

    #################################################################
    # setup output

    min_start = -model_stride * model_crop

    # choose gene sequences
    genes_chr = []
    genes_start = []
    genes_end = []
    genes_strand = []
    for gene_id in gene_list:
        gene = transcriptome.genes[gene_id]
        genes_chr.append(gene.chrom)
        genes_strand.append(gene.strand)

        gene_midpoint = gene.midpoint()
        gene_start = max(min_start, gene_midpoint - seq_len // 2)
        gene_end = gene_start + seq_len
        genes_start.append(gene_start)
        genes_end.append(gene_end)

    #######################################################
    # load splice site annotation

    splice_df = (
        pd.read_csv(
            options.splice_gff,
            sep="\t",
            names=[
                "Chromosome",
                "havana_str",
                "feature",
                "Start",
                "End",
                "feat1",
                "Strand",
                "feat2",
                "id_str",
            ],
            usecols=["Chromosome", "Start", "End", "feature", "feat1", "Strand"],
        )[["Chromosome", "Start", "End", "feature", "feat1", "Strand"]]
        .drop_duplicates(subset=["Chromosome", "Start", "Strand"], keep="first")
        .copy()
        .reset_index(drop=True)
    )

    #################################################################
    # predict scores, write output

    buffer_size = 1024

    print("clip_soft = " + str(options.clip_soft))

    print("n genes = " + str(len(genes_chr)))

    # loop over folds
    for fold_ix in options.folds:
        for cross_ix in options.crosses:
            
            print("-- fold = f" + str(fold_ix) + "c" + str(cross_ix) + " --")

            # (re-)initialize HDF5
            scores_h5_file = "%s/scores_f%dc%d.h5" % (options.out_dir, fold_ix, cross_ix)
            if os.path.isfile(scores_h5_file):
                os.remove(scores_h5_file)
            scores_h5 = h5py.File(scores_h5_file, "w")
            scores_h5.create_dataset("seqs", dtype="bool", shape=(num_genes, seq_len, 4))
            scores_h5.create_dataset(
                "grads", dtype="float16", shape=(num_genes, seq_len, 4, num_targets)
            )
            scores_h5.create_dataset("gene", data=np.array(gene_list, dtype="S"))
            scores_h5.create_dataset("chr", data=np.array(genes_chr, dtype="S"))
            scores_h5.create_dataset("start", data=np.array(genes_start))
            scores_h5.create_dataset("end", data=np.array(genes_end))
            scores_h5.create_dataset("strand", data=np.array(genes_strand, dtype="S"))

            # load model fold
            seqnn_model = seqnn.SeqNN(params_model)

            model_path = model_folder + "/f" + str(fold_ix) + "c" + str(cross_ix) + "/train/model" + str(options.head_i) + "_best.h5"
            if not os.path.isfile(model_path) :
                model_path = model_folder + "/f" + str(fold_ix) + "c" + str(cross_ix) + "/train/model_best.h5"

            seqnn_model.restore(
                model_path,
                options.head_i
            )
            seqnn_model.build_slice(targets_df.index, False)

            for shift in options.shifts:
                print("Processing shift %d" % shift, flush=True)

                for rev_comp in [False, True] if options.rc else [False]:

                    if options.rc:
                        print(
                            "Fwd/rev = %s" % ("fwd" if not rev_comp else "rev"), flush=True
                        )

                    seq_1hots = []
                    gene_slices = []
                    gene_slices_denom = []
                    gene_targets = []

                    for gi, gene_id in enumerate(gene_list):

                        if gi % 500 == 0:
                            print("Processing %d, %s" % (gi, gene_id), flush=True)

                        gene = transcriptome.genes[gene_id]

                        # make sequence
                        seq_1hot = make_seq_1hot(
                            genome_open,
                            genes_chr[gi],
                            genes_start[gi],
                            genes_end[gi],
                            seq_len,
                        )
                        seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)

                        # get splice dataframe
                        gene_splice_df = splice_df.query(
                            "Chromosome == '"
                            + genes_chr[gi]
                            + "' and ((End > "
                            + str(genes_start[gi])
                            + " and End <= "
                            + str(genes_end[gi])
                            + ") or (Start < "
                            + str(genes_end[gi])
                            + " and Start >= "
                            + str(genes_start[gi])
                            + ")) and Strand == '"
                            + str(genes_strand[gi])
                            + "'"
                        ).sort_values(by=["Chromosome", "Start"], ascending=True)

                        gene_slice = None
                        gene_slice_denom = None

                        if len(gene_splice_df) > 0:

                            # get random splice junction (donor or acceptor)
                            rand_ix = np.random.randint(len(gene_splice_df))

                            # get splice junction position
                            splice_start = gene_splice_df.iloc[rand_ix]["Start"]
                            splice_end = gene_splice_df.iloc[rand_ix]["End"]
                            splice_strand = gene_splice_df.iloc[rand_ix]["Strand"]
                            donor_or_acceptor = gene_splice_df.iloc[rand_ix]["feature"]

                            # determine output sequence start
                            seq_out_start = genes_start[gi] + model_stride * model_crop

                            # get relative splice positions
                            splice_seq_start = max(0, splice_start - seq_out_start)
                            splice_seq_end = max(0, splice_end - seq_out_start)

                            # determine output positions

                            if donor_or_acceptor == "donor":

                                # upstream coverage (before donor)
                                bin_start = None
                                bin_end = None
                                if splice_strand == "+":
                                    bin_end = (
                                        int(np.round(splice_seq_start / model_stride)) + 1
                                    )
                                    bin_start = bin_end - 3
                                else:
                                    bin_start = int(np.round(splice_seq_end / model_stride))
                                    bin_end = bin_start + 3

                                # clip right boundaries
                                bin_max = int(
                                    (seq_len - 2.0 * model_stride * model_crop)
                                    / model_stride
                                )
                                bin_start = max(min(bin_start, bin_max), 0)
                                bin_end = max(min(bin_end, bin_max), 0)

                                gene_slice = np.arange(bin_start, bin_end)

                                # downstream coverage (after donor)
                                bin_start = None
                                bin_end = None
                                if splice_strand == "+":
                                    bin_start = (
                                        int(np.round(splice_seq_end / model_stride)) + 1
                                    )
                                    bin_end = bin_start + 3
                                else:
                                    bin_end = int(np.round(splice_seq_start / model_stride))
                                    bin_start = bin_end - 3

                                # clip right boundaries
                                bin_max = int(
                                    (seq_len - 2.0 * model_stride * model_crop)
                                    / model_stride
                                )
                                bin_start = max(min(bin_start, bin_max), 0)
                                bin_end = max(min(bin_end, bin_max), 0)

                                gene_slice_denom = np.arange(bin_start, bin_end)

                            elif donor_or_acceptor == "acceptor":

                                # downstream coverage (after acceptor)
                                bin_start = None
                                bin_end = None
                                if splice_strand == "+":
                                    bin_start = int(np.round(splice_seq_end / model_stride))
                                    bin_end = bin_start + 3
                                else:
                                    bin_end = (
                                        int(np.round(splice_seq_start / model_stride)) + 1
                                    )
                                    bin_start = bin_end - 3

                                # clip right boundaries
                                bin_max = int(
                                    (seq_len - 2.0 * model_stride * model_crop)
                                    / model_stride
                                )
                                bin_start = max(min(bin_start, bin_max), 0)
                                bin_end = max(min(bin_end, bin_max), 0)

                                gene_slice = np.arange(bin_start, bin_end)

                                # upstream coverage (before acceptor)
                                bin_start = None
                                bin_end = None
                                if splice_strand == "+":
                                    bin_end = int(np.round(splice_seq_start / model_stride))
                                    bin_start = bin_end - 3
                                else:
                                    bin_start = (
                                        int(np.round(splice_seq_end / model_stride)) + 1
                                    )
                                    bin_end = bin_start + 3

                                # clip right boundaries
                                bin_max = int(
                                    (seq_len - 2.0 * model_stride * model_crop)
                                    / model_stride
                                )
                                bin_start = max(min(bin_start, bin_max), 0)
                                bin_end = max(min(bin_end, bin_max), 0)

                                gene_slice_denom = np.arange(bin_start, bin_end)

                        else:
                            gene_slice = np.array([0])
                            gene_slice_denom = np.array([0])

                        if gene_slice.shape[0] == 0 or gene_slice_denom.shape[0] == 0:
                            gene_slice = np.array([0])
                            gene_slice_denom = np.array([0])

                        if rev_comp:
                            seq_1hot = dna_io.hot1_rc(seq_1hot)
                            gene_slice = target_length - gene_slice - 1
                            gene_slice_denom = target_length - gene_slice_denom - 1

                        # slice relevant strand targets
                        if genes_strand[gi] == "+":
                            gene_strand_mask = (
                                (targets_df.strand != "-")
                                if not rev_comp
                                else (targets_df.strand != "+")
                            )
                        else:
                            gene_strand_mask = (
                                (targets_df.strand != "+")
                                if not rev_comp
                                else (targets_df.strand != "-")
                            )

                        gene_target = np.array(targets_df.index[gene_strand_mask].values)

                        # accumulate data tensors
                        seq_1hots.append(seq_1hot[None, ...])
                        gene_slices.append(gene_slice[None, ...])
                        gene_slices_denom.append(gene_slice_denom[None, ...])
                        gene_targets.append(gene_target[None, ...])

                        if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size:

                            # concat sequences
                            seq_1hots = np.concatenate(seq_1hots, axis=0)

                            # pad gene slices to same length (mark valid positions in mask tensor)
                            max_slice_len = int(
                                np.max([gene_slice.shape[1] for gene_slice in gene_slices])
                            )
                            max_slice_denom_len = int(
                                np.max(
                                    [
                                        gene_slice_denom.shape[1]
                                        for gene_slice_denom in gene_slices_denom
                                    ]
                                )
                            )

                            gene_masks = np.zeros(
                                (len(gene_slices), max_slice_len), dtype="float32"
                            )
                            gene_slices_padded = np.zeros(
                                (len(gene_slices), max_slice_len), dtype="int32"
                            )
                            for gii, gene_slice in enumerate(gene_slices):
                                for j in range(gene_slice.shape[1]):
                                    gene_masks[gii, j] = 1.0
                                    gene_slices_padded[gii, j] = gene_slice[0, j]

                            gene_slices = gene_slices_padded

                            gene_masks_denom = np.zeros(
                                (len(gene_slices_denom), max_slice_denom_len),
                                dtype="float32",
                            )
                            gene_slices_denom_padded = np.zeros(
                                (len(gene_slices_denom), max_slice_denom_len), dtype="int32"
                            )
                            for gii, gene_slice_denom in enumerate(gene_slices_denom):
                                for j in range(gene_slice_denom.shape[1]):
                                    gene_masks_denom[gii, j] = 1.0
                                    gene_slices_denom_padded[gii, j] = gene_slice_denom[
                                        0, j
                                    ]

                            gene_slices_denom = gene_slices_denom_padded

                            # concat gene-specific targets
                            gene_targets = np.concatenate(gene_targets, axis=0)

                            # batch call gradient computation
                            grads = seqnn_model.gradients(
                                seq_1hots,
                                head_i=0,
                                target_slice=gene_targets,
                                pos_slice=gene_slices,
                                pos_mask=gene_masks,
                                pos_slice_denom=gene_slices_denom,
                                pos_mask_denom=gene_masks_denom,
                                chunk_size=buffer_size,
                                batch_size=1,
                                track_scale=options.track_scale,
                                track_transform=options.track_transform,
                                clip_soft=options.clip_soft,
                                untransform_old=options.untransform_old,
                                use_mean=True,
                                use_ratio=True,
                                use_logodds=False,
                                subtract_avg=True,
                                input_gate=False,
                                dtype="float16",
                            )

                            # undo augmentations and save gradients
                            for gii, gene_slice in enumerate(gene_slices):
                                grad = unaugment_grads(
                                    grads[gii, :, :, None],
                                    fwdrc=(not rev_comp),
                                    shift=shift,
                                )

                                h5_gi = (gi // buffer_size) * buffer_size + gii

                                # write to HDF5
                                scores_h5["grads"][h5_gi] += grad

                            # clear sequence buffer
                            seq_1hots = []
                            gene_slices = []
                            gene_slices_denom = []
                            gene_targets = []

                            # collect garbage
                            gc.collect()

            # save sequences and normalize gradients by total size of ensemble
            for gi, gene_id in enumerate(gene_list):

                # re-make original sequence
                seq_1hot = make_seq_1hot(
                    genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len
                )

                # write to HDF5
                scores_h5["seqs"][gi] = seq_1hot
                scores_h5["grads"][gi] /= float(
                    (len(options.shifts) * (2 if options.rc else 1))
                )

            # collect garbage
            gc.collect()

    # close files
    genome_open.close()
    scores_h5.close()


def unaugment_grads(grads, fwdrc=False, shift=0):
    """Undo sequence augmentation."""
    # reverse complement
    if not fwdrc:
        # reverse
        grads = grads[::-1, :, :]

        # swap A and T
        grads[:, [0, 3], :] = grads[:, [3, 0], :]

        # swap C and G
        grads[:, [1, 2], :] = grads[:, [2, 1], :]

    # undo shift
    if shift < 0:
        # shift sequence right
        grads[-shift:, :, :] = grads[:shift, :, :]

        # fill in left unknowns
        grads[:-shift, :, :] = 0

    elif shift > 0:
        # shift sequence left
        grads[:-shift, :, :] = grads[shift:, :, :]

        # fill in right unknowns
        grads[-shift:, :, :] = 0

    return grads


def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    if start < 0:
        seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot


################################################################################
# __main__
# ###############################################################################
if __name__ == "__main__":
    main()
