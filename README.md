# Quality Assurance testing for the Print &amp; Probability book processing

The QA workflow operates several modules that represent distinct parts of the Print & Probability (P&P) book processing pipeline (i.e. `autocrop`, `line extraction`). Each QA module performs its respective part of the pipeline as well as a quality assurance process over its results. Modules are called one at a time. They take in a yaml config file and/or command line arguments as well as typical input (one or more folders of scanned book pages). They produce the typical output for that part of the P&P pipeline, but also several metadata files that give information about the results of the process. Modules have some common subprocesses, the calling and order of which are specified via the yaml config file (i.e. `clear`, `archive`, `run`, `output_stats`, `collate`). These are provided to allow for customization and quick (re)runs of the QA workflow and are described in the `Modules and Subprocesses` section below.

## Main operation

All QA workflow runs begin by calling the `run_qa.sh` bash script at the command line with `sbatch`:

```
sbatch run_qa.sh <module_name> <yaml_config_filepath>
```

Current valid module names are:
1. `autocrop` and
2. `line_extraction`

The contents of the yaml config file are described in the config section below.

## Modules and Subprocesses

Each QA module itself can initiate its underlying process but also has several 'subprocesses' that can be run before and afterward. (These 'subprocesses' themselves can be broken down into smaller substeps if desired.) Below are some general descriptions of each process and their own substeps (if they have them). Each QA module implements a version of a common base class/interface. For a more detailed look at the respective functions for these subprocesses, see Python class `QA_Module`, in `qa_utilities.py`.

### archive

### clear

### collate

### output_stats

### run






