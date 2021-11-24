
## Replication files for <ins>The Economics of Sovereign Debt and Default </ins> 
### by Mark Aguiar and Manuel Amador

### Overview
This repository replicates the computations and figures presented in the manuscript.  All code is written for Julia. 

We are especially grateful to Stelios Fourakis, who wrote the original code for the quantitative models.  The code in this repository has been written by the authors based on that original code. 


### Installation

The code is written to be a self-contained "package."  The Project.toml and Manifest.toml files contain information on dependencies and should not be edited or changed.  To install all necessary packages, open a julia prompt at the root of this repository and type:

    julia> using Pkg 
    julia> Pkg.activate(".")
    julia> Pkg.instantiate()

The above will download the packages needed to run the code.  After this step, the files in the "scripts" subfolder can be run.  

### Replicating Figures and Tables

The replication programs are in the "scripts" subfolder:
  1.  Chapter5.jl contains the code to replicate tables and figures from Chapter 5.  
  2.  Chapter7.jl contains the code to replicate the tables and figures from Chapter 7 Section 7.9. 
   
   These two files utilize plotting_functions.jl, which contain code for formatting figures as they appear in the manuscript.
      
  3.  The file Chapter7_simplemodel.jl in the "Ch7_simplemodel" subfolder contains the code to generate figures for Sections 7.6 and 7.7 of Chapter 7.  This file uses longbonds_cont_time.jl.  
   
   All figures are saved to the "output" subfolder.


### The SRC Folder

The "src" folder contains the code used to compute the quantitative sovereign bond models of Chapters 5 and 7.  It is not necessary to access these files directly.  The relevant types and functions are collected in the LTBonds module, which is imported by the files in the scripts subfolder.


