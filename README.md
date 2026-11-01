## A vicious cycle of aversive beliefs and avoidance across transdiagnostic symptoms


This repo is associated with the preprint: "A vicious cycle of aversive beliefs and avoidance across transdiagnostic symptoms". It contains data and scripts to reproduce the results and figures in the main text & Supplementary Material.

```{}
The task code can be found in a separate repo: 
Here you can find a summary of the task and study: https://luiantav.github.io/dog-island/
```

Instructions 

- The behavioural analyses scripts are found here: `/gen-avo/data-analysis/`. The computational environment can be reproduced using the     renv package for R environments: To do so, run `install.packages("renv")` and `renv::restore()` from any R script. 
- Modeling scipts and analyses of modeling outputs are found here: `/gen-avo/behavioral-modeling/`. The computational environment can be reproduced using a virtual environment. To do so, run ` source myenv/bin/activate` from your terminal. 
- All data needed for analyses are provided in data folder `/data`.


First steps: 
```{}
1. Clone git repo
2. cd to your cloned repo and activate the environment (see above)
3. Start by running the main analysis script: `/gen-avo/data-analysis/selected_analyses.Rmd`
```

Requirements

    R version 4.2.1 (2022-06-23)
    RStudio (or other r-markdown editor)
    Python 3.10.8 (main, Oct 13 2022, 09:48:40)

Scripts tested on

    macOS Monterey 12.7.6; macOS Catalina 10.15.7


Expected runtime ~10 min for behavioural and modelling output analysis. Raw files of model fitting, parameter and model recovery are provided.

License Code: GNU GPLv3 (see LICENSE.md)
