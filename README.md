# Fact-checking information from large language models can decrease headline discernment

## Paper
For more details, you can find the paper  [here](https://doi.org/10.1073/pnas.2322823121). It should be cited as:

[Matthew R. DeVerna](https://www.matthewdeverna.com/), [Harry Yaojun Yan](https://scholar.google.com/citations?user=tBCQR_8AAAAJ&hl=en), [Kai-Cheng Yang](https://www.kaichengyang.me/), [Filippo Menczer](https://cnets.indiana.edu/fil/) (2023). **Fact-checking information from large language models can decrease headline discernment**, Proc. Natl. Acad. Sci. U.S.A. 121 (50) e2322823121, https://doi.org/10.1073/pnas.2322823121 (2024).

**Bib**
```bib
@article{DeVerna2024AIFactChecking,
  author = {Matthew R. DeVerna and Harry Yaojun Yan and Kai-Cheng Yang and Filippo Menczer},
  title = {Fact-checking information from large language models can decrease headline discernment},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {121},
  number = {50},
  pages = {e2322823121},
  year = {2024},
  doi = {10.1073/pnas.2322823121},
  url = {https://doi.org/10.1073/pnas.2322823121},
}
```

### Project aim
We conduct a [preregistered](https://osf.io/58rmu/) experiment to investigate whether fact checks provided by a large language model (ChatGPT) can serve as an effective misinformation intervention.

## Directories
- `code`: scripts to generate results, figures, etc.
- `data`: data used in the project
- `environment`: python environment files
- `figures`: all generated figures
- `results`: output files including processed data and statistical reports
- `prompt_engineering`: contains everything for the prompt engineering analysis in the supplementary information (the "Accuracy of different prompt methods" secton within the SI).

### Requirements
- Python: see the `environment/` directory.
    - Used for most data wrangling/manipulation/basic stats
- R: version 4.3.0 (2023-04-21)
    - Used for regression analyses

### Replication
We utilize both Python and R coding languages in this project.

#### Python analysis and figure generation
To replicated the Python analysis, please set up an environment as described in the `environment/` directory.
Then, you should be able to run the below code (after changing your current directory to wherever this `README.md` file is saved) to run all analyses and generate all figures:
```bash
cd code
bash run_pipeline.sh
```

#### R analysis and figure generation
The version of R that is utilized in this project is: R version 4.3.0 (2023-04-21).
We also utilized RStudio Version 2023.06.0+421 (2023.06.0+421).
All analyses and figures are created with the RMarkdown files in the `code/r_code/` directory.

After installing the versions of R and RStudio indicated above, you should be able to open the RMarkdown files with RStudio and "Knit" each one, creating the HTML version currently saved in the same location.

#### Prompt engineering supplementary analysis
To generate the results of this analysis, you can run the following code (after changing your current directory to wherever this `README.md` file is saved):
```shell
cd prompt_engineering/code
bash run_pipeline.sh
```

#### Questions

All questions should be directed to [Matt DeVerna](https://www.matthewdeverna.com).
