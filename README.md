# Artificial intelligence is ineffective and potentially harmful for fact checking

## Paper
For more details, you can find the paper [here]. It should be cited as:
- [Matthew R. DeVerna](https://www.matthewdeverna.com/), [Harry Yaojun Yan](https://cns-nrt.indiana.edu/students/trainees/2018/Harry-Yaojun-Yan.html), [Kai-Cheng Yang](https://www.kaichengyang.me/), [Filippo Menczer](https://cnets.indiana.edu/fil/) (2023). **Artificial intelligence is ineffective and potentially harmful for fact checking**, ArXiv preprint:2308.10800. doi: https://doi.org/10.48550/arXiv.2308.10800

```bib
@article{deverna2023artificial,
  title={Artificial intelligence is ineffective and potentially harmful for fact checking},
  author={DeVerna, Matthew R. and Yan, Harry Yaojun and Yang, Kai-Cheng and Menczer, Filippo},
  journal={Preprint arXiv:2308.10800},
  year={2023}
}
```

### Project aim
We conduct a [preregistered](https://osf.io/58rmu/) experiment to investigate whether fact checks provided by a large language model (ChatGPT) can serve as an effective misinformation intervention.

## Directories
- `code`: scripts to generate results, figures, etc.
- `data`: data used in the project
- `environment`: python environment files
- `figures`: all generated figures
- `results`: output files that include processed data and statistical reports


## Replication
We utilize both Python and R coding languages in this project.

### Requirements
- Python: see the `environment/` directory for virtual environment details
    - Used for most data wrangling/manipulation/basic stats
- R: version 4.3.0 (2023-04-21)
    - Used for supplementary regression analyses

#### Python analysis and figure generation
To replicate the Python analysis, please set up an environment as described in the `environment/` directory.
Once that is completed successfully, you should be able to run the below code to conduct all analyses and generate all figures:
```bash
cd code # Change directory to `code`
bash run_pipeline.sh # Runs entire pipeline
```

#### R analysis and figure generation
The version of R that is utilized in this project is: R version 4.3.0 (2023-04-21).
We also utilized RStudio Version 2023.06.0+421 (2023.06.0+421).

If you want to check what version of R you currently have, you can start an R session and run `version`, giving you something like the below:
```shell
> version
               _
platform       aarch64-apple-darwin20
arch           aarch64
os             darwin20
system         aarch64, darwin20
status
major          4
minor          3.0
year           2023
month          04
day            21
svn rev        84292
language       R
version.string R version 4.3.0 (2023-04-21)
nickname       Already Tomorrow
```

If you want to check what version of RStudio you are running, start up RStudio and (on Macs, atleast) you can click the `RStudio > About RStudio` dropdown.

All analyses and figures are created with a single script: `code/r_code/2023_08_11_Final.Rmd`.

After installing the versions of R and RStudio indicated above, you should be able to open the script with RStudio and "Knit" the file, creating the HTML output currently saved in the same location.
