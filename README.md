## Setting Up

- Since some of our data exceeds the 100MB limit, follow the steps here to install Git LFS: https://git-lfs.com/ to be able to clone and push the data files.

## Dependencies

- jupyters
- pandas
- numpy
- matplotlib
- wordcloud
- nltk (punkt)
- spacy ("en_core_web_sm")
- py-readability-metrics (readability)

## Structure

- `processed_data/` contains the data files after preprocessing.
- `codes/` contains the code files the data cleaning, exploratory data analysis and preprocessing.
- `models` contains the code for the trained models.
- `images/` contains the images generated from the code including confusion matrices, word clouds, etc.
