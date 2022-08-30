# limerickation-experiment
# limerickation-experiment
This repository contains code to fine-tune the GPT-2 model using the limerickation data set from https://doi.org/10.5281/zenodo.5722527.

The GPT-2 model is fine tuned twice, once with the data set as is and once with the reversed data set.

For the reverse fine tuning step, the code has been taken from https://github.com/coderalo/11785-automatic-poetry-generation/ and repurposed.
To forward fine tune GPT-2, download the json file from https://doi.org/10.5281/zenodo.5722527 containing the limerick data set and run fine-tune-forward.py
This can take up to a day to train.

To reverse fine tune GPT-2, download the limericks.json file from https://github.com/coderalo/11785-automatic-poetry-generation/tree/main/data/preprocessing and run fine-tune-reverse.py
This can take up to a day to train.

Once the models are ready, run experiment-results.py to see the results of the models at generating limericks when given a first verse prompt.
