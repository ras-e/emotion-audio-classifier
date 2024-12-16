# Emotion Audio Classifier

Multimedia Course, Final Exam, on Department of Electrical and Computer Engineering, Master @ NTUST 2024, Taipei, Taiwan

### Authors:

- [Rasmus Enemark Jacobsen](https://github.com/ras-e)
- [Tobias Erik Rosengren](https://github.com/tobro174)

## How to run:

Manually run: 
- organize_dataset.py : This will unzip the used datasets, extract and organize into a labeled emotion folder.
- preprocess.py
Preprocess args can be found in the preprocess.py main method. However, this is the most common of which will run all with augnmented and enhanced features.

```
python src/preprocess.py --mode process --augment --use_enhanced_features
```
- Run the main.py file to train the model (Parameters can be modified in main.py to either kfold, or a 80, 20 split)
- Run the UI for an interface to submit a .wav audio sample and visualize plots


## Collection of Dataset

### 1) [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976)

> Citation: "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.

- 24 professional actors (12 female, 12 male), two lexically-matched statements in a neutral North American accent.
- Number of Dataset: 1440
- `Emotion` 7 Classes: Calm, Happy, Sad, Angry, Fearful, Surprise and Disgust
- Emotional `Intensity`: Normal, Strong
- 2 Zip files: Speech and Song

### 2) [TESS (Toronto emotional speech set)](https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683%2FSP2%2FE8H2MF)

> License: Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0), https://creativecommons.org/licenses/by-nc-nd/4.0/

- 200 target words were spoken in the carrier phrase "Say the word \_' by two female actresses (aged 26 and 64 years).
- `Emotion` 7 Classes: Anger, Disgust, Fear, Happiness, Pleasant Surprise, Sadness, and Neutral
