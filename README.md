# bitcoinprediction
Predicting bitcoin daily price trends with deep learning

This repository contains feature extraction and running a bayesian approximation neural network code as part of my MSc Cognitive Science and Artificial Intelligence masters thesis at Tilburg Univesity, NL.

For this thesis I looked at extracting a wide range of different features from Bitcoin in order to make an Artificial Neural Network that could predict next day Bitcoin USD price movements better than majority class (i.e a binary classifier predicting a '1' or a next day price rise or a '0' for a next day price fall). Bespoke functions were coded to extract Bitcoin features relating to the general economic environment, public awareness (from Google Trends and the GDELT database), technical analysis indicators and Blockchain technical indicators.

Despite a wide ranging grid search and extensive feature ablation, an ANN model with an accuracy better than the majority trend over the X test set (a price rise, 59.4% of the time) could not be created.

However, by using Dropout in the testing phase of the model an approximation of a Bayesian Neural Network was successfully created. A BNN provides confidence information about each prediction it makes (rather than a simple point estimate provided by an ANN). Using a combination of technical analysis indicators and public awareness features it was possible to create a BNN model whereby the accuracy of predictions increased dramatically with the model's confidence. These results allowed for an investment strategy to be formulated whereby investments are only made on days where the BNN model is particularly confident. This investment strategy proved to be significantly more profitable than a random investment strategy.

Full results and report to be published once the final draft of the thesis is submitted.
