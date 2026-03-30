Just a silly sequential words predictor using simple stochastic model (Markov Chain) and Bayesian Inference.

# Model training

Before making predictions, you must train the model using `--train` flag:

```bash
python model.py --train True
```

# Making predictions

To predict the next N words of a text, you must give the interested text with the `--phrase` flag
and the N quantity (an integer) with the `--length` flag

```bash
python model.py --phrase "your text here" --length N
```
