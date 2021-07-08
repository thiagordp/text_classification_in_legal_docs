# Classificação de sentenças de Juizado Especial Cível utilizando aprendizado de máquina 

Source code for the paper publishe in *Revista Democracia Digital e Governo Eletrônico*.

Link for the paper: https://buscalegis.ufsc.br/revistas/index.php/observatoriodoegov/article/view/316

## Project Structure

### Classical ML Experiments

In this subproject, we add the pipeline used for text classification using Orange 3.
The file can be opened as a project in the Orange 3 tool.

We used the following classical ML techniques:

- k-Nearest Neighbors
- Random Forest
- Naïve Bayes
- feed-forward Neural Networks
- Logistic Regression
- Support Vector Machine
- Neural Network

### Deep Learning Experiments

In this subproject, we present the code of the pipeline for text classification implemented in the Python language and open-source tools, such as, [NLTK](https://www.nltk.org/), [Scikit-Learn](https://scikit-learn.org/), and [Pandas](https://pandas.pydata.org/).

The following DL techniques were used:

- LSTM
- CNN (based on [Kim (2014)](https://arxiv.org/abs/1408.5882))
- Bi-LSTM with Self-Attention [Chalkidis, Kampas (2019)](https://link.springer.com/article/10.1007/s10506-018-9238-9)

## Notes

- We did not make our datasets for text classification available due to the existing personal data  in the documents.
