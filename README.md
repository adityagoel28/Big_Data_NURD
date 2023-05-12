# NURD: Negative-Unlabeled Learning for Online Datacenter Straggler Prediction

This is the Python implementation for experiments in NURD: Negative-Unlabeled Learning for Online Datacenter Straggler Prediction.

## Requirements
* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [imbalance-learn](https://pypi.org/project/imbalanced-learn/)
* [pulearn](https://pulearn.github.io/pulearn/)
* [Grabit](https://github.com/fabsig/KTBoost)
* [PyOD](https://github.com/yzhao062/pyod)

## Data
The data folder includes code for preprocessing Google and Alibaba trace data.

Google trace data is a collection of data that is collected by Google's infrastructure. This data can be used to track the performance of Google's services, identify bottlenecks, and improve the overall user experience.

The trace data is collected from a variety of sources, including:

* Applications: Google's applications, such as Gmail, YouTube, and Google Search, generate trace data that can be used to track the performance of these applications.
* Infrastructure: Google's infrastructure, such as data centers and networks, also generate trace data that can be used to track the performance of these systems.
* Users: Google's users also generate trace data, such as the websites they visit, the videos they watch, and the searches they perform.

## Code
`run_ts.py` includes implementations for the following methods:
* Base (gb): A Basic learner trained on observed tasks and predict stragglers on unseen tasks. Use gradient boosting tree model.
* LR (log): A logistic regression model trained on observed tasks and predict stragglers on unseen tasks. 
* OS (os): A complete solution for straggler prediction using linear support vector machines and oversampling to account for a lack of stragglers in training proposed in [Wrangler](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/wrangler_socc14.pdf).
* DS (ds): A variant of the above but using downsampling instead.
* SS-EN (en): A semi-supervised learning method proposed by [Elkan and Noto](https://cseweb.ucsd.edu/~elkan/posonly.pdf).
* SS-BG (bg): A bagging-based semi-supervised learning method by [Mordelet and Vert](https://arxiv.org/abs/1010.0772).
* Tobit (tb): Tobie regression model for censored data.
* Grabit (kt): Gradient tree-boosted Tobit model by [Sigrist and Hirnschall](https://arxiv.org/abs/1711.08695).
* IPW (gb-ipw): proposed NURD in the paper.

`run_od.py` includes implementations for all the outlier detection methods.
