# fairpul

Code for FAccT paper: Fairness-aware Model-agnostic Positive and Unlabeled Learning.



### Dependencies

- numpy 1.19.2
- scikit-learn 0.23.2
- pulearn 0.0.7



### File Structure

- src/gen_syn.py: generate the synthetic dataset
- src/dataloader.py: load real data sets
- src/PUL.py: train and test our proposed framework



### Usage

We have indicated how each step in All.1 in our paper corresponds to the code in the comments. To reproduce the experimental results in Table 1, simply run `python PUL.py` . The datasets can be downloaded from the weblink in the paper. All the carefully tuned parameters have been specified in the paper.

