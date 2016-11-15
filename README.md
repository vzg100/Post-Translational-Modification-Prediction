# ProteinFolding
Capstone project for Senior Year


<b>Currently used features:</b>

-Protien Sequence

-region sequence (+- 7 AA on either side)

-Amino acid species modifier

<b>Currently used labels:</b>

-Amino acid index (reffering to total sequence)

<b>Currently used Classifiers:</b>

SVC - Linear kernel

Random Forest - n_estimators = 120

KNN - 2N


<b>Current Output: </b>

-High

SVC has a success rate of:0.29411764705882354

Random Forest has a success rate of:0.4117647058823529

KNN has a success rate of:0.35294117647058826

-Norm

SVC has a success rate of:0.29411764705882354

Random Forest has a success rate of:0.35294117647058826

KNN has a success rate of:0.35294117647058826

-Low

SVC has a success rate of:0.058823529411764705

Random Forest has a success rate of:0.35294117647058826

KNN has a success rate of:0.29411764705882354


<b>Current Issues: </b>

<b>Data Issues</b>

-I selected data from the top of the file and it seems like modifications on the same protien are similarly labeled

-Data was not shuffeled when split leading to even more biasing of certain patterns

-Certain data sets are difficult to scrape and I haven't figured out how to cleanly do it ex. Locilization


<b>Classifier Issues</b>

-I think I need to fine tune the classifier parameters, currently it looks like making nodes more "independent" returns better results

-Linear seems to the best classifier for SVC, I am curious as to why


<b>Next Steps:</b>

-Clean up some of the classifier object code

-Collect more features to form a better data set from which to test and train

-Implement data "shuffling" for better training/testing

-Implement filters for specific amino acids so that their 








