Long term goal is to develop into an easy to use library for rapid prototyping and development of PTM tools. Called ptm_Pred

# Post Translational Modification Prediction 
Capstone project for Senior Year at Tulane University 

A full write up of using supervised learning and class imbalance methods can be found here: https://docs.google.com/document/d/1Yi3vMEq4l0SLw95HtiVRHsn010nrVaNZiZlV9pi7TjU/edit?usp=sharing 


The supervised methods generate precision and accuracy in the 80-90% range with recall in the 10-20% range.

Recently I have started using unsupervised learning methods with interesting results. The word2vec implementations are averaging around 75 in recall, precision, and accuracy for most post translational modifications tests. This presents a possible solution to the recall issue which has plagued post translational modification prediction for the last decade. 


# TODO:
Write FASTA -> CSV converter for benchmark tests

Implement benchmarks into word2vec.

Try prot2vec implementations

Try using exon/intron as an additional feature set.


### Notes:
The data posted comes from dbptm.mbc.nctu.edu.tw which is a great rescource for protien related machine learning projects.
