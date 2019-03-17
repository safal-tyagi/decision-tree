19S.6375.005 PA1     
by SKT180001, VXV170013 

------------------------------------------------------------------------------------------------------------------------
Folder structure:
[Your Path]
	+ data [the given data folder]
	- decision_tree.py
	- report.pdf
	- readme.txt

How to run:
Everything is in same file, you can directly run the program. Follow the __main__ function. It covers all a, b, c, d parts step-wise
We have put most of the documentation in the code itself.


Important new methods added:
id3_evaluate		// uses our id3 algorithm, prints prediction errors and confusion matrix for any data set
sklearn_evaluate	// uses sklearn tree algorithm, prints prediction errors and confusion matrix for any data set
binarize		// pre-process dataset into binary features

Description on these method is in the code itself.
