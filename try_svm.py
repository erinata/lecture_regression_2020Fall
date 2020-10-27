import kfold_template
from sklearn import svm
import pandas


dataset = pandas.read_csv("regression_dataset.csv")

target = dataset.iloc[:,2].values
data = dataset.iloc[:,3:9].values

# machine = linear_model.LogisticRegression()
machine = svm.SVC(kernel="linear")

results_accuracy, results_confusion = kfold_template.run_kfold(5,data,target,machine)

print(results_accuracy)

for i in results_confusion:
	print(i)


