import kfold_template
from sklearn import linear_model
import pandas


dataset = pandas.read_csv("regression_dataset.csv")

target = dataset.iloc[:,2].values
data = dataset.iloc[:,3:9].values

machine = linear_model.LogisticRegression()

results_accuracy, results_confusion = kfold_template.run_kfold(4,data,target,machine)

print(results_accuracy)

for i in results_confusion:
	print(i)


