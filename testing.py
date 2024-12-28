from train_nlp import training_pipe
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

prepare = training_pipe.PrepareData(file_path="E:\\Projects\\auto_nlp\\question_category.csv",
                                file_type=".csv",
                                target_col_name="Category",
                                input_col_name="Question")
prepare.readData()
X, y = prepare.splitData()
print(prepare.show())

process_data = training_pipe.ProcessData(vectorizer_abbrivation='tf-idf')
process_data.fit(X, y)
X, y = process_data.processData()

print(X)
print(y)

analyzer = training_pipe.Analyze()
analyzer.fit(X=X, y=y)
analyzer.trainAndAnalyze(all=True, model_abbrivation='lr')
trainer = training_pipe.Trainer()
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
analyzer.analyzeModel(model_abbrivation='rf', y_true=y_test, y_pred=y_pred)

# tuner = training_pipe.HyperParameterTuner(model='lr', scoring='accuracy')
# result = tuner.bestEstimatorSelector(X=X, y=y)
# print(result)

