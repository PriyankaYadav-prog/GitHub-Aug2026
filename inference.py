import pickle


def make_predictions(model_path, data):
	with open(model_path, "rb") as model_file:
		model = pickle.load(model_file)

	predictions = model.predict(data)
	return predictions


data = [[
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]]

predictions = make_predictions("./logistic_regression_model.pkl", data)
print(predictions)
