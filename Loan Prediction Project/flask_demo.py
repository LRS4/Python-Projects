import numpy as np
from flask import Flask, abort, jsonify, request
import pickle

my_random_forest = pickle.load(open("loan_rfc.pkl", "rb"))

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def make_predict():
	data = request.get_json(force=True)
	#convert json into numpy array
	predict_request = [data["Gender"], data["Married"], data["Dependents"], data["Education"], data["Self_Employed"], data["ApplicantIncome"], data["CoapplicantIncome"], data["LoanAmount"], data["Loan_Amount_Term"], data["Credit_History"], data["Property_Area"]]
	predict_request = np.array(predict_request)
	#np array goes into random forest, prediction comes out
	y_hat = my_random_forest.predict(predict_request)
	# return prediction
	output = [y_hat[0]]
	return jsonify(results=output)
	
if __name__ == '__main__':
	app.run(port = 9000, debug = True)