import pandas as pd
import numpy as np
import plotly
import uuid
import os
import json
import plotly.graph_objects as go
from flask import Flask, render_template, redirect, request, url_for, send_from_directory
app = Flask(__name__)

@app.route('/')
def index():
	filepath = "./AVG.txt"
	df = pd.read_csv(filepath, encoding="utf-16", sep='\t')

	x_col_name = df.columns[0]
	y_row_name = df.columns[1]
	z_name = df.columns[3]

	df = df.sort_values(by=[y_row_name, x_col_name])

	# convertiamo in "set" per eliminare i duplicati
	x_label = list(set(df[x_col_name]))
	y_label = list(set(df[y_row_name]))
	x_label.sort()
	y_label.sort()
	x_dim = len(x_label)
	y_dim = len(y_label)
	data = np.ndarray(shape=(y_dim, x_dim), dtype=float)
	df_number_of_row = df.shape[0]
	df_counter_row = 0

	# creazione superficie:
	for y_index in range(y_dim):

		dict_element = {}
		while y_label[y_index] == df.iloc[df_counter_row][y_row_name]:
			dict_element[df.iloc[df_counter_row][x_col_name]] = df.iloc[df_counter_row][z_name]
			df_counter_row += 1
			if df_counter_row >= df_number_of_row:
				break
		dict_len = len(dict_element)
		dict_counter = 0
		# Se l'y è nel dizionario
		for x_index in range(x_dim):
			if x_label[x_index] in dict_element:
				data[y_index, x_index] = dict_element[x_label[x_index]]
			else:
				data[y_index, x_index] = 0

	figure = go.Figure(data=[go.Surface(x=x_label, y=y_label, z=data, hovertemplate=x_col_name + ": %{x}" + \
																					"<br>" + y_row_name + ": %{y}" + \
																					"<br>" + z_name + ": %{z}<extra></extra>")])
	figure.update_layout(scene=dict(xaxis_title=df.columns[0], yaxis_title=df.columns[1], zaxis_title=df.columns[3]))
	graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template('notdash2.html', graphJSON=graphJSON, header="Grafico: ", description="")

@app.route('/pdf')
def get_pdf():  
    return send_from_directory('/root/tradingChart/static/assets/pdf/','UserGuide.pdf',mimetype='application/pdf')

@app.route('/',methods=['POST'])
def getLocalFile():

	df = pd.read_csv(request.files['file'],encoding="utf-16", sep='\t')

	x_col_name = df.columns[0]
	y_row_name = df.columns[1]
	z_name = df.columns[3]

	df = df.sort_values(by=[y_row_name, x_col_name])

	#convertiamo in "set" per eliminare i duplicati
	x_label = list(set(df[x_col_name]))
	y_label = list(set(df[y_row_name]))
	x_label.sort()
	y_label.sort()
	x_dim = len(x_label)
	y_dim = len(y_label)
	data = np.ndarray(shape=(y_dim, x_dim), dtype=float)
	df_number_of_row = df.shape[0]
	df_counter_row = 0

	#creazione superficie:
	for y_index in range(y_dim):

		dict_element = {}
		while y_label[y_index] == df.iloc[df_counter_row][y_row_name]:
			dict_element[df.iloc[df_counter_row][x_col_name]] = df.iloc[df_counter_row][z_name]
			df_counter_row += 1
			if df_counter_row >= df_number_of_row:
				break
		dict_len = len(dict_element)
		dict_counter = 0
		#Se l'y è nel dizionario
		for x_index in range(x_dim):
			if x_label[x_index] in dict_element:
				data[y_index, x_index] = dict_element[x_label[x_index]]
			else:
				data[y_index, x_index] = 0

	figure = go.Figure(data=[go.Surface(x=x_label, y=y_label, z=data, hovertemplate=x_col_name + ": %{x}"+\
								 "<br>" + y_row_name + ": %{y}"+\
								 "<br>" + z_name + ": %{z}<extra></extra>")])
	figure.update_layout(scene = dict(xaxis_title=df.columns[0], yaxis_title=df.columns[1], zaxis_title=df.columns[3]))
	graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template('notdash2.html', graphJSON=graphJSON, header="Grafico: ", description="")


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port='5010')
