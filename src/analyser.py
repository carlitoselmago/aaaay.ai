from flask import Flask, jsonify,request
import tensorflow as tf
import numpy as np
import _pickle as cPickle
from _pickle import dumps, loads
import json



"""
app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def hello():
	
	var=request.get_data()#RAW DATA
	#var=(request.form.get("hola"))#POST
	#var=int(request.args.get("hola"))#GET
	#var+=1
	#var=map(bin,bytearray(var))
	print (loads(var))
	return "hola"
	#clientID= request.json["client"]
	#return "hola"

@app.route("/analysiseFrame",methods=["POST"])
def analysiseFrame():
		input=request.get_data()
		# Unpersists graph from file
		     
		decoded_image=loads(input)
		predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': decoded_image})
		#self.labelsData=dict(zip(self.labels, predictions[0]))
		return json.dumps(predictions[0].tolist())

		for i,pred in enumerate(predictions[0]):
		    pass
		    #self.graphs[i].append(pred)
		prediction = predictions[0]

		# Get the highest confidence category.
		prediction = prediction.tolist()
		max_value = max(prediction)
		max_index = prediction.index(max_value)
		predicted_label = labels[max_index]

		print("%s (%.2f%%)" % (predicted_label, max_value * 100))
		if predicted_label != "normal":
			pass
			#position=random.randint(0,len(self.soundsLoaded[predicted_label])-1)
			#self.playSound(self.soundsLoaded[predicted_label][position],0)

		#self.processing=False
		return predicted_label 
              
def get_labels():
	
	with open('train/retrained_labels.txt', 'r') as fin:
		labels = [line.rstrip('\n') for line in fin]
		return labels

def __init__():
	labels=get_labels()
	with tf.gfile.FastGFile("train/retrained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	app.run(host="127.0.0.1", port=23948)

if __name__ == '__main__':
	__init__()
"""



class analyser():

	def __init__(self):
		labels=self.get_labels()
		with tf.gfile.FastGFile("train/retrained_graph.pb", 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')
		with tf.Session() as self.sess:
			self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
		

	def get_labels(self):
	
		with open('train/retrained_labels.txt', 'r') as fin:
			labels = [line.rstrip('\n') for line in fin]
			return labels

app = Flask(__name__)

anal=analyser()

@app.route("/",methods=["GET","POST"])
def hello():
	
	var=request.get_data()#RAW DATA
	#var=(request.form.get("hola"))#POST
	#var=int(request.args.get("hola"))#GET
	#var+=1
	#var=map(bin,bytearray(var))
	
	return "hola"
	#clientID= request.json["client"]
	#return "hola"

@app.route("/analysiseFrame",methods=["POST"])
def analysiseFrame():

		input=request.get_data()
		# Unpersists graph from file
		     
		decoded_image=loads(input)
		predictions = anal.sess.run(anal.softmax_tensor, {'DecodeJpeg:0': decoded_image})
		#self.labelsData=dict(zip(self.labels, predictions[0]))
		return json.dumps(predictions[0].tolist())

		

if __name__ == '__main__':
	app.run(host="127.0.0.1", port=23948,threaded=True)