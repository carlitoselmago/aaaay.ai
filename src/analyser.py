from flask import Flask, jsonify,request
import tensorflow as tf
import numpy as np
import _pickle as cPickle
from _pickle import dumps, loads
import json
import os
import threading

class analyser():

	response=False

	def __init__(self):
		labels=self.get_labels()
		with tf.gfile.FastGFile(os.path.dirname(os.path.realpath(__file__))+"/train/retrained_graph.pb", 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')
		with tf.Session() as self.sess:
			self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
		

	def get_labels(self):
	
		with open(os.path.dirname(os.path.realpath(__file__))+'/train/retrained_labels.txt', 'r') as fin:
			labels = [line.rstrip('\n') for line in fin]
			return labels

	def runAnalyser(self,inputdata):
		
		# Unpersists graph from file
		     
		decoded_image=loads(inputdata)
		predictions = anal.sess.run(anal.softmax_tensor, {'DecodeJpeg:0': decoded_image})
		#self.labelsData=dict(zip(self.labels, predictions[0]))
		print (json.dumps(predictions[0].tolist()))
		self.response=json.dumps(predictions[0].tolist())
		return 


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
		"""
		t = threading.Thread(target=anal.runAnalyser, args = (request.get_data(),))
		t.start()
		counter=0
		while not anal.response:
			#print("wait",counter)
			counter+=1

			if counter>119709422:
				#cancel thread
				print ("TIME OUT CANCEL THREEAD!")
				anal.response="[0,0,1]"
				
		print ("counter",counter)
		returnvalue=anal.response
		print ("returnvalue",returnvalue)
		anal.response=False
		return returnvalue
		"""
		decoded_image=loads(request.get_data())
		predictions = anal.sess.run(anal.softmax_tensor, {'DecodeJpeg:0': decoded_image})
		#self.labelsData=dict(zip(self.labels, predictions[0]))
		print (json.dumps(predictions[0].tolist()))
		return json.dumps(predictions[0].tolist())



if __name__ == '__main__':
	"""
	from tornado.wsgi import WSGIContainer
	from tornado.httpserver import HTTPServer
	from tornado.ioloop import IOLoop
	
	http_server = HTTPServer(WSGIContainer(app))
	http_server.listen(23948)
	IOLoop.instance().start()
	"""
	app.run(host="127.0.0.1", port=23948,threaded=True)