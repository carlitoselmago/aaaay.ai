# use with python 3
import time
import cv2
import numpy as np
import sys
import threading
import ntpath
from time import sleep
import pygame
import os
import os.path
import random
import multiprocessing
from multiprocessing import Array, Value
import subprocess
import glob
import grequests
import requests
from _pickle import dumps, loads
import json
import random

pygame.mixer.init()

class aaaay():
	
	fullscreen=True
	overlayMode="words"#options stats, words
	soundBoard="all"
	boringLimit=1000
	screenSize= (1280, 720)
	minScandalous=0.6 #scale from 0.0 to 1.0
	windowName="aaaay.ai"
	
	processing=False
	videomode="pygame" #options: gtk,pygame
	graphs=[] 
	colors=[[0,255,255],[255,255,0],[255,0,255],[125,125,0]]
	soundsLoaded={}
	soundNamesLoaded={}
	
	animationWords=[]
	skip=1 # in seconds
	analyzingCount=0
	analyzing=False
	responseTime=0
	firstRequestAnswer=False
	
	def __init__(self):
		self.labels=self.get_labels()
		
		#self.analx = multiprocessing.Process(target=self.runAnalyser)
		#self.analx.start()
		self.prepareAudios()
		self.prepareWords()
		print ("init aaaay class")
		
		self.labelsData={}
		for i,label in enumerate(self.labels):
			self.labelsData[label]=0
			self.graphs.append([0.00])
		try:
			pass
			
		except:
			print("No labels where created yet")
		if self.videomode=="pygame":
			self.initPygameScreen(1)


		
	def runAnalyser(self):
		self.anal=subprocess.Popen("python3 analyser.py")

	def stopAnalyser(self):
		#self.analx.anal.kill()
		#self.analx.terminate()
		self.anal.kill()
		self.analx.terminate()

	   
	def prepareAudios(self):
		
		for path, subdirs, files in os.walk('sounds'+os.sep+self.soundBoard):
			for dirs in subdirs:
				#print (dirs)
				self.soundsLoaded[dirs]=[]
				#self.soundNamesLoaded[dirs]=[]
			for name in files:
				label= path.split("\\")[-1]
				#self.soundsLoaded[label].append(AudioSegment.from_ogg(os.path.abspath(os.path.join(path, name)).replace("\\","/")))
				#sound=pygame.mixer.Sound(os.path.abspath(os.path.join(path, name)).replace("\\","/"))
				self.soundsLoaded[label].append(pygame.mixer.Sound(os.path.abspath(os.path.join(path, name)).replace("\\","/")))
				#self.soundNamesLoaded[label].append(name.split(".")[0])
				
	def prepareWords(self):
		for path, subdirs, files in os.walk('words'+os.sep+"analysis"):
			for name in files:
				label= name.split(".")[0]
				self.soundNamesLoaded[label]=[]
				words=json.load(open(os.path.abspath(os.path.join(path, name)).replace("\\","/"),encoding='utf-8'))
				for word in words:
					self.soundNamesLoaded[label].append(word)

		self.soundNamesLoaded["boring"]=json.load((open("words/boring.json",encoding='utf-8')))
	
	def playSound(self,sound,delay):
		#play(sound)
		sound.play()

	
		
	def capture_images_from_camera(self,save_folder):
		"""Stream images off the camera and save them."""
		camera = PiCamera()
		camera.resolution = self.captureRes
		camera.framerate = self.framerate

		# Warmup...
		time.sleep(self.skip)

		# And capture continuously forever.
		for i, frame in enumerate(camera.capture_continuous(
			save_folder + '{timestamp}.jpg',
			'jpeg', use_video_port=True
		)):
			pass
		
	def get_labels(self):
		"""Get the labels our retraining created."""
		with open('train/retrained_labels.txt', 'r') as fin:
			labels = [line.rstrip('\n') for line in fin]
			return labels
		
	def runCommand(self,some_command):
		import subprocess
		p = subprocess.Popen(some_command, stdout=subprocess.PIPE, shell=True)
		(output, err) = p.communicate()  
		p_status = p.wait()


	def downloadAndWatch(self,url):
		
		id=url.split("?v=")[1]
		if not os.path.isfile("movies/"+id+"_PAL.mp4"):
			
			print("DOWNLOADING VIDEO, please wait")
			youtubeDLCommand='youtube-dl -f "best[width<=1080,height<=720]" --output "movies/'+id+'.mp4" -k --extract-audio --audio-format mp3 '+url
			#youtubeDLCommand2='youtube-dl -f webm --output "movies/'+id+'.webm" '+url
			self.runCommand(youtubeDLCommand)
			print("DOWNLOADING AUDIO, please wait")
			#self.runCommand(youtubeDLCommand2)
			ffmpegCommand="ffmpeg -i movies/"+id+".mp4 -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -movflags +faststart -vf scale=-2:576,format=yuv420p movies/"+id+"_PAL.mp4"
			
			print("CONVERTING VIDEO, please wait")
			self.runCommand(ffmpegCommand)
		
		self.run_classification("movies/"+id+"_PAL.mp4")
	

	def predict_on_image(self,image):

		# Unpersists graph from file
		with tf.gfile.FastGFile("train/retrained_graph.pb", 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')

		with tf.Session() as sess:
			softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

			# Read in the image_data
			image_data = tf.gfile.FastGFile(image, 'rb').read()

			try:
				predictions = sess.run(softmax_tensor, \
					 {'DecodeJpeg/contents:0': image_data})
				prediction = predictions[0]
			except:
				print("Error making prediction.")
				sys.exit()

			# Return the label of the top classification.
			prediction = prediction.tolist()
			max_value = max(prediction)
			max_index = prediction.index(max_value)
			predicted_label = self.labels[max_index]

			return predicted_label,max_value

	def toMultiProcessArray(self,input):
		return input.flatten()

	def toNumpyImage(self,input,shape):
		return np.array(input,dtype=np.uint8).reshape(shape)

	def callAnalyser(self,frame):
		self.Analbusy=True
		#r = ([grequests.post("http://127.0.0.1:23948/analysiseFrame", data=dumps(frame),headers={'Content-Type': 'application/octet-stream'},timeout=8,hooks={'response':self.manageAnalResponse})])
		#grequests.map(r, exception_handler=self.errorRequests)
		#grequests.map(r)
		
		r=requests.post("http://127.0.0.1:23948/analysiseFrame",data=dumps(frame))
		self.manageAnalResponse(r)
		return
		
	def errorRequests(self,a,b):
		
		if "Read timed out" in str(b):
			print(a,b)
			print (type(b))
			
			print ("excpetion handler call with grequests")
			self.Analbusy=False
			self.responseTime=0
		return

	def generateanimationWords(self,label,amount):
		
		if label !="normal":
			for word in range(int(amount*10.0)):
				wordAnim={"text":random.choice(self.soundNamesLoaded[label]),"speed":random.randint(4,8),"top":random.randint(10,self.h),"left":-random.randint(50,120)}
				self.animationWords.append(wordAnim)
		

	def manageAnalResponse(self,response):
		
		prediction=json.loads(response.text)
		
		if self.overlayMode=="stats":
			for i,pred in enumerate(prediction):
				self.graphs[i].append(pred)
		
		max_value = max(prediction)
		max_index = prediction.index(max_value)
		predicted_label = self.labels[max_index]

		#print ("max_value",max_value)
		
		if self.overlayMode=="words" and max_value>self.minScandalous:
			self.generateanimationWords(predicted_label, max_value)

		#print("%s (%.2f%%)" % (predicted_label, max_value * 100))
		if predicted_label != "normal" and max_value>self.minScandalous:
			self.analyzingCount=0
			position=random.randint(0,len(self.soundsLoaded[predicted_label])-1)
			self.playSound(self.soundsLoaded[predicted_label][position],0)
		
		
		#sleep(0.3)
		self.firstRequestAnswer=True
		self.Analbusy=False
		self.responseTime=0
		return predicted_label 

	def text_to_screen(self,text, size = 50,color = (255, 255, 255), font_type = 'fonts/courierbold.ttf'):
		text = str(text)
		#font = pygame.font.Font(font_type, size)
		font=self.font
		text = font.render(text, True, color)
		return text

	def run_classification(self,input):
		
		cap = cv2.VideoCapture(input)
		
		#self.anal = subprocess.Popen("python3 analyser.py")
		print("wait till analyser is ready")
		sleep(10)
		
		self.Analbusy=False

		if isinstance(input, int ):
			#videocapture
			mode="livecapture"
		else:
			#file mode
			mode="file"

		self.w=int(cap.get(3))
		self.h=int(cap.get(4))
		
		if mode=="file":
			total_frames = cap.get(7)
			fps = cap.get(5)
			#play music
			
	
		if mode=="file":
			FPS = fps

		if self.videomode=="gtk":
			if self.fullscreen:
				cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
				cv2.setWindowProperty(self.windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)       

		if mode=="file":
			audiosource=input.replace("_PAL","").replace(".mp4",".mp3")
			pygame.mixer.music.load(audiosource)
			pygame.mixer.music.play()

		running=True
		


		while running: 
			
			#print (self.responseTime)
			ret, frameCV = cap.read()
			
			if self.videomode=="pygame":
				#self.screen.fill((0,0,0))
				frame=self.cvimage_to_pygame(frameCV)
				frame = pygame.transform.scale(frame, self.screenSize)
				self.screen.blit(frame,(0,0))
			
			if not self.Analbusy and self.analyzing:
				
				#print("call analyzer")
				t = threading.Thread(target=self.callAnalyser, args = (frameCV,))
				t.start()
				#try:
					#self.callAnalyser(frameCV)
					
				#except:
				#	print("couldn't start analyser thread")
			else:
				if self.analyzing:
					self.responseTime+=1
				#fill graph to add speed
				if self.overlayMode=="stats":
					for i,g in enumerate(self.graphs):
						self.graphs[i].append(self.graphs[i][-1])

			


			#graphs
			l=5
			step=l
			stepCount=0

			
			
			if self.overlayMode=="stats":
				lastPoint=(self.w/2,int(0*float(self.h)))
				for c,label in enumerate(self.graphs):
					if self.labels[c] !="normal":
						for d,value in enumerate(reversed(label)):

							currentPoint=int((self.w/2)-stepCount),int(self.h-(int(value*float(self.h))+30))
							if d!=0 and d<len(label)-1:
								cv2.line(frame,(currentPoint),(lastPoint),self.colors[c],1)

							lastPoint=currentPoint

							stepCount+=step
						stepCount=0

						#frame=self.cvimage_to_pygame(frame)
						#cv2.putText(frame,label+":"+str(round(self.labelsData[label],2))+"%", (90,(i*50)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
						cv2.putText(frame,self.labels[c], (int((self.w/2)+20),self.h-int(self.graphs[c][-1]*float(self.h))-27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[c],2)
						
			
			
				

			if self.overlayMode=="words":
				for i,word in enumerate(self.animationWords):
					x=int(word["left"])
					y=int(word["top"])
					text=self.text_to_screen(word["text"], size = 20)
					self.screen.blit(text,(x,y))
					#cv2.putText(frame,word["text"], (int(word["left"]),int(word["top"])), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255))
					self.animationWords[i]["left"]+=word["speed"]
					if self.animationWords[i]["left"]>self.screenSize[0]:
						del self.animationWords[i]


			if self.analyzing:
				self.analyzingCount+=1
				if self.analyzingCount>self.boringLimit:
					#BORINGGG...
					self.generateanimationWords("boring", 0.3)
					self.analyzingCount=0

			if self.videomode=="pygame":
				#FPS=25
				#frame=self.cvimage_to_pygame(frame)
				#frame = pygame.transform.scale(frame, self.screenSize)
				#self.screen.blit(frame,(0,0))
				#self.screen.fill(frame)
				
				pygame.display.update()
				#self.clock.tick(FPS)

			if self.responseTime>200 and self.firstRequestAnswer:
				"""
				self.analx.exit.set()
				time.sleep(3)
				self.analx = multiprocessing.Process(target=self.runAnalyser)
				self.analx.start()
				"""
				#server timeout, force request
				"""
				self.Analbusy=False
				self.responseTime=0
				print ("REST RESPONSE TIME, FORCE REQUEST!")
				time.sleep(1)
				"""
				print ("TOO LONG RESPONSE TIME, RESET ANALYZER!")
				self.Analbusy=False
				self.responseTime=0
				
			if self.videomode=="gtk":
				cv2.imshow(self.windowName, frame)
			
			if self.videomode=="gtk":
				key = cv2.waitKey(30)
				if  key == 27:
					running = False
					break  # esc to quit
					
			if self.videomode=="pygame":
				
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						running = False  # Be interpreter friendly
					elif event.type == pygame.KEYDOWN:
						
						if event.key == pygame.K_ESCAPE:
							running = False

						if event.key == pygame.K_p:
							if self.analyzing:
								self.analyzing=False
							else:
								self.analyzing=True
							print("analyzing:",self.analyzing)

						if event.key == pygame.K_k:
							self.Analbusy=False
							self.responseTime=0
							
					if (event.type is pygame.KEYDOWN and event.key == pygame.K_f):
						if self.screen.get_flags() & pygame.FULLSCREEN:
							pygame.display.set_mode(self.screenSize)
						else:
							pygame.display.set_mode(self.screenSize, pygame.FULLSCREEN)
		
				
		#self.stopAnalyser()
		
		if self.videomode=="gtk":
			cv2.destroyAllWindows()
		if self.videomode=="pygame":
			pygame.quit()
			
		   
	def cvimage_to_pygame(self,image):
		"""Convert cvimage into a pygame image"""
		#return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return pygame.image.frombuffer(image.tostring(), image.shape[1::-1],"RGB")
				
	def playAudio(self,filename,format="webm",delay=0):
		sleep(delay)
		#filename=filename.split(".")[0]
		song = AudioSegment.from_file(filename,format)
		play(song)
			
	def initPygameScreen(self,mode=0):
		if mode==1:
			pygame.init()
			pygame.display.init()
			#self.clock = pygame.time.Clock()
			
			self.screen = pygame.display.set_mode( ( self.screenSize ) )#PAL
			self.font= pygame.font.SysFont("monospace", 40)
		else:
			"Ininitializes a new pygame screen using the framebuffer"
			# Based on "Python GUI in Linux frame buffer"
			# http://www.karoltomala.com/blog/?p=679
			disp_no = os.getenv("DISPLAY")
			if disp_no:
				print ("I'm running under X display = ".format(disp_no))

			# Check which frame buffer drivers are available
			# Start with fbcon since directfb hangs with composite output
			drivers = ['fbcon', 'directfb', 'svgalib']
			found = False
			for driver in drivers:
				# Make sure that SDL_VIDEODRIVER is set
				if not os.getenv('cd '):
					os.putenv('SDL_VIDEODRIVER', driver)
				try:
					pygame.display.init()
					self.clock = pygame.time.Clock()
				except pygame.error:
					print ('Driver:  failed.'.format(driver))
					continue
				found = True
				break

			if not found:
				raise Exception('No suitable video driver found!')

			size = (pygame.display.Info().current_w, pygame.display.Info().current_h)
			self.screenSize=size
			print ("Framebuffer size: %d x %d" % (size[0], size[1]))
			self.screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
			
			pygame.display.update()
		pygame.mouse.set_visible(0)
			
			
		  
if __name__ == '__main__':
			
	ay=aaaay()
	#ay.capture_images_from_file("vandamme.mp4")
	#print (ay.predict_on_image("captures/pouritup_18.jpg"))
	#ay.run_classification("movies/pouritup_small.mp4")#"movies/anaconda.mp4")
	#ay.prepareAudios()
	#ay.downloadAndWatch("https://www.youtube.com/watch?v=d2smz_1L2_0")
	#ay.run_classification("movies/dHULK1M-P08_PAL.mp4")
	#ay.downloadAndWatch("https://www.youtube.com/watch?v=JgffRW1fKDk")
	#ay.downloadAndWatch("https://www.youtube.com/watch?v=ZXIe6pouJTc")
	ay.run_classification(1)#live mode
	print("FINISHED")