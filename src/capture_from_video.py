from moviepy.editor import *
import sys
import os

moviesFolder="movies/"
capturesFolder="captures/"
framerate=10
captureRes=[320,240]

def capture_images_from_file(filename):
		
	clip =VideoFileClip(moviesFolder+filename)
	clip=clip.set_fps(framerate)
	clip=clip.resize( (captureRes) )
	print ("clip duration",clip.duration/60," minutes ::::")
	
	count=0
	for t in range(0,int(clip.duration)):
		if count==2:
			print (capturesFolder+filename.split(".")[0]+"_"+str(t)+".jpg",t)
			clip.save_frame(capturesFolder+filename.split(".")[0]+"_"+str(t)+".jpg", t=t)
			count=0
		count+=1

"""
if len(sys.argv) >= 2:
	for i,file in enumerate(sys.argv):
		if i>0:
			print("downloading "+file)
			capture_images_from_file(file)

else:
	print ("no input suplied")
"""

for root, dirs, files in os.walk(moviesFolder):
	for file in files:
		print("processing "+file)
		if not "procesed" in file:
			try:
				capture_images_from_file(file)
			except:
				print("Not a valid video file")