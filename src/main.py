from moviepy.editor import *
import time
import tensorflow as tf
import cv2
import numpy as np
import sys
import threading
from pydub import AudioSegment
from pydub.playback import play
import ntpath
from time import sleep

class aaaay():
    
    captureRes=[320,240]
    skip=2 # in seconds
    framerate=10
    moviesFolder="movies/"
    capturesFolder="captures/"
    windowName="aaaay.ai"
    processing=False
    
    def __init__(self):
        
        print ("init aaaay class")
        try:
            self.labels=self.get_labels()
        except:
            print("No labels where created yet")
    
    def capture_images_from_file(self,filename):
        clip = VideoFileClip(self.moviesFolder+filename)
        clip=clip.set_fps(self.framerate)
        clip.resize( (self.captureRes) )
        print ("clip duration",clip.duration/60," minutes ::::")
        
        count=0
        for t in range(0,int(clip.duration)):
            if count==2:
                print (self.capturesFolder+filename.split(".")[0]+"_"+str(t)+".jpg",t)
                clip.save_frame(self.capturesFolder+filename.split(".")[0]+"_"+str(t)+".jpg", t=t)
                count=0
            count+=1
        
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
        
    def run_classification(self,videofilename):
        
        #resize video
        """
        clip = VideoFileClip(videofilename)
        clip=clip.set_fps(self.framerate)
        clip.resize( (self.captureRes) )
        clip.write_videofile("test.mp4")
        """
        
        """

        camera = PiCamera()
        camera.resolution = (320, 240)
        camera.framerate = 2
        rawCapture = PiRGBArray(camera, size=(320, 240))

        # Warmup...
        time.sleep(2)
        """
        cap = cv2.VideoCapture(videofilename)
        #cap = cv2.VideoCapture("test.mp4")
        
        total_frames = cap.get(7)
        fps = cap.get(5)

        # Unpersists graph from file
        with tf.gfile.FastGFile("train/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            
        #play audio
        tA = threading.Thread(target=self.playAudio, args = (videofilename,"webm",1))
        tA.start()
        
        #self.playAudio(videofilename)
        

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)       
           
            while(cap.isOpened()):  # note that we don't have to use frame number here, we could read from a live written file.

                
                ret, frame = cap.read()
                
                #print (ret,frame)
                start_time = time.time()
                
                #ACTION
                  
                #print (frame)
                #sys.exit()
                # Get the numpy version of the image.
                decoded_image = frame#image.array
                
                
                if not self.processing:
                    t = threading.Thread(target=self.analysiseFrame, args = (sess,softmax_tensor,decoded_image))
                    #t.daemon = True
                    t.start()
                    pass
                   
               
                #END ACTION 


                totalTime=(time.time() - start_time)
                
                #print totalTime

                currentTimeFrame=cap.get(1)+(totalTime*fps)

                #cap.set(1,currentTimeFrame) #frames to skip



                cv2.imshow(self.windowName, frame)

                #cv2.setTrackbarPos("time", self.windowName, int(currentTimeFrame)) 


                #cv2.waitKey(int(fps*1000)) # time to wait between frames, in mSec
                #ret, frame = cap.read() # read next frame, get next return code

                key = cv2.waitKey(50)
                if  key == 32: 
                    break  # esc to quit
                


            cv2.destroyAllWindows()
            
    def analysiseFrame(self,sess,softmax_tensor,decoded_image):
        self.processing=True
        # Make the prediction. Big thanks to this SO answer:
        # http://stackoverflow.com/questions/34484148/feeding-image-data-in-tensorflow-for-transfer-learning
        predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': decoded_image})
        prediction = predictions[0]

        # Get the highest confidence category.
        prediction = prediction.tolist()
        max_value = max(prediction)
        max_index = prediction.index(max_value)
        predicted_label = self.labels[max_index]

        print("%s (%.2f%%)" % (predicted_label, max_value * 100))
        if predicted_label != "normal":
            self.playAudio("sounds/gritoarus.m4a","m4a")
        self.processing=False
        # Reset the buffer so we're ready for the next one.
        #rawCapture.truncate(0)
        
    def playAudio(self,filename,format="webm",delay=0):
        sleep(delay)
        filename=filename.split(".")[0]
        song = AudioSegment.from_file(filename+"."+format,format)
        play(song)
            
            
            
          

        
ay=aaaay()
#ay.capture_images_from_file("pouritup.mp4")
#ay.capture_images_from_file("vandamme.mp4")
#print (ay.predict_on_image("captures/pouritup_18.jpg"))
ay.run_classification("movies/pouritup_small.mp4")#"movies/anaconda.mp4")
print("FINISHED")