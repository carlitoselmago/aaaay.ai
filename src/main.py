
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
import pygame
import os
import os.path
from screeninfo import get_monitors

class aaaay():
    
    captureRes=[320,240]
    skip=2 # in seconds
    framerate=10
    moviesFolder="movies/"
    capturesFolder="captures/"
    windowName="aaaay.ai"
    processing=False
    videomode="pygame" #options: gtk,pygame
    graphs=[] 
    colors=[[0,255,255],[255,255,0],[255,0,255],[125,125,0]]
    
    
    def __init__(self):
        
        print ("init aaaay class")
        
        try:
            self.screenSize=get_monitors()[0]
        except:
            print("could not determine screenSize")
        self.labels=self.get_labels()
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
            
        
    
    def capture_images_from_file(self,filename):
        from moviepy.editor import *
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
        
    def runCommand(self,some_command):
        import subprocess

        #This command could have multiple commands separated by a new line \n
        #some_command = "export PATH=$PATH://server.sample.mo/app/bin \n customupload abc.txt"

        p = subprocess.Popen(some_command, stdout=subprocess.PIPE, shell=True)

        (output, err) = p.communicate()  

        #This makes the wait possible
        p_status = p.wait()

        #This will give you the output of the command being executed
        print "Command output: " + output

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
            print ffmpegCommand
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
        audioFilename=videofilename.split(".")[0]+".mp3"
        tA = threading.Thread(target=self.playAudio, args = (audioFilename,"mp3",1))
        tA.start()
        
        #self.playAudio(videofilename)
        FPS = fps

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            if self.videomode=="gtk":
                cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(self.windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)       
           
            running=True
            while running:  # note that we don't have to use frame number here, we could read from a live written file.
                start_time = time.time()
                ret, frame = cap.read()
                #print (ret,frame)
                #ACTION
                  
                #print (frame)
                #sys.exit()
                # Get the numpy version of the image.
                
                
                
                if not self.processing:
                    t = threading.Thread(target=self.analysiseFrame, args = (sess,softmax_tensor,frame))
                    #t.daemon = True
                    t.start()
                    pass
                   
               
                #END ACTION 
                #cap.set(1,currentTimeFrame) #frames to skip
                
                #paint frame
                
                #for i,label in enumerate(self.labelsData):
                    
                #graphs
                w=self.screenSize.width
                h=(self.screenSize.height)-300
                l=5
                step=l
                stepCount=0
                print (self.graphs)
                print ("LELELELE")
                lastPoint=(w/2,int(0*float(h)))
                for c,label in enumerate(self.graphs):
                    if self.labels[c] !="normal":
                        for d,value in enumerate(reversed(label)):

                            """
                            cv2.line(frame,
                             ((w/2)-stepCount,int(value*float(h))),
                            (((w/2)-stepCount)+l,int(value*float(h)))
                            ,self.colors[c],1)
                            """
                            currentPoint=(w/2)-stepCount,h-(int(value*float(h))+30)
                            if d!=0 and d<len(label)-1:
                                cv2.line(frame,(currentPoint),(lastPoint),self.colors[c],1)

                            lastPoint=currentPoint

                            stepCount+=step
                        stepCount=0
                        #cv2.putText(frame,label+":"+str(round(self.labelsData[label],2))+"%", (90,(i*50)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
                        cv2.putText(frame,self.labels[c], ((w/2)+20,h-int(self.graphs[c][-1]*float(h))-27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[c])
                if self.videomode=="pygame":
                    
                    frame=self.cvimage_to_pygame(frame)
                    frame = pygame.transform.scale(frame, self.screenSize)
                    self.screen.blit(frame,(0,0))
                    #self.screen.fill(frame)
                    pygame.display.update()
                    self.clock.tick(FPS)
                    
                if self.videomode=="gtk":
                    cv2.imshow(self.windowName, frame)
                
                #cv2.setTrackbarPos("time", self.windowName, int(currentTimeFrame)) 


                #cv2.waitKey(int(fps*1000)) # time to wait between frames, in mSec
                #ret, frame = cap.read() # read next frame, get next return code
                if self.videomode=="gtk":
                    key = cv2.waitKey(50)
                    if  key == 32:
                        running = False
                        break  # esc to quit
                        
                if self.videomode=="pygame":
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False  # Be interpreter friendly
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                                
                        if (event.type is pygame.KEYDOWN and event.key == pygame.K_f):
                            if self.screen.get_flags() & pygame.FULLSCREEN:
                                pygame.display.set_mode(self.screenSize)
                            else:
                                pygame.display.set_mode(self.screenSize, pygame.FULLSCREEN)

                    totalTime=(time.time() - start_time)
                    #print (totalTime)
                    #totalTime=0.1
                    currentTimeFrame=cap.get(1)+(totalTime*fps)
                    #cap.set(1,currentTimeFrame) #frames to skip

                

            if self.videomode=="gtk":
                cv2.destroyAllWindows()
            if self.videomode=="pygame":
                pygame.quit()
            
           
    def cvimage_to_pygame(self,image):
        """Convert cvimage into a pygame image"""
        return pygame.image.frombuffer(image.tostring(), image.shape[1::-1],
                                       "RGB")
                                       
    def analysiseFrame(self,sess,softmax_tensor,decoded_image):
        self.processing=True
        # Make the prediction. Big thanks to this SO answer:
        # http://stackoverflow.com/questions/34484148/feeding-image-data-in-tensorflow-for-transfer-learning
        predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': decoded_image})
        self.labelsData=dict(zip(self.labels, predictions[0]))
        for i,pred in enumerate(predictions[0]):
            print ("pred",pred)
            self.graphs[i].append(pred)
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
        #filename=filename.split(".")[0]
        song = AudioSegment.from_file(filename,format)
        play(song)
            
    def initPygameScreen(self,mode=0):
        if mode==1:
            pygame.display.init()
            self.clock = pygame.time.Clock()
            self.screenSize= (720, 576)
            self.screen = pygame.display.set_mode( ( self.screenSize ) )#PAL
        else:
            "Ininitializes a new pygame screen using the framebuffer"
            # Based on "Python GUI in Linux frame buffer"
            # http://www.karoltomala.com/blog/?p=679
            disp_no = os.getenv("DISPLAY")
            if disp_no:
                print "I'm running under X display = {0}".format(disp_no)

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
                    print 'Driver: {0} failed.'.format(driver)
                    continue
                found = True
                break

            if not found:
                raise Exception('No suitable video driver found!')

            size = (pygame.display.Info().current_w, pygame.display.Info().current_h)
            self.screenSize=size
            print "Framebuffer size: %d x %d" % (size[0], size[1])
            self.screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
            # Clear the screen to start
            #self.screen.fill((0, 0, 0))        
            # Initialise font support
            #pygame.font.init()
            # Render the screen
            pygame.display.update()
        pygame.mouse.set_visible(0)
            
            
          

        
ay=aaaay()
#ay.capture_images_from_file("pouritup.mp4")
#ay.capture_images_from_file("vandamme.mp4")
#print (ay.predict_on_image("captures/pouritup_18.jpg"))
#ay.run_classification("movies/pouritup_small.mp4")#"movies/anaconda.mp4")
ay.downloadAndWatch("https://www.youtube.com/watch?v=dHULK1M-P08")
print("FINISHED")