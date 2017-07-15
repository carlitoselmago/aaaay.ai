import multiprocessing
from multiprocessing import Array, Value
from time import sleep
import numpy as np
import cv2


def run_classification():
	running=True
	while running:
		ret, frame = cap.read()
		sharedFrame.value=frame
		count.value+=1
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow("hola", frame)
		print (count.value)
		
		key = cv2.waitKey(1)
		if  key == 32:
			running = False
			x.terminate()
			break  # esc to quit


def analysiseFrame(process_name, count,sharedFrame):
	while True:
		
		#frame=np.frombuffer(sharedFrame[:])
		#print(frame,"KKK")
		frame=sharedFrame[:]
		#frame= np.rollaxis(frame, axis=2, start=0)
		
		frame=np.array(frame,dtype=np.uint8).reshape((480, 640, 3))
		
		#print(frame,"KKK")
		print(frame.dtype,"KKK")
		gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#gray_image=frame
		cv2.imshow("gray", gray_image)

		key = cv2.waitKey(1)
		if  key == 32:
			running = False
			x.terminate()
			break  # esc to quit
		count.value +=100
		#print (process_name,count)
		sleep(3)
		
		#return count



if __name__ == '__main__':
	
	cap = cv2.VideoCapture(0)
	ret, tempframe = cap.read()
	print(tempframe.dtype)
	tempframe=tempframe.flatten()
	print (tempframe)
	count = multiprocessing.Value('i',0)  # (type, init value)
	sharedFrame = multiprocessing.Array('i', tempframe)  # (type, init value)
	x = multiprocessing.Process(target=analysiseFrame, args=("analysiseFrame", count,sharedFrame))
	x.start()

	run_classification()