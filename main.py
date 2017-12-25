#!/usr/bin/python 
import multiprocessing
import time
import libpysunergy
import cv2
import os
from os import listdir
from os.path import isfile, join
import math
import numpy
import sys
import fcntl

company_id = 777
prop_id = 777

frame_interval = 3
top = 1
video_root = '/home/zhao/Desktop/face_beijing/'
# for 8G GPU memory
thread_num = 2
gpu_num = 4

threshold = 0.24


root = './'

def estconn():
	#connection to database
	conn = MySQLdb.connect(
		host = '52.162.166.103',
		user = 'vmaxx',
		passwd = 'Xjtu123456',
		db = 'age_gender',
		charset = 'utf8'
	)
	return conn

class Worker(object):
	def start(self):
		record = []
		man = multiprocessing.Manager()

		print "create worker"
		print "generating file list:"
		file_list = numpy.array([]) 
		for dirpath, dirnames, filenames in os.walk(video_root):
			for f in filenames:
				if isfile(join(dirpath, f)):
					file_list = numpy.append(file_list,join(dirpath, f))

		l = numpy.array_split(file_list,thread_num*gpu_num)
		

		print "start to create threads:"
		for i in range(0,gpu_num):
			for j in range(0, thread_num):
				mx = man.Value('gpu_num',i)
				ml = man.list()
				for k in range(0,len(l[i*thread_num + j])):
					ml.append(l[i*thread_num + j][k])

				process = multiprocessing.Process(target=service,args=(mx,ml))
				process.daemo = True
				process.start()
				record.append(process)

		for process in record:
			process.join()

def insert_db(age_file,gender_file,f):
	lock = open(video_root + 'age.lock','w')
	fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

	result = open(video_root + 'result.txt','a')
	#filename,age,male,female
	#result.write('filename,<20,20-25,25-30,30-35,35-40,40-45,45-50,50-55,55-60,>60,male,female')
	result.write(f + ',')
	for age in age_file:
		result.write(str(age_file[age])+',')

	result.write(str(gender_file['male'])+',')
	result.write(str(gender_file['female'])+'\n')
	result.close()

	lock.close()


	'''
	#convert to ratio
	age_sum = 0
	for age in age_file:
		age_sum += age_file[age]

	for age in age_file:
		age_file[age] /= age_sum

	gender_sum = 0
	for gender in gender_file:
		gender_sum += gender_file[gender]

	for gender in gender_file:
		gender_file[gender] /= gender_sum



	cam_id = f.split('_')[1]
	db_date_time = cam.split('_')[2] + "_" + cam.split('_')[3]

	conn = estconn()
	cur = conn.cursor()

	cur.execute('insert into `camera` (`company_id`, `property_id`, `object_id`, `type`,`male`, `female`, `date_time`) \
				VALUES(\''+ str(company_id) + '\', \''+ str(prop_id) + '\', \'' + str(cam_id) + '\', \'hourly\',\'' + str(db_date_time) \
				+'\',\'' + str(total_cam_trip) +'\',\'' + str(pure_cam_unique) \
				+'\',\'' + str(delta_cam_unique) + '\') ON DUPLICATE KEY UPDATE \
				total_count = total_count + values(total_count), pure_unique = pure_unique + values(pure_unique), \
				delta_unique = delta_unique + values(delta_unique)')
	'''
	

def static_init():
	age = {}
	gender = {}
	
	age['<20'] = 0
	age['20-25'] = 0
	age['25-30'] = 0
	age['30-35'] = 0
	age['35-40'] = 0
	age['40-45'] = 0
	age['45-50'] = 0
	age['50-55'] = 0
	age['55-60'] = 0
	age['>60'] = 0

	gender['male'] = 0
	gender['female'] = 0
	return age, gender

def service(x,l):
	#sunergy classifier
	net0,names0 = libpysunergy.load(root + "data/face.data", root + "cfg/yolo_face1.1.cfg", root + "weights/yolo_face1.1.weights",x.value)
	net1,names1 = libpysunergy.load(root + "data/age1.1.data", root + "cfg/age1.1.cfg", root + "weights/age1.1.weights",x.value)
	net2,names2 = libpysunergy.load(root + "data/gender1.1.data", root + "cfg/gender1.1.cfg", root + "weights/gender1.1.weights",x.value)

	for f in l:
		t0 = time.time()
		age_file, gender_file = static_init()
		cap = cv2.VideoCapture(f)
		
		frame_count = -1
		while(True):
			# Capture frame-by-frame
			ret, frame = cap.read()
			if not ret:
				break

			frame_count += 1
			if frame_count % frame_interval is not 0:
				continue

			age_frame,gender_frame = predict(net0, net1, net2, names0, names1, names2, frame)

			for age in age_frame:
				age_file[age] += age_frame[age]
			for gender in gender_frame:
				gender_file[gender] += gender_frame[gender]

			

		
		insert_db(age_file,gender_file,f)
		t1 = time.time()
		print f
		print t1-t0


	
	'''
	print x.value
	'''
	

def predict(net0, net1, net2, names0, names1, names2, frame):
	age_frame, gender_frame = static_init()
	try:

		(img_h, img_w, img_c) = frame.shape
		img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		cfg_size = (1024, 1024)
		img_input = cv2.resize(img_rgb, cfg_size)


		faces = libpysunergy.detect(img_input.data, img_w, img_h, img_c, threshold, net0, names0)

		facenum = len(faces)



		#{py_names[class], prob, left, right, top, bot}
		for i in range(0,facenum):
			[x0,x1,y0,y1] = [faces[i][2], faces[i][3], faces[i][4], faces[i][5]]

			x0 = max(x0,0)
			y0 = max(y0,0)
			x1 = min(x1, img_w-1)
			y1 = min(y1,img_h-1)

			faceimg = img_rgb[y0:y1, x0:x1].copy()
			
			(h,w,c) = faceimg.shape
			#get result from sunergy
			dets1 = libpysunergy.predict(faceimg.data, w, h, c, top, net1, names1)
			age = int(str(dets1[0][0]))	

			dets2 = libpysunergy.predict(faceimg.data, w, h, c, top, net2, names2)
			gender = dets2[0][0]

			if gender == 'male':
				gender_frame['male'] += 1
			elif gender == 'female':
				gender_frame['female'] += 1

			if int(age) < 20:
				age_frame['<20'] += 1

			if int(age) >= 20 and int(age) <25:
				age_frame['20-25'] += 1

			if int(age) >= 25 and int(age) <30:
				age_frame['25-30'] += 1

			if int(age) >= 30 and int(age) <35:
				age_frame['30-35'] += 1

			if int(age) >= 35 and int(age) <40:
				age_frame['35-40'] += 1

			if int(age) >= 40 and int(age) <45:
				age_frame['40-45'] += 1

			if int(age) >= 45 and int(age) <50:
				age_frame['45-50'] += 1

			if int(age) >= 50 and int(age) <55:
				age_frame['50-55'] += 1

			if int(age) >= 55 and int(age) <60:
				age_frame['55-60'] += 1

			if int(age) >= 60:
				age_frame['>60'] += 1

		return age_frame, gender_frame

	except Exception, e:
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
	

if __name__ == '__main__':
	worker = Worker()
	worker.start()
	
