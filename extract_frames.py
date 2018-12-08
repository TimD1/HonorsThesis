import segment
import imageio
import scipy.misc
import os
import shutil
import sys
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from operator import itemgetter

# define color-based constants
SWIPE_THRESHOLD = 20 # constant threshold for image binarization
WHITE = 255
BLACK = 0

def save_swipe(orig_image, filename):
	"""Extracts the swipe from grayscale <image> and saves as <filename>. """

	# define kernel sizes
	kernel1 = np.ones((3,3), np.uint8)
	kernel2 = np.ones((25,1), np.uint8)
	kernel3 = np.ones((11,11), np.uint8)
	kernel4 = np.ones((1,25), np.uint8)
	kernel5 = np.ones((7,7), np.uint8)

	# threshold image
	image = orig_image.copy()
	swipe = image[:,:] > SWIPE_THRESHOLD
	back = image[:,:] <= SWIPE_THRESHOLD
	image[swipe] = WHITE
	image[back] = BLACK

	# remove salt-and-pepper noise
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)

	# dilate image so that when swipe is removed, we err on the side        
	# of removing section of table (therefore considered part of swipe)        
	# better to accidentally include near-zero region than part of hand        
	image = cv2.dilate(image, kernel3, iterations=1)                        

	# remove thin horizontal or vertical lines                              
	no_swipe = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2)             
	no_swipe = cv2.morphologyEx(no_swipe, cv2.MORPH_OPEN, kernel4)          

	# subtract frame w/o swipe from original frame to get swipe             
	swipe = image - no_swipe                                                

	# remove gradient around hand                                           
	swipe = cv2.morphologyEx(swipe, cv2.MORPH_OPEN, kernel5)                

	# use swipe region as mask over original video                          
	inv_mask = swipe[:,:] == BLACK                                          
	image = orig_image.copy()                                               
	image[inv_mask] = BLACK   

	# save extracted swipe image
	scipy.misc.toimage(image).save(filename)



def make_re(pressures, mats):
	""" Generate a regular expression for swipe file names, 
	given list of pressures and materials being considered. """

	# start pressure group
	re_string = r"("
	for p in pressures:
		re_string += p + "|"

	# remove last |, finish group, expect _, start mat group
	re_string = re_string[:-1] + ")_("
	for m in mats:
		re_string += m+"|"

	# remove last |, finish group, num group, file ext
	re_string = re_string[:-1]+")([0-9]+).mov"
	return re_string


def extract_frames(folder, users, mats, pressures, fpv, swipe):

	# clear directory "user/frames/" which stores extracted frames
	print('Clearing "frames/" directories...')
	for u in users:
		if os.path.exists(folder+"/"+u+"/frames/"):
			shutil.rmtree(folder+"/"+u+"/frames/")
		for m in mats:
			for p in pressures:
				os.makedirs(folder+"/"+u+"/frames/"+m+"/"+p+"/")
	print('All frames/ directories cleared.')

	# clear directory "user/swipe_frames/" which stores preprocessed frames
	if swipe:
		print('Clearing "swipe_frames/" directories...')
		for u in users:
			if os.path.exists(folder+"/"+u+"/swipe_frames/"):
				shutil.rmtree(folder+"/"+u+"/swipe_frames/")
			for m in mats:
				for p in pressures:
					os.makedirs(folder+"/"+u+"/swipe_frames/"+m+"/"+p+"/")
		print('All swipe_frames/ directories cleared.')

	# determine which frames to use for further analysis
	print('Extracting relevant frames from each swipe video...')
	for u in users:
		print('\tExtracting frames for user', u)
		for f in os.listdir(folder+"/"+u+"/segments/"):

			# extract mat, pressure, and id from filename
			match = re.match(make_re(pressures, mats), f, re.I)
			if match:
				p = match.groups()[0]
				m = match.groups()[1]
				n = match.groups()[2]

				# set up image reader
				filename = folder + "/" + u + "/segments/" + f
				reader = imageio.get_reader(filename, 'ffmpeg')

				# calculate total frames and average frame intensity
				total_intensity = 0
				frame_count = 0
				for i, image in enumerate(reader):
					frame_count += 1
					image = np.array(image).astype(np.uint8)[:,:,0].astype(np.float32)
					total_intensity += np.sum(image)
				avg_intensity = total_intensity // frame_count

				# save all frames brighter than average (user's hand is in video)
				high_intensity_frames = []
				for i, image in enumerate(reader):
					image = np.array(image).astype(np.uint8)[:,:,0].astype(np.float32)
					if np.sum(image) > avg_intensity:
						high_intensity_frames.append(i)

				# set times when hand enters and leaves video
				hand_enters = min(high_intensity_frames)
				hand_leaves = min(hand_enters + 42, max(high_intensity_frames))

				# ignore first third since user isn't swiping yet
				# this is approximately the start and stop of the swipe
				start = hand_enters + (hand_leaves-hand_enters) // 3
				stop = hand_leaves
				if stop-start < frames_per_vid:
					print('\t\tWARNING: only '+str(stop-start)+' frames extracted from '+f)
					print('\t\tPlease verify that there are no errors in this video')
					continue

				# from within the swipe, take the middle fpv frames
				first = start + (stop-start-frames_per_vid)//2
				last = first + frames_per_vid

				# extract and save relevant frames
				for i, image in enumerate(reader):
					if i >= first and i < last:
						idx = i-first+1
						image = np.array(image).astype(np.uint8)[:,:,0]
						scipy.misc.toimage(image).save(folder+"/"+u+"/frames/"+
								m+"/"+p+"/"+n+"_"+str(idx)+".jpg")
						if swipe:
							save_swipe(image, folder+"/"+u+"/swipe_frames/"+
									m+"/"+p+"/"+n+"_"+str(idx)+".jpg")
	print('Frames have been extracted from each video.')
	if swipe:
		print('Swipe frames have been extracted as well.')

	

if __name__ == '__main__':

	# choose folder to operate on
	if len(sys.argv) == 1:
		folder = "data"
	elif len(sys.argv) == 2:
		folder = sys.argv[1]
	else:
		print('Usage: \n\tpython extract_frames.py <folder>')
		sys.exit(1)

	# set parameters for this script
	users = ["alex", "ben", "miao", "natasha", "nick", "sarah", 
			"sean", "spencer", "tim", "yijun"]
	mats = ["cloth", "concrete", "door", "drywall", "laminant", "whiteboard"]
	pres = ["hard", "soft"]
	frames_per_vid = 8

	# extract 8 frames per video
	extract_frames(folder, users, mats, pres, fpv=frames_per_vid, swipe=True)
