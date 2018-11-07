import segment
import imageio
import scipy.misc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from operator import itemgetter

# set globals for this script
folder = "data"
users = ["alex", "ben", "miao", "natasha", "nick", "sarah", "sean", "spencer", "tim", "yijun"]

# define various entry points to script
class StartAt(IntEnum):
	SEGMENTS = 1
	FRAMES = 2
start = StartAt.SEGMENTS

# validate command-line parameters
if len(sys.argv) == 2:
	if sys.argv[1] == "segments":
		start = StartAt.SEGMENTS
	elif sys.argv[1] == "frames":
		start= StartAt.FRAMES
	else:
		print('Usage: \n\tpython main.py <start>')
		print('\t\twhere <start> is one of: "segments", "frames"')
		sys.exit(1)
else:
	print('Usage: \n\tpython main.py <start>')
	print('\t\twhere <start> is one of: "segments", "frames"')
	sys.exit(1)


if start <= StartAt.SEGMENTS:
	# clear directory "user/frames/" which stores extracted frames
	print('Clearing "frames/" directories...')
	for u in users:
		if not os.path.exists(folder+"/"+u+"/frames/"):
			os.makedirs(folder+"/"+u+"/frames/")

		for f in os.listdir(folder+"/"+u+"/frames/"):
			os.remove(folder+"/"+u+"/frames/"+f)
	print('All frames/ directories cleared.')

	# determine which frames to use for further analysis
	print('Extracting relevant frames from each swipe video...')
	for u in users:
		print('\tExtracting frames for user', u)
		for f in os.listdir(folder+"/"+u+"/segments/"):

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

			high_intensity_frames = []
			for i, image in enumerate(reader):
				image = np.array(image).astype(np.uint8)[:,:,0].astype(np.float32)
				if np.sum(image) > avg_intensity:
					high_intensity_frames.append(i)

			# select frames during user swipe
			first = min(high_intensity_frames)
			last = min(first + 42, max(high_intensity_frames))
			for i, image in enumerate(reader):
				if i >= (first + (last-first)//3) and i < last:
					image = np.array(image).astype(np.uint8)[:,:,0]
					scipy.misc.toimage(image).save(folder+"/"+u+"/frames/"+
							os.path.splitext(f)[0]+"_"+str(i)+".jpg")
	print('Frames have been extracted from each video..')
