import imageio
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys

# color-based constants
WHITE = 255
HAND_CUTOFF = 20
BLACK = 0

# video segment lengths, in frames
MIN_SEGMENT_LENGTH = 25
MAX_SEGMENT_LENGTH = 150

# image width/height, bounding box placement
HEIGHT = 512
WIDTH = 640
Y0, Y1 = 224, 444
X0, X1 = 168, 376


# check input arguments, set filepaths
if len(sys.argv) != 2:
	print("Usage: python segment.py <filename>")
	sys.exit()
filename, filetype = os.path.splitext(sys.argv[1])
folder, filename = os.path.split(filename)
if len(folder) > 0:
	folder = folder+"/"
if not os.path.isfile(folder+filename+filetype):
	print("ERROR: File", folder+filename+filetype, "does not exist.")
	sys.exit()
if not os.path.exists(folder+"segments"):
	os.makedirs(folder+"segments")

# set flags
last_new_segment = 0	# frame at which segment began
was_low = False			# if hand left image since segment began
low_count = 0			# consecutive frames for which hand was gone
segment = 0				# segment id number

# initialize summary/segment writers, set background as first frame
reader = imageio.get_reader(folder+filename+filetype, 'ffmpeg')
fps = reader.get_meta_data()['fps']
nframes = reader.get_meta_data()['nframes']
segment_writer = imageio.get_writer(folder+"segments/"+filename+str(segment)+filetype, 
		'ffmpeg', fps=fps, macro_block_size=None)
segment_writer.close()
summary_writer = imageio.get_writer(folder+filename+"_summary"+filetype, 
		'ffmpeg', fps=fps*2)
background = np.array(reader.get_data(0)).astype(int)[:,:,0]

# process video and segment
for i, image in enumerate(reader):

	# background subtract, threshold at zero
	image = np.array(image).astype(int)[:,:,0]
	image = np.maximum(image - background, np.zeros(image.shape))

	# check if at least 10 bottom pixels belong to a hand
	if(np.sum(image[Y1:Y1+1, X0:X1] > HAND_CUTOFF) > 3):

		# if hand just entered image and segment was long enough, start new segment
		if(i - last_new_segment > MIN_SEGMENT_LENGTH and was_low):
			if not segment_writer.closed:
				segment_writer.close()
			segment += 1
			segment_writer = imageio.get_writer(
					folder+"segments/"+filename+str(segment)+filetype, 
					'ffmpeg', fps=fps, macro_block_size=None)
			last_new_segment = i
			was_low = False
			low_count = 0

	else: # hand isn't in image, after 1/10 second decide it has left
		low_count += 1
		if low_count >= 3:
			was_low = True
		
	# segment has reached maximum length, end it
	if i - last_new_segment > MAX_SEGMENT_LENGTH:
		if not segment_writer.closed:
			segment_writer.close()

	# add border around bounding area which is captured 
	image[Y0-1,X0:X1] = WHITE*np.ones(X1-X0)
	image[Y1,X0:X1] = WHITE*np.ones(X1-X0)
	image[Y0:Y1,X0-1] = WHITE*np.ones(Y1-Y0)
	image[Y0:Y1,X1] = WHITE*np.ones(Y1-Y0)

	# record with segment/summary writers
	if not segment_writer.closed:
		segment_writer.append_data(image[Y0:Y1,X0:X1].astype('uint8'))
		summary_writer.append_data(image.astype('uint8'))

	else: # add lines to indicate not recording, add to summary writer
		for x in range(X0, X1, 10):
			image[Y0:Y1,x] = WHITE*np.ones(Y1-Y0)
		summary_writer.append_data(image.astype('uint8'))

	# display progress processing video
	if i % 100 == 0:
		percent = (i / nframes)
		bars = percent*40
		sys.stdout.write("\rProgress: [{0}{1}] {2}%".format("|"*int(bars), " "*int(40-bars), int(percent*100)))
		sys.stdout.flush()

# close writers
print("")
segment_writer.close()
summary_writer.close()
