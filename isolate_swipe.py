import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from enum import IntEnum
import pandas as pd
import os.path
import sys
import re
import scipy.misc
import scipy.integrate

# define constants
WHITE = 255
SWIPE_THRESHOLD = 4
SWIPE_RADIUS = 10
REMOVE_N_BOT = 3
GRAY = 128
BLACK = 0
N_BINS = 10
MAX_FRAMES = 150
EPSILON = 0.00001

# define pressure enum
class Pressure(IntEnum):
	HARD = 0
	SOFT = 1

# define material enum
class Material(IntEnum):
	CLOTH      = 0
	CONCRETE   = 1
	DOOR       = 2
	DRYWALL    = 3
	LAMINANT   = 4
	WHITEBOARD = 5

# make name and associated data containers accessible by enum index
realnames = ["alex", "ben", "miao", "natasha", "nick", "sarah", "sean", "spencer", "tim", "yijun"]
names = ["user"+str(i) for i in range(10)]
pressure_names = ["hard", "soft"]
material_names = ["cloth", "concrete", "door", "drywall", "laminant", "whiteboard"]
avg_dicts     = [[{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}]] # 6x2 list of dicts [material][pressure]
profile_dicts = [[{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}]] # 6x2 list of dicts [material][pressure]
timing_dicts  = [[{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}]] # 6x2 list of dicts [material][pressure]
for name in names:
	for i in range(len(material_names)):
		for j in range(len(pressure_names)):
			avg_dicts[i][j][name]     = []
			profile_dicts[i][j][name] = []
			timing_dicts[i][j][name]  = []
		

# create massive global dataframe containing swipe info
swipe_idx = 0
avgbin_columns = ["avgbin" + str(i) for i in range(N_BINS)]
maxbin_columns = ["maxbin" + str(i) for i in range(N_BINS)]
bin_columns = avgbin_columns + maxbin_columns
swipes = pd.DataFrame(columns = ['name', 'pressure', 'material', 'length', 'angle', 'deviation', 'avg', 'max']+bin_columns)

# create massive global dataframe containing timing info
timing_idx = 0
timebin_columns = ["timebin" + str(i) for i in range(N_BINS)]
timing = pd.DataFrame(columns = ['total_time', 'swipe_time', 'fade_time']+timebin_columns)


def process_video(folder, studentname, filename, outfile):
	# print(folder, studentname, filename, outfile)

	# set pressure based on video name
	pressure = 0
	if filename[:4] == "soft":
		pressure = Pressure.SOFT
	elif filename[:4] == "hard":
		pressure = Pressure.HARD
	else:
		print("ERROR: video doesn't follow pressure naming convention")
		sys.exit()

	# set material based on video name
	material = 0
	if filename[5:10] == "cloth":
		material = Material.CLOTH
	elif filename[5:13] == "concrete":
		material = Material.CONCRETE
	elif filename[5:9] == "door":
		material = Material.DOOR
	elif filename[5:12] == "drywall":
		material = Material.DRYWALL
	elif filename[5:13] == "laminant":
		material = Material.LAMINANT
	elif filename[5:15] == "whiteboard":
		material = Material.WHITEBOARD
	else:
		print("ERROR: video doesn't follow material naming convention")
		sys.exit()

	# initialize file reader and writer
	reader = imageio.get_reader(folder+filename, 'ffmpeg')
	fps = reader.get_meta_data()['fps']
	nframes = reader.get_meta_data()['nframes']
	writer = imageio.get_writer(folder+outfile, 'ffmpeg', 
			fps=fps, macro_block_size=None)

	# create matrices to store info
	pixel_sums = np.zeros((reader.get_meta_data()['size'][1]-REMOVE_N_BOT, 
		reader.get_meta_data()['size'][0]))

	# initialize morphological filters
	kernel1 = np.ones((3,3), np.uint8)
	kernel2 = np.ones((25,1), np.uint8)
	kernel3 = np.ones((11,11), np.uint8)
	kernel4 = np.ones((1,25), np.uint8)
	kernel5 = np.ones((7,7), np.uint8)

	# transform each frame
	for i, orig_image in enumerate(reader):

		# threshold image
		orig_image = np.array(orig_image).astype(np.uint8)[:,:,0]
		image = orig_image.copy()
		swipe = image[:,:] > SWIPE_THRESHOLD
		back = image[:,:] <= SWIPE_THRESHOLD
		image[swipe] = WHITE
		image[back] = BLACK

		# remove salt and pepper noise
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
		writer.append_data(image.astype('uint8'))

		# save pixel summation over time for swipe identification
		# ignore last few rows: as the hand enters the image, 
		# it appears to be a thin swipe
		pixel_sums += image[:-REMOVE_N_BOT,:]

	# close video writer
	writer.close()

	# create 2D matrix which summarizes video
	# summary = np.square(pixel_sums)
	summary = pixel_sums.copy()
	# cv2.imwrite(folder+os.path.splitext(filename)[0]+".jpg", summary)

	# generate list of nonzero points
	n_pts = (summary != 0).sum()
	X = np.zeros(n_pts)
	Y = np.zeros(n_pts)
	W = np.zeros(n_pts)
	i = 0
	for x in range(summary.shape[1]):
		for y in range(summary.shape[0]):
			if summary[y][x] != 0:
				X[i] = x
				Y[i] = y
				W[i] = summary[y][x]
				i += 1


	#########################
	### LINEAR REGRESSION ###
	#########################
	b = 0; m = 0
	color1 = np.max(summary)
	color2 = color1/2
	if n_pts < 3:
		return

	[b, m] = np.polynomial.polynomial.polyfit(X, Y, deg = 1, w = W*W)

	# # plot line of best fit on image
	# for x in range(summary.shape[1]):
	# 	y = int(m*x+b)
	# 	if y >= 0 and y < summary.shape[0]:
	# 		summary[y][x] = color2


	############################
	### SPACE TRANSFORMATION ###
	############################

	# calculate rotation based on best-fit slope
	theta = np.arctan(m)
	sin_theta = np.sin(theta)
	cos_theta = np.cos(theta)

	# transform data points
	summaryt = np.zeros(summary.shape)
	Xt = X*cos_theta + Y*sin_theta
	Yt = X*(-sin_theta) + Y*cos_theta - b + (summaryt.shape[0]/2)

	# display swipe on transformed image
	for i in range(n_pts):
		if Xt[i] > 0 and Xt[i] < summaryt.shape[1]:
			if Yt[i] > 0 and Yt[i] < summaryt.shape[0]:
				summaryt[int(Yt[i]), int(Xt[i])] = W[i]


	##############################
	### ORDINARY BEST FIT POLY ###
	##############################
	# calculate polynomial best fit in transformed space
	[x0, x1, x2] = np.polynomial.polynomial.polyfit(Xt, Yt, deg = 2, w = W*W)
	x0a = x0 + SWIPE_RADIUS
	x0b = x0 - SWIPE_RADIUS

	# remove all points outside accepted range of polynomial
	mask = np.abs(x0 + x1*Xt + x2*Xt*Xt - Yt) <= SWIPE_RADIUS
	X = X[mask]; Xt = Xt[mask]
	Y = Y[mask]; Yt = Yt[mask]
	W = W[mask]
	n_pts = W.shape[0]


	###############################
	### SAVE INTENSITY FEATURES ###
	###############################

	# calculate exact and approximate swipe arc length
	# NOTE: since we're rotating and translating, this is the same as length in X
	L_exact = scipy.integrate.quad(lambda x: np.sqrt(4*x2*x2*x*x+4*x1*x2*x+x1*x1+1), 
							 Xt.min(), Xt.max())[0]
	L_approx = Xt.max() - Xt.min()
	deviation = L_exact / L_approx
	
	# store average pixel intensities according to swipe type
	pixel_avg = W.sum() / (nframes * L_exact)
	max_pixel = W.max()
	global swipe_idx 
	avgbins = [0]*N_BINS
	maxbins = [0]*N_BINS
	avg_dicts[material.value][pressure.value][studentname].append(pixel_avg)
	for i in range(n_pts):
		x_norm = (Xt[i]-Xt.min()-EPSILON) / L_approx
		pixel_bin = int(np.floor(x_norm * N_BINS))
		avgbins[pixel_bin] += W[i]
		maxbins[pixel_bin] = max(maxbins[pixel_bin], W[i])
	swipes.loc[swipe_idx] = [studentname, pressure_names[pressure.value], material_names[material.value], L_exact, theta, deviation, pixel_avg, max_pixel] + avgbins + maxbins
	swipe_idx += 1

	# store swipe profile
	for i in range(n_pts):
		x_norm = (Xt[i]-Xt.min()-EPSILON) / L_approx
		profile_dicts[material.value][pressure.value][studentname].append((x_norm, W[i]))


	#################################
	### CALCULATE TIMING FEATURES ###
	#################################

	# get first and last non-zero frames
	reader = imageio.get_reader(folder+outfile, 'ffmpeg')
	global timing_idx
	first_frame = -1; last_frame = -1
	for i, image in enumerate(reader):
		image = np.array(image).astype(np.uint8)[:,:,0]
		intensity = image.sum()
		if intensity > 0:
			if first_frame < 0:
				first_frame = i
			last_frame = i

	# if nearly all non-zero pixels have been non-zero, then the swipe has
	# just ended and the remaining time is the swipe fading
	nonzero_pixels = np.zeros(summary.shape, dtype=bool)
	for i, image in enumerate(reader):
		image = np.array(image).astype(np.uint8)[:-REMOVE_N_BOT,:,0] != 0
		nonzero_pixels = np.logical_or(nonzero_pixels, image)
	nonzero_pixels = nonzero_pixels.sum()

	found_pixels = np.zeros(summary.shape, dtype=bool)
	mid_frame = -1
	for i, image in enumerate(reader):
		image = np.array(image).astype(np.uint8)[:-REMOVE_N_BOT,:,0] != 0
		found_pixels = np.logical_or(found_pixels, image)
		if found_pixels.sum() > 0.95 * nonzero_pixels:
			mid_frame = i
			break

	# save all data to array for graphing
	for i in range(first_frame, last_frame):
		frame_idx = i - first_frame
		image = np.array(reader.get_data(i)).astype(int)
		timing_dicts[material.value][pressure.value][studentname].append((frame_idx, image.sum()))

	# set features
	timebins = [0]*N_BINS
	for i in range(first_frame, last_frame):
		image = np.array(reader.get_data(i)).astype(int)
		frame_idx = i - first_frame
		frame_bin = int(np.floor(((frame_idx-EPSILON)/MAX_FRAMES) * N_BINS))
		timebins[frame_bin] += image.sum()

	total_time = last_frame - first_frame
	swipe_time = mid_frame - first_frame
	fade_time = last_frame - mid_frame
	timing.loc[timing_idx] = [total_time, swipe_time, fade_time]+timebins

	timing_idx += 1


	###########################
	### PLOTTING AND OUTPUT ###
	###########################
	
	# show polynomial in transformed space
	for x in range(0, summaryt.shape[1]):
		ya = int(x0a + x1*x + x2*x**2)
		yb = int(x0b + x1*x + x2*x**2)
		if ya >= 0 and ya < summary.shape[0]:
			summaryt[ya][x] = color2
		if yb >= 0 and yb < summary.shape[0]:
			summaryt[yb][x] = color2

	# print('ols')
	# print(x0, x1, x2)

	# graph polynomial in original coordinate system
	# don't worry if polynomial doesn't show up, it just means we
	# happened to not choose a range which overlaps the image
	for xt in range(int(Xt.min())-300, int(Xt.max())+300):

		# # plot center line
		# yt = int(x0 + x1*xt + x2*xt*xt) + b - (summaryt.shape[0]/2)
		# x = int(xt*cos_theta + yt*(-sin_theta))
		# y = int(xt*sin_theta + yt*cos_theta)
		# if x >= 0 and x < summary.shape[1]:
		# 	if y >= 0 and y < summary.shape[0]:
		# 		summary[y][x] = color1

		# plot boundary line 1
		yta = int(x0a + x1*xt + x2*xt*xt) + b - (summaryt.shape[0]/2)
		xa = int(xt*cos_theta + yta*(-sin_theta))
		ya = int(xt*sin_theta + yta*cos_theta)
		if xa >= 0 and xa < summary.shape[1]:
			if ya >= 0 and ya < summary.shape[0]:
				summary[ya][xa] = color1

		# plot boundary line 2
		ytb = int(x0b + x1*xt + x2*xt*xt) + b - (summaryt.shape[0]/2)
		xb = int(xt*cos_theta + ytb*(-sin_theta))
		yb = int(xt*sin_theta + ytb*cos_theta)
		if xb >= 0 and xb < summary.shape[1]:
			if yb >= 0 and yb < summary.shape[0]:
				summary[yb][xb] = color1

	result = np.zeros(summary.shape)
	for i in range(n_pts):
		result[int(Y[i])][int(X[i])] = W[i]

	

	# save summed pixel intensities over video duration
	# cv2.imwrite(folder+os.path.splitext(filename)[0]+"t.jpg", summaryt)
	# cv2.imwrite(folder+os.path.splitext(filename)[0]+"poly.jpg", summary)
	# cv2.imwrite(folder+os.path.splitext(filename)[0]+"r.jpg", result)



# check input arguments, set filepaths
if len(sys.argv) != 2:
	print("Usage: python isolate_swipe.py <folder_name>")
	sys.exit()
folder = sys.argv[1]
if folder[-1] != "/":
	folder += "/"
if not os.path.isdir(folder):
	print("ERROR: Folder", folder, "does not exist.")
	sys.exit()

# process each video file in given directory
for n, name in enumerate(names):
	sfolder = folder + realnames[n] + "/segments/"
	print(sfolder)
	files = os.listdir(sfolder)
	files.sort()
	nfiles = len(files)
	for i, filename in enumerate(files):
		if filename.endswith(".mov"):
			match = re.match(r"(soft|hard)_(cloth|concrete|door|drywall|laminant|whiteboard)([0-9]+[ab]*\.mov)", filename, re.I)
			if match:
				outfile = match.groups()[0] + match.groups()[1] + "swipe" + match.groups()[2]
				process_video(sfolder, "user"+str(n), filename, outfile)
