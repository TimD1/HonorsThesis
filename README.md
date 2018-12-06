# Thermal Finger Swipe Pressure Detection

### Procedure
0. __ffmpeg__ was used to crop the videos in time to avoid dynamic rescaling occurring at the start of videos. The syntax for performing this with minimal information loss due to compression is as follows:
ffmpeg -i input.mov -ss start/fps -t (end-start)/fps -vf "crop=256:256:xstart:ystart" -c:v libx264 -crf 17 output.mov
1. __segment.py__ is called manually for each unsegmented video with the correct x and y parameters to capture swipe paths.
2. __main.py__ Runs entire pipeline, using globals _folder_, _users_, _pressures_, and _materials_. A second input argument ("segments", "frames") can specify where on the pipeline to start.


### Scripts
1. __main.py__ Runs entire pipeline, using globals _folder_, _users_, _pressures_, and _materials_. A second input argument ("segments", "frames") can specify where on the pipeline to start.
2. __segment.py__ segments a single long video capture into consecutive swipes


### Directory Structure
More on this later

### Running project in Jupyter Notebook on GPU
The current directory should contain both all Github code and a folder with the swipe data. First ensure that the code is up-to-date with the Github repo:
```
> git pull
```

Then rebuild the docker file "thesis" with the updated current directory:
```
> docker build -t thesis .
```

Run the Docker file with GPU backend, leaving port 8888 open for our Jupyter Notebook server and open up this instance in BASH:
```
> sudo docker run --runtime=nvidia -it -p 8888:8888 thesis:latest bash
```

Run the Jupyter Notebook server in the Docker container:
```
$ jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

Access Jupyter Notebook from your host machine (outside the Docker container) by navigating to `http://localhost:8888/tree`. The required session token should be listed in the terminal where you started the Jupyter Notebook server inside the Docker container.

#### To get a second Docker shell
List all currently running Docker containers
```
> docker container ls
```

Using the hash value listed next to thesis:latest, open up a shell in BASH:
```
> docker exec -t <hash> bash
```
You should see a new shell spawn.
