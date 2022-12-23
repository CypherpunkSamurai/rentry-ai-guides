#Stable Diffusion Training

#new guide https://rentry.org/informal-training-guide

##NOTE: Although this guide does work, it's recommended to use waifu-diffusion repo instead. It's more full and prints epoch process, YOU ALSO ONLY NEED 30 GB FOR TRAINING >>88477948. I'm writing the documentation but it will take a while. You can also take a look at live training progress on the channel #live-training at https://discord.gg/hKu6ZvneJy
## JOIN THE TOUHOUAI DISCORD FOR WAIFU-DIFF INFO https://discord.gg/7w3rBsSkrj


Status: Complete but needs more explanations

####1. Obtaining and organizing Danbooru2021 dataset

Download Rsync, on windows you need to download [Cygwin](https://www.cygwin.com/install.html), on Linux, it's most likely already installed. If not download it from your distro's repositories.

(Optional) Get Filelist:
``` bash
 rsync rsync://176.9.41.242:873/danbooru2021/
```
Filelist description:
233,738,664B filelist.txt: list of files across all directories
6,857,152B filelist.txt.xz: same as above but compressed
3,632,091,536B metadata.json.tar.xz: metadata folder compressed
7,996,891B nsfw-ids.txt: nsfw-ids (keep reading for why this exists)
FOLDER 512px: folder containing resized 512x512 images, only safe and general rating, no questionable, explicit or sensitive.
FOLDER metadata: folder containing all metadata files.
FOLDER original: original images, contains all ratings.

For now we are only going to work with 512px folder. I WILL update the script below to resize and parse Q, E, and S images soon.

Move to a folder/drive with a ton of free space (20gb for now):
``` bash
cd E:
```
Download the first batch of posts metadata, "posts000000000000.json" (about 420k entries, 820MB).
``` bash
rsync rsync://176.9.41.242:873/danbooru2021/metadata/posts000000000000.json ./metadata/
```
Download posts from id 0000 to id 0009 (you can download more, but if you are using a cloud instance I guess you are short on storage, use S3 maybe)
``` bash
rsync rsync://176.9.41.242:873/danbooru2021/512px/000* ./512px/
```
Create a directory named "labeled_data":
``` bash
mkdir labeled_data
```
Download the following python script to parse the metadata to text files (Requires Python 3.x):
https://github.com/chavinlo/stable-diffusion-scripts/blob/main/parser.py
Place it on the folder you have chosen (parent of labaled_data, 512px, and metadata)
Run it:
``` bash
py parser.py
```
Your labeled_data should look like the following:
![Image description](https://i.imgur.com/FR93hUL.png)
Dataset sample: https://github.com/chavinlo/stable-diffusion-scripts/tree/main/dataset-examples
####2. Verifying Danbooru2021 dataset
Prefferably search for an image you know the context of. For example, 882012.jpg:
![882012.jpg](https://cdn.donmai.us/original/74/c8/__saigyouji_yuyuko_touhou_drawn_by_satsuki_gogotaru__74c895d9ae7dde444b9fbb168bf9caba.jpg)
Open the text file with the same name (except .jpg of course):
![cat 882012.txt](https://i.imgur.com/uZbnUuz.png)
Does the tags match the image? if so good. If you want to be more confident go to its respective danbooru page: https://danbooru.donmai.us/posts/882012

####3. Zip everything
Windows:
![](https://i.imgur.com/2YX5Ygs.png)

####4. Preparing the Training Instance and Training
Requirements:
- At least 40GB of VRAM available
- At least 20GB of storage available
- At least 16GB of Memory available (Might work at 8GB)
!!! info
    If you need just a few more mb/gb you could install a head-less linux instalation.

My instance:
CPU: Intel Xeon Gold 6342 (12) @ 2.8GHz
GPU: NVIDIA A100-SXM4-80GB
Memory: 90.5 GiB

#### Dependencies
Clone stable-diffusion and taming-transformers repos from CompVis  (Do not use another one, this is just for dependencies)
``` bash
git clone https://github.com/CompVis/stable-diffusion
```
``` bash
git clone https://github.com/CompVis/taming-transformers
```
``` bash
pip install -e stable-diffusion/
```
!!! info
    If this throws a "doesn't exists" error, run it as: ```python -m pip install -e stable-diffusion/```
``` bash
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
```
``` bash
sudo apt install libopenmpi-dev
```
``` bash
pip install pytorch-lightning omegaconf einops mpi4py
```
``` bash
git clone https://github.com/Jack000/glid-3-xl-stable
```
``` bash
pip install -e glid-3-xl-stable/
```
On windows, go to glid-3-xl-stable folder.
``` bash
cd glid-3-xl-stable
```
Get the full rawpath of the glid-3-xl-stable folder. For me its '/notebooks/glid-3-xl-stable'
If on linux, you can run ```pwd``` to obtain yours
```
root@centro:/notebooks/glid-3-xl-stable# pwd
/notebooks/glid-3-xl-stable
```
On windows, go to scripts folder.
``` bash
cd scripts
```
On windows, edit the file named image_train_stable.py and follow the instructions below.
``` bash
nano image_train_stable.py
```
between ```import argparse``` and ```from guided_diffusion import dist_util, logger``` add:
``` python
import sys
sys.path.append('/notebooks/glid-3-xl-stable')
print(sys.path)
```
Remember to change '/notebooks/glid-3-xl-stable' for the full raw path of your glid-3-xl-stable folder.
Save the modifications.

Return to glid-3-xl-stable folder
```bash
cd ..
```

Copy the ldm folder from parent-folder>stable-diffusion>ldm to glid-3-xl-stable>ldm (create a new folder and copy the contents there)
```bash
cp -dr ../stable-diffusion/ldm/ ./ldm
```
Do the same with taming-transformers/taming
```bash
cp -dr ../taming-transformers/taming/ ./taming
```

#### Weights
This is where you put your weights. It's highly recommended to use the full-ema version of the weights rather than the release ones, as it contains more data useful for training.
List of sd-v1-4-full-ema.ckpt mirrors: https://github.com/chavinlo/stable-diffusion-scripts/blob/main/mirrors.txt
!!! info
    It's recommended to use the google drive mirrors as those are the fastest. In order to download them from a terminal, use gdown. ```pip install gdown``` then ```gdown link```, replace link for the link that you get AFTER pressing the download button, in other words the page that tells you that its too heavy to analyze for viruses.

Move your sd-v1-4-full-ema.ckpt file to the glid-3-xl-stable directory:
```bash
cp /storage/sd-v1-4-full-ema.ckpt sd-v1-4-full-ema.ckpt
```
replace /storage/sd-v1-4-full-ema.ckpt for the path of your file.

split the weights into split diffusion.pt and kl.pt
```bash
python split.py sd-v1-4-full-ema.ckpt
```

#### Dataset
move your dataset zip file to glid-3-xl-stable
```bash
cp /storage/labeled_data.zip labeled_data.zip
```
replace /storage/labeled_data.zip for the path of your file.

unzip it
```bash
unzip labeled_data.zip
```
There should be a new folder inside of glid-3-xl-stable named "labeled_data, and inside, the txt and img files we created before.

delete the zip file
```bash
rm labeled_data.zip
```

#### Execution
Edit train.sh with nano
```bash
nano train.sh
```
In the last line, change /path/to/image-and-text-files to the dataset folder. In my case it's /notebooks/glid-3-xl-stable/labeled_data/
```bash
python scripts/image_train_stable.py --data_dir /path/to/image-and-text-files $MODEL_FLAGS $TRAIN_FLAGS
```
To
```bash
python scripts/image_train_stable.py --data_dir /notebooks/glid-3-xl-stable/labeled_data/ $MODEL_FLAGS $TRAIN_FLAGS
```

make train.sh executable
```bash
chmod +x train.sh
```

execute train.sh
```bash
./train.sh
```
If you need help you can contact me at https://discord.gg/hKu6ZvneJy or create an issue at https://github.com/Jack000/glid-3-xl-stable (Only issues you know is related to the repo and not this guide!)

####5. Merging
Once the training is over, you can marge the checkpoint with the released weights so you can use it with other stable diffusion tools (including web-uis)
The check points are located at glid-3-xl-stable/logs
Keep in mind that it can very quickly fill up storage.

go to glid-3-xl-stable and run merge.py:
```bash
python merge.py sd-v1-4-full-ema.ckpt /notebooks/glid-3-xl-stable/logs/ema_0.9999_010000.pt
```
In this example the checkpoint I want to merge with is the 10k one. Only use files that start with "ema".
Change "/notebooks/glid-3-xl-stable/logs/ema_0.9999_010000.pt" for the path of your checkpoint

A new file named "model-merged.pt" will appear, you can rename this to "model.ckpt" and use it as normal with any other tools.


WIP Status: Should work now. The basics are covered. I will keep adding more info but this should be enough to start training for someone who knows a bit of python and linux.

Heres a video of the training progress over 4 hours: https://siasky.net/AAAeTOsvRmxMmFTDa4di7WllD30eeindRKHoJfeO6rWRyw

Credits:
	- Jack Qiao : https://github.com/Jack000 : Creator of glid-3-xl-stable
	- AstraliteHeart#7662 : add libopenmpi-dev dependency instruction

