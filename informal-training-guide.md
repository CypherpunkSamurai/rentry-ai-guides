!!! note Do you have a 3090? We want you!
    We are going to do a **distributed training** run on November 15th using hivemind! Join us at the discord: https://discord.gg/8Sh2T6gjd2

!!! note New discord server
    The old discord server is now unrelated to this project. For the new discord server join: https://discord.gg/8Sh2T6gjd2


### Use the one in haru's repo for the meantime: https://github.com/harubaru/waifu-diffusion/

If you need help, ask us on discord: https://discord.gg/8Sh2T6gjd2
### Minimun requirements:

8 Bit Adam:
- 17.4GB of VRAM
- 12.5GB of RAM

8 Bit Adam and FP16:
- 19GB of VRAM
- 8GB of RAM

### Instructions:
Execute the following:
```
git clone https://github.com/harubaru/waifu-diffusion/blob/main/diffusers_trainer.py
cd waifu-diffusion
pip install -r requirements.txt
```
Download your dataset and put it in a folder. It must be in the following format:

mydataset
├── image0.jpg
├── image0.txt
├── image1.jpg
└── image1.txt

The model to train on top of has to be in diffusers format. Search "CKPT to diffusers script" on google and save it under convert_original_stable_diffusion_to_diffusers.py, then execute:
```
python3 convert_original_stable_diffusion_to_diffusers.py --checkpoint_path mycheckpoint.ckpt --dump_path diffuser
```
Change "mycheckpoint.ckpt" to the name of the model CKPT file to convert, and "diffuser" to the name of the folder where you want the new converted diffuser model to be at.

In order to start training execute the following command:
```
python3 diffusers_trainer.py --model diffuser --dataset mydataset --run_name mytrainedmodel --use_8bit_adam True
```
This command will:
- Use the folder "diffuser" as the base diffuser model to train on top of
- Use the folder "mydataset" as the training data
- Name the new model "mytrainedmodel"
- Use 8-bit Adam Optimizer

Of course you can change any of those parameters to your liking.
Once training finishes, your trained model will be available in the "output" folder.

You can also do distributed training, but such feature is under alpha testing and does not work correctly yet. Check the repository for more information: https://github.com/chavinlo/distributed-diffusion

### How long will this take?
It depends on what GPU you are using. But what is indeed a fact is that it is going to be much faster than the previous trainer.
Some experiments ran on A100-80GB-XM4 
ETA is for 75K Images/10 Epochs/750K Itterations. In other words, analyzing 750,000 Images.
Loaded via directories

8 Bit Adam only, Batch size 1, lr 5e-6:
- It/s: Min. 2.60; Max. 3.20; Avg. 3.05
- ETA: Around 70 Hours
- 17.4GB of VRAM
- 12.5GB of RAM

8 Bit Adam only, Batch size 4, lr 5e-6:
- It/s: Min. 1.02; Max. 1.26; Avg. 1.15
- ETA: Around 47 Hours
- 37GB of VRAM
- 11.2GB of RAM

8 Bit Adam only, Batch size 8, lr 5e-6:
- s/it: Min. 1.39; Max. 2.16; Avg. 1.40
- ETA: Around 40 Hours
- 63GB of VRAM
- 11.2GB of RAM

If you need help, ask us on discord: https://discord.gg/8Sh2T6gjd2

18721
##### This guide has been reworked. Archived version: https://rentry.org/dq6vm

