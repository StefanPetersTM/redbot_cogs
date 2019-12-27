# redbot_cogs
This repo contains 2 extensions(add-ons) for [*redbot*](https://github.com/Cog-Creators/Red-DiscordBot), a discord bot.

## Intro
This repo has been created as a highschool project, and as such any feedback is welcome.

- The first extension is called *chatbot*. This chatbot can simulate human-like conversation by using the [GPT-2](https://github.com/openai/gpt-2) model.

- The second extension is called *objrec*. This extension can process images sent to the bot using [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) . The bot then sends a reply containing any objects found in the image.

## Usage
First start by creating an environment using python 3.7, then installing the requirements.txt using pip:

```
pip install -r /path/to/requirements.txt
``` 

Then follow the instructions on how to get up and running with redbot.

Then copy this repo where the bot's cogs files are located.

The last step requires you to download the model weights and place them in their respective folders:

- objrec, first [file](https://1drv.ms/u/s!Annk_cU7Ejkpg06Hq-S6DXEoMawQ?e=Z17Dbr); second [file](https://1drv.ms/u/s!Annk_cU7Ejkpg0_cvAXOVBt5wQ-v?e=VeWulX)
    
    
    Place these two files in:
    ```
    redbot_cogs\objrec\data\data\darknet_weights
    ```

- chatbot, [file](https://1drv.ms/u/s!Annk_cU7Ejkpg1BBRdQZTixrhAWP?e=adUeXi)


    Place this file in:
    ```
    redbot_cogs\chatbot\data\774M
    ```
    
    Don't forget to load the cogs using:
    ```
    [p]load chatbot and [p] load objrec
    ```
    
### Alternate usage
You can also directly do the setup from a running redbot:
    
- [p]pipinstall ALL_DEPENDENCIES FROM requirements.txt
    
- [p]repo add redbot_cogs https://github.com/StefanPetersTM/redbot_cogs master    
    
- [p]cog install redbot_cogs objrec
    
- [p]cog install redbot_cogs chatbot

    
    
Then profit! You can now chat with redbot and send him images!

# Acknowledgements
The chatbot extension is build using as a base one of the Trusty-cogs called [cleverbot](https://github.com/TrustyJAID/Trusty-cogs/tree/master/cleverbot) 

The objrec extension is based on wizyoung's [implementation](https://github.com/wizyoung/YOLOv3_TensorFlow) of YOLOv3 in tensorflow