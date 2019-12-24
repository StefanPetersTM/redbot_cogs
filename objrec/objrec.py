import os
import datetime
from urllib.request import Request, urlopen

import tensorflow as tf
import cv2
import discord
from redbot.core import commands

from .data import ObjectRecognition
from .data import VideoRecognition


class objrec(commands.Cog):
    """Process images"""

    def __init__(self, bot):
        print("   Loading objrec...")
        self.bot = bot
        g2 = tf.Graph()
        with g2.as_default():
            print("      Creating Tensorflow session...")
            self.sess2, self.boxes, self.scores, self.labels, self.input_data, self.classes, self.color_table = ObjectRecognition.session()
            print("      Tensorflow session created.\n")
            self.img_extensions = [".JPG", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".bmp", ".pbm", ".pgm", ".ppm",
                                   ".tiff", ".tif"]
            self.vid_extensions = [".mp4", ".avi", ".gif"]

    @commands.command()
    async def objrec(self, ctx):
        """Object recognition using YOLOv3."""
        message = ctx.message
        print("\n" * 2)

        if message.attachments:
            print("Image sent to bot by {}: ".format(message.author) + str(message.attachments[0].filename))
            filename, extension = os.path.splitext(message.attachments[0].filename)
            if extension in self.img_extensions:
                req = Request(message.attachments[0].url, headers={'User-Agent': 'Mozilla/5.0'})
                img = urlopen(req).read()

                currentDT = datetime.datetime.now()
                curr_img_path = 'D:\Discord_bot\Discord_bot_saved_images\{}'.format(
                        str(message.author) + currentDT.strftime("_%Y-%m-%d_%H-%M.jpg"))
                fhand = open(curr_img_path, 'wb')
                fhand.write(img)
                fhand.close()

                processed_img, labels_, scores_ = ObjectRecognition.obj_rec(curr_img_path, self.sess2, self.boxes,
                                                                            self.scores,
                                                                            self.labels, self.input_data, self.classes,
                                                                            self.color_table)

                unique_labels = []
                for i in labels_:
                    if i not in unique_labels:
                        unique_labels.append(i)

                print(unique_labels)
                print(', '.join(unique_labels))
                await message.channel.send(content=None, file=discord.File(processed_img, 'processed_image.jpg'))
                await message.channel.send(
                        content="I can see that you sent me a picture containing " + ', '.join(unique_labels),
                        delete_after=10)

                received_image = self.bot.get_cog('chat')
                print("Chatbot cog is: {}".format(received_image))
                if received_image is not None:
                    await received_image.rec_img(ctx, unique_labels)

            elif extension in self.vid_extensions:
                req = Request(message.attachments[0].url, headers={'User-Agent': 'Mozilla/5.0'})
                img = urlopen(req).read()

                currentDT = datetime.datetime.now()
                curr_img_path = 'D:\Discord_bot\Discord_bot_saved_images\{}'.format(
                        str(message.author) + currentDT.strftime("_%Y-%m-%d_%H-%M-%S.mp4"))
                fhand = open(curr_img_path, 'wb')
                fhand.write(img)
                fhand.close()

                processed_vid, labels_, inference_time = VideoRecognition.vid_rec(curr_img_path, self.sess2, self.boxes,
                                                                                  self.scores,
                                                                                  self.labels, self.input_data,
                                                                                  self.classes, self.color_table)

                await message.channel.send(
                        content="Average inference time: " + str(inference_time) + '\n' + str(labels_),
                        file=discord.File(processed_vid, filename='processed_video.mp4'))

            else:
                await message.channel.send(content=str("Unsupported file format!"))

        else:
            await message.channel.send(
                    content="No image sent! Send an image and comment '{}objrec' to see some magic.".format(ctx.prefix))

    @commands.command()
    async def bug(self, message):
        """Take a bug report"""
        await message.channel.send("Please fill out this form to report a bug: https://forms.gle/FyeWKuGdzu5wfyUL6")

    @commands.command()
    async def improve(self, message):
        """Suggest us what we can do to improve the bot"""
        await message.channel.send("Suggest us what we can improve in this form: https://forms.gle/Q4ASeARZGmqyR2Qg7")

    def cog_unload(self):
        self.bot.loop.create_task(self.session.close())


__unload = cog_unload
