from discord.ext import commands
import discord

import json
import os
import sys

import chat
import encoder
import model1
import numpy as np
import sample
import tensorflow as tf

#Hyperparameters
model_name = 'E:\\Programs\\PyCharmCommunityEdition2019.1.3\\PycharmProjects\\NexthinkChatbotPrototype\\models\\774M'
seed=None
length=20
temperature = .85
top_k=0

bot_name = 'Bot'

#Open session and maintain it
enc = encoder.get_encoder(model_name)
hparams = model1.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

sess = tf.Session()
np.random.seed(seed)
tf.set_random_seed(seed)
context = tf.placeholder(tf.int32, [1, None])
output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=1,
        temperature=temperature, top_k=top_k
        )

saver = tf.train.Saver()
ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
saver.restore(sess, ckpt)
print("\nUsing checkpoint from:\n" + ckpt)


prefix = "?"
token = 'NjE4ODU3MzAxMTM4MjEwODQ2.XXF4uA.boORRlDN7JvN76Cpvqp62MWNipY'
bot = commands.Bot(command_prefix=prefix)

# Main loop starts here
# Confirm bot is online
@bot.event
async def on_ready():
    print("\nEverything's up and running chief.")
    await bot.change_presence(activity=discord.Game(name='!!! for new conv'))

#@bot.commands
#

@bot.event
async def on_message(message):
    conversation = """{0}: hi. what's your name?
{1}: {1}. and you?
{0}: I'm {0}
{1}: so what can I do for you?
""".format(message.author.display_name, bot_name)

    for i in bot.cached_messages._SequenceProxy__proxied:
        if str(i.author) == "BotMcBotty#3002":
            conversation = conversation + (i.content + "\n")
        else:
            conversation = conversation + ("{}: ".format(message.author.display_name) + i.content + "\n{}: ".format(bot_name))

    if str(message.author) == "BotMcBotty#3002":
        print("\nBot message duplicate")
    else:
        if message.content == "!!!":
            conversation = None
            await message.channel.send("New conversation from here on:")
            await message.channel.send('Clearing messages...')
            #async for msg in client.logs_from(message.channel):
            #   if not msg.pinned:
            #        await bot.delete_message(msg)
            #messages = []
            #for i in bot.cached_messages._SequenceProxy__proxied:
            #   i.delete()
            return
        if message.content == "???":
            await message.channel.send("Restarting bot")
            python = sys.executable
            os.execl(python, python, *sys.argv)

        reply, conversations = chat.get_reply(enc, sess, output, context, message.content, message.author.display_name, bot_name, conversation)

        conversation = conversation + reply
        print("\n\n\nCURRENT CONVERSATION with {}:\n".format(message.author.display_name) + conversation)
        await message.channel.send(reply, tts=True)

bot.run(token)

