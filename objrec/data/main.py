import os
import sys
import random
import tensorflow as tf
import datetime

from urllib.request import Request, urlopen
import discord
from discord.ext import commands

import chat
import ObjectRecognition

# Creating bot object that communicates with discord
bot = commands.Bot(command_prefix='!', description='''List of all commands''')


# These are just decorator for events that occur to the bot
# This first one is called when the bot has successfully authenticated to the discord servers
@bot.event
async def on_ready():
    print('\nLogged in as')
    print(bot.user.name)
    print('------')
    print("\nEverything's up and running chief.")
    await bot.change_presence(activity=discord.Game(name='Use at your own risk'))


@bot.command()
async def add(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left + right)


@bot.command()
async def roll(ctx, dice: str):
    """Rolls a dice in NdN format."""
    try:
        rolls, limit = map(int, dice.split('d'))
    except Exception:
        await ctx.send('Format has to be in NdN!')
        return

    result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
    await ctx.send(result)


@bot.command(description='For when you wanna settle the score some other way')
async def choose(ctx, *choices: str):
    """Chooses between multiple choices."""
    await ctx.send(random.choice(choices))


@bot.command()
async def repeat(ctx, times: int, content='repeating...'):
    """Repeats a message multiple times."""
    for i in range(times):
        await ctx.send(content)


@bot.command()
async def joined(ctx, member: discord.Member):
    """Says when a member joined."""
    await ctx.send('{0.name} joined in {0.joined_at}'.format(member))


@bot.group()
async def cool(ctx):
    """Says if a user is cool.
    In reality this just checks if a subcommand is being invoked.
    """
    if ctx.invoked_subcommand is None:
        await ctx.send('No, {0.subcommand_passed} is not cool'.format(ctx))


@cool.command(name='bot')
async def _bot(ctx):
    """Is the bot cool?"""
    await ctx.send('Yes, the bot is cool.')


@bot.command()
async def ping(ctx):
    """Simple ping test tool"""
    await ctx.send(f'Pong! {round(bot.latency * 1000)}ms')


@bot.command()
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Missing arguments")
    else:
        print("Command error: " + error)
        await ctx.send("Command error: " + error)


@bot.command()
async def bug(message):
    """Take a bug report"""
    await message.channel.send("Please fill out this form to report a bug: https://forms.gle/FyeWKuGdzu5wfyUL6")


@bot.command(description='Restart conversation')
async def clear(message):
    """Clear the current conversation"""
    await message.channel.send('Clearing messages...', delete_after=2)
    await message.channel.send("New conversation from here on:")
    # async for msg in client.logs_from(message.channel):
    #   if not msg.pinned:
    #        await bot.delete_message(msg)
    # messages = []


#    async for msg in bot.cached_messages._SequenceProxy__proxied(message.channel):
#        if not msg.pinned:
#            await bot.delete_message(msg)

async def on_message(message):


    # for i in bot.cached_messages._SequenceProxy__proxied:
    #    await i.delete()
    return


@bot.event
async def on_message(message):
    if not message.attachments and not message.content.startswith("!") and str(
            message.author) != "BotMcBotty#3002" and "```" not in message.content:
        conversation = """{0}: hi. what's your name?
{1}: {1}. and you?
{0}: I'm {0}
{1}: so what can I do for you?
""".format(message.author.display_name, bot_name)

        for i in bot.cached_messages._SequenceProxy__proxied:
            try:
                if not i.content.startswith("!") and "!help" not in str(i.content):
                    if str(i.author) == "BotMcBotty#3002" and ";;;" not in message.clean_content:
                        conversation = conversation + (i.content + "\n")
                    elif ";;;" not in message.clean_content:
                        conversation = conversation + (
                                "{}: ".format(message.author.display_name) + i.content + "\n{}: ".format(bot_name))
            except:
                pass

        if str(message.author) == "BotMcBotty#3002":
            print("\nBot message duplicate")
        else:
            if message.content == ";;;":
                conversation = None

            if message.content == "///":
                await message.channel.send("Restarting bot. Please wait for ~30s.")
                python = sys.executable
                os.execl(python, python, *sys.argv)

            reply, conversations = chat.get_reply(enc, sess1, output, context, message.content,
                                                  message.author.display_name, bot_name, conversation)

            conversation = conversation + reply
            print("\n\n\nCURRENT CONVERSATION with {}:\n".format(message.author.display_name) + conversation)
            await message.channel.send(reply, tts=True)

    elif message.attachments and str(message.author) != "BotMcBotty#3002":
        print("\n" * 2)
        print("Image sent to bot: " + str(message.attachments))
        req = Request(message.attachments[0].url, headers={'User-Agent': 'Mozilla/5.0'})
        img = urlopen(req).read()

        currentDT = datetime.datetime.now()
        curr_img_path = 'D:\Discord_bot\Discord_bot_saved_images\{}'.format(
            str(message.author) + currentDT.strftime("_%Y-%m-%d_%H-%M.jpg"))
        fhand = open(curr_img_path, 'wb')
        fhand.write(img)
        fhand.close()

        processed_img, labels_, scores_ = ObjectRecognition.obj_rec(curr_img_path, sess2, boxes, scores,
                                                           labels, input_data, classes, color_table)

        await message.channel.send(content=str(labels_)+"\n"+str(scores_), file=discord.File(processed_img, 'processed_image.jpg'))

    elif str(message.author) != "BotMcBotty#3002":
        await bot.process_commands(message)


g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    enc, sess1, output, context, bot_name = chat.session()
with g2.as_default():
    sess2, boxes, scores, labels, input_data, classes, color_table = ObjectRecognition.session()
bot.run('NjE4ODU3MzAxMTM4MjEwODQ2.Xbr9HQ.cUkf6_RP0AZHueBLcMkLy9xSxeM')
