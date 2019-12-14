from discord.ext import commands
import discord

import os
import sys
import random
import json

import chat
import encoder
import model1
import numpy as np
import sample
import tensorflow as tf


def session():
    # Hyperparameters
    model_name = 'E:\\Programs\\PyCharmCommunityEdition2019.1.3\\PycharmProjects\\NexthinkChatbotPrototype\\models\\774M'
    seed = None
    length = 20
    temperature = .85
    top_k = 0

    global bot_name
    bot_name = "Bot"

    # Open session and maintain it
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
    return enc, sess, output, context, bot_name


description = '''An example bot to showcase the discord.ext.commands extension
module.
There are a number of utility commands being showcased here.'''
bot = commands.Bot(command_prefix=commands.when_mentioned_or('!'), description= description)


@bot.event
async def on_ready():
    print('\nLogged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------\n\n')
    print("\nEverything's up and running chief.")
    await bot.change_presence(activity=discord.Game(name='! for help'))

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

@bot.command(category="Chatbot",description='Restart conversation')
async def clear(self):
    """Restart conversation"""
    return
    # await ctx.channel.send('Clearing messages...')
    # await ctx.channel.send("New conversation from here on:")
    # async for msg in client.logs_from(message.channel):
    #   if not msg.pinned:
    #        await bot.delete_message(msg)
    # messages = []
    #for i in bot.cached_messages._SequenceProxy__proxied:
     #   i.delete()
    #return

@bot.command(description="See your ping to the bot.")
async def ping(ctx):
    """Check ping to the bot"""
    await ctx.send(f'Pong! {round(bot.latency*1000)}ms')

# @bot.event
# async def on_command_error(ctx, error):
#     if isinstance(error, commands.MissingRequiredArgument):
#         await ctx.send("Missing arguments")
#     else:
#         print("Command error: " + error)
#         await ctx.send("Command error: " + error)

@bot.command(aliases=[''], description="Chat with the bot!")
async def chat(message):
    """Have a chat with the bot"""
    if "!" not in message.content:
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

            if message.content == "???":
                await message.channel.send("Restarting bot")
                python = sys.executable
                os.execl(python, python, *sys.argv)

            reply, conversations = chat.get_reply(enc, sess, output, context, message.content, message.author.display_name, bot_name, conversation)

            conversation = conversation + reply
            print("\n\n\nCURRENT CONVERSATION with {}:\n".format(message.author.display_name) + conversation)
            await message.channel.send(reply, tts=True)
# else:
#     if message.content == "!ping":
#         ping(message.)
#     if message.content == "!clear":
#         clear(ctx)
    else:
        pass

enc, sess, output, context, bot_name = session()
bot.run('NjE4ODU3MzAxMTM4MjEwODQ2.XXF4uA.boORRlDN7JvN76Cpvqp62MWNipY')
