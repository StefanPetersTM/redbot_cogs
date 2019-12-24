import aiohttp
import discord
import tensorflow as tf

from redbot.core import commands, checks, Config
from .data.discord import chat as cht


class chat(commands.Cog):
    """Chat with me"""

    listener = getattr(commands.Cog, "listener", None)

    if listener is None:  # thanks Sinbad
        def listener(name=None):
            return lambda x: x

    def __init__(self, bot):
        g1 = tf.Graph()
        with g1.as_default():
            self.enc, self.sess1, self.output, self.context, self.bot_name = cht.session()

        self.bot = bot
        self.config = Config.get_conf(self, 127486454786)
        default_global = {"api": None, "io_user": None, "io_key": None, "allow_dm": False}
        default_guild = {"channel": None, "toggle": False}
        self.config.register_global(**default_global)
        self.config.register_guild(**default_guild)
        self.session = aiohttp.ClientSession(loop=self.bot.loop)
        self.instances = {}
        self.conv = {}

    @commands.command()
    async def chat(self, ctx, *, message):
        """Talk with me by DM with me or with '[p]chat <message>'"""

        print("\n\n\n\nMessage:{}".format(str(message)))
        author = str(ctx.message.author).split("#")[0]
        channel = ctx.message.channel

        if ctx.author.id != self.bot.user.id:
            async with channel.typing():
                if author in self.conv:
                    current_conv = self.conv[author]
                    current_conv += "\n{0}: {1}".format(author, message)
                    reply, conversation = cht.get_reply(self.enc, self.sess1, self.output, self.context, current_conv,
                                                        author, self.bot_name)
                    print("Conv with {}:\n".format(author) + conversation)
                    self.conv[author] = conversation

                else:
                    print("no existing conversation\n\n")
                    self.conv[author] = """{0}: hi. what's your name?
{1}: {1}. and you?
{0}: I'm {0}
{1}: so what can I do for you?
{0}: {2}""".format(author, self.bot_name, message)
                    reply, conversation = cht.get_reply(self.enc, self.sess1, self.output, self.context,
                                                        self.conv[author], author, self.bot_name)
                    self.conv[author] = conversation
                    print("conv with {0}: {1}".format(author, conversation))
                await ctx.send(str(reply))

    @commands.command()
    async def clear(self, ctx):
        """Clear the current conversation with the bot"""

        await ctx.send("Deleting conversation...", delete_after=2)
        try:
            del self.conv[str(ctx.message.author).split("#")[0]]
        except:
            await  ctx.send("Conversation has allready been cleared.")

    @commands.group()
    async def chatbotset(self, ctx):
        """
            Settings for the chatbot
        """
        pass

    @chatbotset.command()
    @commands.guild_only()
    @checks.mod_or_permissions(manage_channels=True)
    async def toggle(self, ctx):
        """Toggles reply on mention"""
        guild = ctx.message.guild
        if not await self.config.guild(guild).toggle():
            await self.config.guild(guild).toggle.set(True)
            await ctx.send("I will reply on mention.")
        else:
            await self.config.guild(guild).toggle.set(False)
            await ctx.send("I won't reply on mention anymore.")

    @chatbotset.command()
    @checks.is_owner()
    async def dm(self, ctx):
        """Toggles reply in DM"""
        if not await self.config.allow_dm():
            await self.config.allow_dm.set(True)
            await ctx.send("I will reply directly to DM's.")
        else:
            await self.config.allow_dm.set(False)
            await ctx.send("I won't reply directly to DM's.")

    @chatbotset.command()
    @checks.mod_or_permissions(manage_channels=True)
    @commands.guild_only()
    async def channel(self, ctx, channel: discord.TextChannel = None):
        """
            Toggles channel for automatic replies

            do `[p]cleverbot channel` after a channel is set to disable.
        """
        guild = ctx.message.guild
        cur_auto_channel = await self.config.guild(guild).channel()
        if not cur_auto_channel:
            if channel is None:
                channel = ctx.message.channel
            await self.config.guild(guild).channel.set(channel.id)
            await ctx.send("I will reply in {}".format(channel.mention))
        else:
            await self.config.guild(guild).channel.set(None)
            await ctx.send("Automatic replies turned off.")

    async def rec_img(self, ctx, img_labels):
        """
        This should not be shown in the help section
        """
        author = str(ctx.author).split("#")[0]
        message = "Take a look at this picture of {}".format(', '.join(img_labels))
        if author in self.conv:
            print("working")
            current_conv = self.conv[author]
            current_conv += "\n{0}: {1}".format(author, message)
            reply, conversation = cht.get_reply(self.enc, self.sess1, self.output, self.context,
                                                current_conv, author, self.bot_name)
            print("Conv with {}:\n".format(author) + conversation)
            self.conv[author] = conversation

        else:
            pass

    @listener()
    async def on_message(self, message):
        if message.author.id != self.bot.user.id:
            ctx = await self.bot.get_context(message)
            print("message prefix:{}".format(ctx.prefix))
            print("author: {0}\nmessage content:{1}".format(str(message.author), message.content))
            if not ctx.prefix:
                author = str(message.author).split("#")[0]
                guild = message.guild
                if guild is None:
                    if await self.config.allow_dm():
                        channel = ctx.message.channel
                        message = message.clean_content
                        async with channel.typing():
                            if author in self.conv:
                                current_conv = self.conv[author]
                                current_conv += "\n{0}: {1}".format(author, message)
                                reply, conversation = cht.get_reply(self.enc, self.sess1, self.output, self.context,
                                                                    current_conv, author, self.bot_name)
                                print("Conv with {}:\n".format(author) + conversation)
                                self.conv[author] = conversation

                            else:
                                print("no existing conv")
                                self.conv[author] = """{0}: hi. what's your name?
{1}: {1}. and you?
{0}: I'm {0}
{1}: so what can I do for you?
{0}: {2}""".format(author, self.bot_name, message)
                                reply, conversation = cht.get_reply(self.enc, self.sess1, self.output, self.context,
                                                                    self.conv[author], author, self.bot_name)
                                self.conv[author] = conversation
                                print("conv with {0}: {1}".format(author, conversation))
                            await ctx.send(str(reply))
                    return

            else:
                return

    def cog_unload(self):
        self.bot.loop.create_task(self.session.close())

    __unload = cog_unload
