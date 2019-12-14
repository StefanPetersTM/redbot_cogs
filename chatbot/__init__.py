from .chatbot import chat

def setup(bot):
    bot.add_cog(chat(bot))