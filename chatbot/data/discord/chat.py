# Use the commented code below to run a session
# Use the function below to get a reply from gpt2

# Define function that gets called everytime to get a reply


from .encoder import *
from .model import *
from .sample import *

def session():
    # Hyperparameters
    model_name = r'D:\GithubProjects\TM\774M'
    seed = None
    length = 50
    temperature = 0.85
    top_k = 0

    #global bot_name
    bot_name = "Bob"

    # Open session and maintain it
    enc = get_encoder(model_name)
    hparams = default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    print('1')
    sess1 = tf.Session()
    print('2')
    np.random.seed(seed)
    print('3')
    tf.set_random_seed(seed)
    print('4')
    context = tf.placeholder(tf.int32, [1, None])
    print('5')
    output = sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=1,
        temperature=temperature, top_k=top_k
    )
    print('6')
    saver1 = tf.train.Saver()
    print('7')
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    print('8')
    saver1.restore(sess1, ckpt)

    print("\nUsing checkpoint from:\n" + ckpt)
    return enc, sess1, output, context, bot_name

def get_reply(
        enc,
        sess1,
        output,
        context,

        conv,
        user_name,
        bot_name,
):
    print("1")
    encoded_conversation = enc.encode(conv)
    print('2')
    result = sess1.run(output, feed_dict={
        context: [encoded_conversation]
    })[:, len(encoded_conversation):]
    print('3')
    text = enc.decode(result[0])
    #print(text)
    print('4')
    splits = text.split('\n')
    print(splits)
    reply = splits[0]
    done = False

    if len(reply) < 2:
        for s in splits:
            if s.startswith(bot_name):
                reply = s[len(bot_name):
                        ]
                #print(reply)
                break

    reply = reply.lstrip(":.,;?")
    #print(reply)
    reply = reply.strip()

    if reply is None:
        reply = "I'm affraid I can't give you an answer to that..."

    print(reply+"\n")
        #reply = str(reply)
    conversation = conv + "\n{}: ".format(bot_name) + (reply)
    #print("Current conversation of {0} with {1}:".format(bot_name, user_name) + conversation)
    #print("Bot reply: " + str(reply))
    #print("\nfinal conv" + str(conversation))

    return reply, conversation