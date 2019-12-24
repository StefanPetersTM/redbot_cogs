from .encoder import *
from .model import *
from .sample import *


# Define function that gets called on every message to get a reply
def session():
    # Hyperparameters
    model_name = r'D:\GithubProjects\TM\774M'
    seed = None
    length = 50
    temperature = 0.85
    top_k = 0

    # global bot_name
    bot_name = "Bob"

    # Open session and maintain it
    enc = get_encoder(model_name)
    hparams = default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    sess1 = tf.Session()
    np.random.seed(seed)
    tf.set_random_seed(seed)
    context = tf.placeholder(tf.int32, [1, None])
    output = sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k
    )
    saver1 = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
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
    encoded_conversation = enc.encode(conv)
    result = sess1.run(output, feed_dict={
        context: [encoded_conversation]
    })[:, len(encoded_conversation):]
    text = enc.decode(result[0])
    splits = text.split('\n')
    print(splits)
    reply = splits[0]

    if len(reply) < 2:
        for s in splits:
            if s.startswith(bot_name):
                reply = s[len(bot_name):
                        ]
                break

    reply = reply.lstrip(":.,;?")
    reply = reply.strip()

    if reply is None or len(reply) == 0:
        reply = "I'm affraid I can't give you an answer to that..."

    print(reply + "\n")
    conversation = conv + "\n{}: ".format(bot_name) + (reply)
    return reply, conversation
