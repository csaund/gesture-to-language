print "loading tf"
import tensorflow as tf
print "loading tfh"
import tensorflow_hub as hub

import numpy as np

tf.compat.v1.disable_eager_execution()


# embed = hub.Module("/tmp/sentence-encoder")

# using google's transformer encoder.

## think I'm using this? https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15

# Compute a representation for each message, showing various lengths supported.
elephant = "Elephant"
dog = "Dog"
wolf = "Wolf"
vacuum = "Vacuum"

word = "Elephant"

sentence = "I am a sentence for which I would like to get its embedding."
sentence1 = "Jazz is a great genre of music."
sentence2 = "I am a big fan of rock music."
sentence3 = "It is nice listening to the sounds of nature."




def embed_useT(module):
    with tf.compat.v1.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

embed_fn = embed_useT("/tmp/sentence-encoder")

paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
# messages = [word, sentence, paragraph]


messages = [sentence, sentence1, sentence2, sentence3]

# Reduce logging output.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# with tf.compat.v1.Session() as session:
#     session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
message_embeddings = embed_fn(messages)
for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        print("Message: {}".format(messages[i]))
        print("Embedding size: {}".format(len(message_embedding)))
        message_embedding_snippet = ", ".join((str(x) for x in        message_embedding[:3]))
        print("Embedding[{},...]\n".
                   format(message_embedding_snippet))

# print message_embeddings[0]


def get_dist(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

print "%s / %s dist: %s" % (sentence1, sentence2, np.inner(message_embedding[1], message_embedding[2]).max())
print "%s / %s dist: %s" % (sentence1, sentence, np.inner(message_embedding[1], message_embedding[0]).max())
print "%s / %s dist: %s" % (sentence2, sentence, np.inner(message_embedding[2], message_embedding[0]).max())
print "%s / %s dist: %s" % (sentence2, sentence3, np.inner(message_embedding[2], message_embedding[3]).max())
print "%s / %s dist: %s" % (sentence, sentence3, np.inner(message_embedding[0], message_embedding[3]).max())



animals = ["tigers", "lions", "bears"]
nature = ["trees", "ocean", "rivers"]
emotions = ['happy', 'sad', 'irritated']

a_mat = embed_fn(animals)
n_mat = embed_fn(nature)
e_mat = embed_fn(emotions)

a_test = "birds"
n_test = "cliff"
e_test = "joy"

a_test_mat = embed_fn([a_test])
n_test_mat = embed_fn([n_test])
e_test_mat = embed_fn([e_test])



print "Animal test"
print "Animal similarity: %s " % (np.inner(a_test_mat, a_mat)).max()
print "Nature similarity: %s " % (np.inner(a_test_mat, n_mat)).max()
print "Emotion similarity: %s " % (np.inner(a_test_mat, e_mat)).max()

print "Nature test"
print "Animal similarity: %s " % (np.inner(n_test_mat, a_mat)).max()
print "Nature similarity: %s " % (np.inner(n_test_mat, n_mat)).max()
print "Emotion similarity: %s " % (np.inner(n_test_mat, e_mat)).max()

print "Emotion test"
print "Animal similarity: %s " % (np.inner(e_test_mat, a_mat)).max()
print "Nature similarity: %s " % (np.inner(e_test_mat, n_mat)).max()
print "Emotion similarity: %s " % (np.inner(e_test_mat, e_mat)).max()

# print "loading torch"
# import torch
# print "loading transformers"
# from transformers import *
#
# from pytorch_pretrained_bert.tokenization import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
# marked_text = "[CLS] " + text + " [SEP]"
#
# print (marked_text)
#
# tokenized_text = tokenizer.tokenize(marked_text)
#
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#
# segments_ids = [1] * len(tokenized_text)
#
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])
#
# # Load pre-trained model (weights)
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Put the model in "evaluation" mode, meaning feed-forward operation.
# model.eval()
#
# with torch.no_grad():
#     encoded_layers, _ = model(tokens_tensor, segments_tensors)
#
#
#
#
# concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
#
# summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]
#
# sentence_embedding = torch.mean(encoded_layers[11], 1)
#
# print ("Our final sentence embedding vector of shape:"), sentence_embedding[0].shape[0]
#
#
#
#
#
#
# print "Done"
