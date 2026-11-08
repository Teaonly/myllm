This is a simple program that gets the token embeddings from an LLM in
GGUF format, and adds it into a [Redis vector set](https://github.com/redis/redis/tree/unstable/modules/vector-sets). After the embeddings are added into
Redis you can easily check what are the embeddings more similar to others, an
operation that allows to build some mental model about the tokens embedding
space the LLM learned during training (spoiler: it is quite different than
than word2vec or alike: often certain words are near to unexpected words:
that's likely due to the fact we can't fully appreciate how
the models use all the components of the embedding in the Transformer blocks
inference).

To compile the program, stay in this directory and perform the following:

1. git clone https://github.com/redis/hiredis
2. cd hiredis; make
3. cd ..
4. make

Then do something like:

    ./emb2redis my_llm.gguf llm_embeddings_key -h 127.0.0.1 -p 6379

At the end, try something like this:

    redis-cli VSIM llm_embeddings_keys ELE "banana"

