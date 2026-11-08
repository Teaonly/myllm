all: emb2redis

hiredis/libhiredis.a:
	@echo "Please, enter the hiredis directory and build it."
	@exit 1

emb2redis: emb2redis.c hiredis/libhiredis.a
	$(CC) emb2redis.c ../gguflib.c ../sds.c ../fp16.c hiredis/libhiredis.a -o emb2redis -O2 -Wall -W

clean:
	rm -f emb2redis
