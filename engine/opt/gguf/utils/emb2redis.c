#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

#include "../gguflib.h"
#include "hiredis/hiredis.h"

uint64_t tokens_processed = 0;

/* Free the token strings array. */
void free_token_strings(char **tokens, uint64_t num_tokens) {
    if (!tokens) return;
    for (uint64_t i = 0; i < num_tokens; i++) {
        free(tokens[i]);
    }
    free(tokens);
}

/* Load all token strings into an array, so that we can lookup
 * the token by its ID in constant time.
 * Returns an array of token strings that must be freed by the caller. */
char **load_token_strings(gguf_ctx *ctx, uint64_t *num_tokens_out) {
    gguf_key key;
    char **tokens = NULL;
    *num_tokens_out = 0;

    gguf_rewind(ctx);

    /* Look for tokenizer.ggml.tokens array. */
    const char *tensorname = "tokenizer.ggml.tokens";
    size_t tensorname_len = strlen(tensorname);
    while (gguf_get_key(ctx, &key)) {
        if (key.namelen == tensorname_len &&
            memcmp(key.name, tensorname, key.namelen) == 0 &&
            key.type == GGUF_VALUE_TYPE_ARRAY)
        {
            /* Found the tokens array. */
            uint32_t etype = key.val->array.type;
            uint64_t len = key.val->array.len;

            if (etype != GGUF_VALUE_TYPE_STRING) {
                fprintf(stderr,
                    "Unexpected token type in array (not string)\n");
                return NULL;
            }

            printf("Found %llu tokens in vocabulary\n",
                (unsigned long long)len);

            /* Allocate array for all token strings. */
            tokens = calloc(len, sizeof(char*));
            if (!tokens) {
                fprintf(stderr, "Failed to allocate token array.\n");
                return NULL;
            }

            /* Skip array header. */
            ctx->off += 4 + 8; /* 4 for type, 8 for length. */

            /* Load all token strings. */
            for (uint64_t j = 0; j < len; j++) {
                struct gguf_string *str =
                    (struct gguf_string*)(ctx->data + ctx->off);

                /* Allocate and copy token string. */
                tokens[j] = malloc(str->len+1);
                if (!tokens[j]) {
                    fprintf(stderr, "Failed to allocate token string\n");
                    /* Free already allocated tokens and return error. */
                    free_token_strings(tokens,j);
                    return NULL;
                }

                memcpy(tokens[j], str->string, str->len);
                tokens[j][str->len] = '\0';

                /* Move to next string. */
                ctx->off += 8 + str->len;

                /* Show progress. */
                if ((j + 1) % 10000 == 0) {
                    printf("  Loaded %llu / %llu token strings...\n",
                           (unsigned long long)(j + 1),
                           (unsigned long long)len);
                }
            }

            *num_tokens_out = len;
            printf("Successfully loaded all %llu token strings\n",
                   (unsigned long long)len);
            return tokens;
        } else {
            /* Skip this key-value pair. */
            gguf_do_with_value(ctx, key.type, key.val, NULL, 0, 0, NULL);
        }
    }

    fprintf(stderr, "Could not find tokenizer.ggml.tokens array\n");
    return NULL;
}

/* Process the token embeddings tensor and add to Redis */
int process_token_embeddings(gguf_ctx *ctx, redisContext *rctx,
                            const char *key_name, char **token_strings,
                            uint64_t num_token_strings) {
    gguf_tensor tensor;
    int found = 0;

    /* Skip all key-value pairs to get to tensors. */
    gguf_skip_key_values_section(ctx);

    /* Look for token_embd.weight tensor. */
    const char *tensorname = "token_embd.weight";
    size_t tensorname_len = strlen(tensorname);
    while (gguf_get_tensor(ctx, &tensor)) {
        if (tensor.namelen == tensorname_len &&
            memcmp(tensor.name, tensorname, tensor.namelen) == 0)
        {
            found = 1;
            break;
        }
    }

    if (!found) {
        fprintf(stderr, "Could not find token_embd.weight tensor\n");
        return 0;
    }

    printf("\nFound token embeddings tensor:\n");
    printf("  Type: %s\n", gguf_get_tensor_type_name(tensor.type));
    printf("  Dimensions: [%llu, %llu]\n",
           (unsigned long long)tensor.dim[0],
           (unsigned long long)tensor.dim[1]);
    printf("  Total tokens: %llu\n", (unsigned long long)tensor.dim[1]);
    printf("  Embedding dimension: %llu\n", (unsigned long long)tensor.dim[0]);

    uint64_t emb_dim = tensor.dim[0];
    uint64_t num_tokens = tensor.dim[1];

    /* Verify that we have matching number of tokens. */
    if (num_tokens != num_token_strings) {
        fprintf(stderr, "Warning: Mismatch between embedding tokens (%llu) and vocabulary (%llu)\n",
                (unsigned long long)num_tokens,
                (unsigned long long)num_token_strings);
        /* Use the minimum to be safe */
        if (num_tokens > num_token_strings) {
            num_tokens = num_token_strings;
        }
    }

    /* Convert tensor to float if needed, there are files where the
     * embeddings are also quantized. */
    printf("Converting tensor to float format...\n");
    float *embeddings = gguf_tensor_to_float(&tensor);
    if (!embeddings) {
        if (errno == EINVAL) {
            fprintf(stderr, "Unsupported tensor type for conversion: %s\n",
                    gguf_get_tensor_type_name(tensor.type));
        } else {
            fprintf(stderr, "Out of memory converting tensor\n");
        }
        return 0;
    }

    printf("\nAdding %llu tokens to Redis key '%s'...\n",
           (unsigned long long)num_tokens, key_name);

    /* Process each token. */
    for (uint64_t token_id = 0; token_id < num_tokens; token_id++) {
        /* Get the token string. */
        char *token_str = token_strings[token_id];

        /* Get the embedding vector for this token. */
        float *token_emb = embeddings + (token_id * emb_dim);

        /* Build VADD command: VADD key FP32 vector element */
        const char *argv[5];
        size_t arglen[5];

        argv[0] = "VADD";
        arglen[0] = 4;

        argv[1] = key_name;
        arglen[1] = strlen(key_name);

        argv[2] = "FP32";
        arglen[2] = 4;

        argv[3] = (char*)token_emb;
        arglen[3] = emb_dim * sizeof(float);

        argv[4] = token_str;
        arglen[4] = strlen(token_str);

        /* Execute the command. */
        redisReply *reply = redisCommandArgv(rctx, 5, argv, arglen);
        if (!reply) {
            fprintf(stderr, "Error executing VADD for token %llu: %s\n",
                    (unsigned long long)token_id, rctx->errstr);
            free(embeddings);
            return 0;
        }

        if (reply->type == REDIS_REPLY_ERROR) {
            fprintf(stderr, "VADD error for token %llu (%s): %s\n",
                    (unsigned long long)token_id, token_str, reply->str);
            freeReplyObject(reply);
            free(embeddings);
            return 0;
        }

        freeReplyObject(reply);
        tokens_processed++;

        /* Progress indicator every 10000 tokens. */
        if (tokens_processed % 10000 == 0) {
            printf("  Processed %llu / %llu tokens (%.1f%%)\n",
                   (unsigned long long)tokens_processed,
                   (unsigned long long)num_tokens,
                   (double)tokens_processed * 100.0 / num_tokens);
        }
    }

    printf("\nSuccessfully added all %llu tokens to Redis\n",
           (unsigned long long)num_tokens);

    free(embeddings);
    return 1;
}

/* Print usage information */
void usage(const char *progname) {
    fprintf(stderr, "Usage: %s <gguf-file> <redis-key> [options]\n", progname);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h <host>    Redis host (default: 127.0.0.1)\n");
    fprintf(stderr, "  -p <port>    Redis port (default: 6379)\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s model.gguf llm_embeddings\n", progname);
    fprintf(stderr, "  %s model.gguf llm_embeddings -h localhost -p 6379\n", progname);
}

int main(int argc, char **argv) {
    char *gguf_file = NULL;
    char *redis_key = NULL;
    char *redis_host = "127.0.0.1";
    int redis_port = 6379;
    char **token_strings = NULL;
    uint64_t num_token_strings = 0;

    /* Parse command line arguments */
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    gguf_file = argv[1];
    redis_key = argv[2];

    /* Parse optional arguments. */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            redis_host = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            redis_port = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    printf("==============================================\n");
    printf("GGUF to Redis Vector Set Importer\n");
    printf("==============================================\n");
    printf("GGUF file: %s\n", gguf_file);
    printf("Redis target: %s:%d\n", redis_host, redis_port);
    printf("Vector set key: %s\n", redis_key);
    printf("==============================================\n\n");

    /* Open GGUF file. */
    printf("Loading GGUF file...\n");
    gguf_ctx *ctx = gguf_open(gguf_file);
    if (!ctx) {
        fprintf(stderr, "Failed to open GGUF file: %s\n", strerror(errno));
        return 1;
    }

    printf("GGUF file loaded successfully (version %d)\n", ctx->header->version);
    printf("  Key-value pairs: %llu\n", (unsigned long long)ctx->header->metadata_kv_count);
    printf("  Tensors: %llu\n\n", (unsigned long long)ctx->header->tensor_count);

    /* First, load all token strings into memory. */
    printf("Loading vocabulary tokens...\n");
    token_strings = load_token_strings(ctx, &num_token_strings);
    if (!token_strings) {
        fprintf(stderr, "Failed to load token strings\n");
        gguf_close(ctx);
        return 1;
    }

    /* Connect to Redis. */
    printf("\nConnecting to Redis...\n");
    redisContext *rctx = redisConnect(redis_host, redis_port);
    if (!rctx || rctx->err) {
        if (rctx) {
            fprintf(stderr, "Redis connection error: %s\n", rctx->errstr);
            redisFree(rctx);
        } else {
            fprintf(stderr, "Cannot allocate redis context\n");
        }
        free_token_strings(token_strings, num_token_strings);
        gguf_close(ctx);
        return 1;
    }

    printf("Connected to Redis successfully\n");

    /* Process the embeddings, adding it to Redis. */
    if (!process_token_embeddings(ctx, rctx, redis_key,
                                 token_strings, num_token_strings)) {
        fprintf(stderr, "Failed to process token embeddings\n");
        redisFree(rctx);
        free_token_strings(token_strings, num_token_strings);
        gguf_close(ctx);
        return 1;
    }

    /* Cleanup and reporting. */
    free_token_strings(token_strings, num_token_strings);
    redisFree(rctx);
    gguf_close(ctx);

    printf("\n==============================================\n");
    printf("Import completed successfully!\n");
    printf("Total tokens added: %llu\n", (unsigned long long)tokens_processed);
    printf("==============================================\n\n");
    printf("You can now use VSIM to find similar tokens:\n");
    printf("  VSIM %s ELE \"apple\" COUNT 10\n", redis_key);
    printf("  VSIM %s ELE \"python\" WITHSCORES\n", redis_key);
    printf("  VCARD %s  # Check total count\n", redis_key);
    return 0;
}
