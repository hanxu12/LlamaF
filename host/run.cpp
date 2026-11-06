/*
 * LlamaF host program (XRT) for the ZCU102.
 *
 * Derived from runq.c in llama2.c by Andrej Karpathy (MIT License):
 *   https://github.com/karpathy/llama2.c
 * The Q8_0 group-wise quantization, checkpoint format, tokenizer, sampler and
 * forward pass come from runq.c. LlamaF adds GQMV offload to the FPGA kernels
 * (matmul1/matmul2) and asynchronous weight loading.
 */
/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
//#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
// XRT includes
#include "xrt/xrt_bo.h"
#include <experimental/xrt_xclbin.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
// C includes
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#define K1_NGROUPS 8
#define K2_NGROUPS 22
// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights
float* wcls_ws;
float* wq_ws;
float* wk_ws;
float* wv_ws;
float* w1_ws;
float* w2_ws;
float* w3_ws;
float* wo_ws;
// common bo/map
xrt::bo xb2_bo; 
xrt::bo xq_bo;
xrt::bo wo_bo;
float* xb2_map;
int8_t* xq_map;
int8_t* wo_map;
// qkv
xrt::bo qkv_bo;
xrt::bo wq_wk_wv_bo;
xrt::bo wq_wk_wv_scale_bo;
float* qkv_map;
int8_t* wq_wk_wv_map;
float* wq_wk_wv_scale_map;
// wo
xrt::bo wo_scale_bo;
float* wo_scale_map;
// w1 w3
xrt::bo hb_hb2_bo;
xrt::bo w1_w3_bo;
xrt::bo w1_w3_scale_bo;
float* hb_hb2_map;
int8_t* w1_w3_map;
float* w1_w3_scale_map;
// w2
xrt::bo xb_bo;
xrt::bo hq_bo;
xrt::bo w2_bo;
xrt::bo w2_scale_bo;
float* xb_map;
int8_t* hq_map;
int8_t* w2_map;
float* w2_scale_map;
// logits
xrt::bo wcls_bo;
xrt::bo logits_bo;
xrt::bo logits_scale_bo;
int8_t* wcls_map;
float* logits_map;
float* logits_scale_map;
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float *)calloc(p->dim, sizeof(float));
    s->xb = (float *)calloc(p->dim, sizeof(float));
    s->xb2 = (float *)calloc(p->dim, sizeof(float));
    s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = (int8_t *)calloc(p->dim, sizeof(int8_t)), .s = (float *)calloc(p->dim, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = (int8_t *)calloc(p->hidden_dim, sizeof(int8_t)), .s = (float *)calloc(p->hidden_dim, sizeof(float)) };
    s->q = (float *)calloc(p->dim, sizeof(float));
    s->k = (float *)calloc(kv_dim, sizeof(float));
    s->v = (float *)calloc(kv_dim, sizeof(float));
    s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // all-zero group: write zeros instead of computing 0/0 (NaN -> int8 is undefined behavior)
        if (wmax == 0.0f) {
            for (int i = 0; i < GS; i++) qx->q[group * GS + i] = 0;
            continue;
        }

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

void quantize2(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    int x_cnt = 0;
    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[x_cnt++] = scale;

        // all-zero group: write zeros instead of computing 0/0 (NaN -> int8 is undefined behavior)
        if (wmax == 0.0f) {
            for (int i = 0; i < GS; i++) qx->q[group * GS + i] = 0;
            continue;
        }

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = (QuantizedTensor *)malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void reinitweights_layer(float* weights, QuantizedTensor *original_weights, int layer, int n, int d) {
    int w_cnt = 0;
    for (int l = 0; l < layer; l++) {
        for (int i = 0; i < d; i++) {
            int in = i * n;
            for (int j = 0; j <= n - GS; j += GS) {
                weights[w_cnt++] = (original_weights + l)->s[(in + j) / GS];
            }
        }
    }
    printf("w_cnt: %d\n", w_cnt);
}

void reinitweights(float* weights, QuantizedTensor* original, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    int w_cnt = 0;
    int i;
    for (i = 0; i < d; i++) {

        int in = i * n;

        int j;
        for (j = 0; j <= n - GS; j += GS) {
            weights[w_cnt++] = original->s[(in + j) / GS];
        }
    }
    std::cout << "Reinit wcls_ws done with w_cnt: " << w_cnt << std::endl;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = (float *)malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    reinitweights_layer(wq_ws, w->wq, p->n_layers, p->dim, p->dim);
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    reinitweights_layer(wk_ws, w->wk, p->n_layers, p->dim, kv_dim);
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    reinitweights_layer(wv_ws, w->wv, p->n_layers, p->dim, kv_dim);
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);
    reinitweights_layer(wo_ws, w->wo, p->n_layers, p->dim, p->dim);
    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    reinitweights_layer(w1_ws, w->w1, p->n_layers, p->dim, p->hidden_dim);    
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    reinitweights_layer(w2_ws, w->w2, p->n_layers, p->hidden_dim, p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    reinitweights_layer(w3_ws, w->w3, p->n_layers, p->dim, p->hidden_dim);
    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
    reinitweights(wcls_ws, w->wcls, p->dim, p->vocab_size);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }
    // read in the version number (uint32), has to be 2
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(EXIT_FAILURE); }
    int header_size = 256; // the header size for version 2 in bytes
    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size; // the group size used in quantization
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size; // set as global, as it will be used in many places
    std::cout << "Group size: " << GS << std::endl;
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
    free(wcls_ws);
    free(wq_ws);
    free(wk_ws);
    free(wv_ws);
    free(wo_ws);
    free(w1_ws);
    free(w2_ws);
    free(w3_ws);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}

void matmul2(float* xout, QuantizedTensor *x, QuantizedTensor *w, float* w_remap, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    int w_cnt = 0;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        int x_cnt = 0;
        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w_remap[w_cnt++] * x->s[x_cnt++];
            ival = 0;
        }

        xout[i] = val;
    }
}

// xrt::run::wait() does not throw when a kernel run fails: it returns the command state.
// Stop on anything other than COMPLETED, instead of continuing with stale output buffers.
static void wait_run(xrt::run& run, const char* kernel, int layer) {
    ert_cmd_state state = run.wait();
    if (state != ERT_CMD_STATE_COMPLETED) {
        fprintf(stderr, "error: kernel run '%s' (layer %d) ended with state %d, expected %d (COMPLETED)\n",
                kernel, layer, (int)state, (int)ERT_CMD_STATE_COMPLETED);
        exit(EXIT_FAILURE);
    }
}

void forward(Transformer* transformer, int token, int pos, xrt::device& device, xrt::kernel& krnl, xrt::kernel& matmul2_krnl) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        // quantize(&s->xq, s->xb, dim);
        // matmul(s->q, &s->xq, w->wq + l, dim, dim);
        // matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        // matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
        quantize2(&s->xq, s->xb, dim);
        std::memcpy(xq_map, s->xq.q, dim * sizeof(int8_t)); // dim = 2048  
        std::memcpy(wq_wk_wv_scale_map, s->xq.s, 8 * sizeof(float)); // xs (8)      
        xq_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // wq/wk/wv and their ws were synced right after they were prefetched; only xs changes here
        wq_wk_wv_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, K1_NGROUPS * sizeof(float), 0);
        auto qkv_run = krnl(qkv_bo, xq_bo, wq_wk_wv_bo, wq_wk_wv_scale_bo, dim + kv_dim + kv_dim);
        // prepare buffer for wo
        std::memcpy(wo_map, (w->wo + l)->q, 2048 * 2048 * sizeof(int8_t));
        std::memcpy(wo_scale_map + 8, wo_ws + l * 8 * 2048, 2048 * 8 * sizeof(float));
        // flush wo and its ws to DDR while the qkv kernel is still running (off the critical path)
        wo_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        wo_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        wait_run(qkv_run, "qkv", l);
        qkv_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(s->q, qkv_map, dim * sizeof(float));
        std::memcpy(s->k, qkv_map + dim, kv_dim * sizeof(float));
        std::memcpy(s->v, qkv_map + dim + kv_dim, kv_dim * sizeof(float));

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        // quantize(&s->xq, s->xb, dim);
        // matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        quantize2(&s->xq, s->xb, dim);
        std::memcpy(xq_map, s->xq.q, 2048 * sizeof(int8_t));
        std::memcpy(wo_scale_map, s->xq.s, 8 * sizeof(float));
        // wo and its ws were synced during the qkv kernel; only the activation and xs change here
        xq_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        wo_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, K1_NGROUPS * sizeof(float), 0);
        auto wo_run = krnl(xb2_bo, xq_bo, wo_bo, wo_scale_bo, dim); // s->xb2
        std::memcpy(w1_w3_map + dim * hidden_dim + 1024 * hidden_dim, (w->w3 + l)->q + 1024 * hidden_dim, (1024 * hidden_dim * sizeof(int8_t)));
        // flush the second half of w3 while the wo kernel runs (the first part was synced during w2)
        w1_w3_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 1024 * hidden_dim * sizeof(int8_t), (dim + 1024) * hidden_dim * sizeof(int8_t));
        wait_run(wo_run, "wo", l);
        xb2_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(s->xb2, xb2_map, dim * sizeof(float));

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        // quantize(&s->xq, s->xb, dim);
        // matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        // matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        quantize2(&s->xq, s->xb, dim);
        std::memcpy(xq_map, s->xq.q, dim * sizeof(int8_t)); // dim = 2048
        std::memcpy(w1_w3_scale_map, s->xq.s, 8 * sizeof(float)); // xs (8)
        xq_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // w1/w3 and their ws were synced while the previous kernels ran; only xs changes here
        w1_w3_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, K1_NGROUPS * sizeof(float), 0);
        auto w1_w3_run = krnl(hb_hb2_bo, xq_bo, w1_w3_bo, w1_w3_scale_bo, hidden_dim * 2);
        // pre-allocate buffer for current layer's w2
        if (l != 0) {
            std::memcpy(w2_map, (w->w2 + l)->q, hidden_dim * dim * sizeof(int8_t));
            std::memcpy(w2_scale_map + 22, w2_ws + l * 22 * 2048, 22 * 2048 * sizeof(float));
            w2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            w2_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        // prepare buffer for next layer's wq, wk, wv
        if (l != p->n_layers - 1) {
            std::memcpy(wq_wk_wv_map, (w->wq + l + 1)->q, dim * dim * sizeof(int8_t)); // copy wq (dim * dim = 2048 * 2048)
            std::memcpy(wq_wk_wv_map + dim * dim, (w->wk + l + 1)->q, dim * kv_dim * sizeof(int8_t)); // copy wk (dim * kv_dim = 2048 * 256)
            std::memcpy(wq_wk_wv_map + dim * (dim + kv_dim), (w->wv + l + 1)->q, dim * kv_dim * sizeof(int8_t)); // copy wv (dim * kv_dim = 2048 * 256)
            std::memcpy(wq_wk_wv_scale_map + 8, wq_ws + (l + 1) * 8 * dim, 8 * dim * sizeof(float)); // wq_ws (32 * dim = 32 * 2048)
            std::memcpy(wq_wk_wv_scale_map + 8 + 8 * dim, wk_ws + (l + 1) * 8 * kv_dim, 8 * kv_dim * sizeof(float)); // wk_ws (32 * kv_dim = 32 * 256)
            std::memcpy(wq_wk_wv_scale_map + 8 * (1 + dim + kv_dim), wv_ws + (l + 1) * 8 * kv_dim, 8 * kv_dim * sizeof(float)); // wv_ws (32 * kv_dim = 32 * 256)
            wq_wk_wv_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            wq_wk_wv_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        wait_run(w1_w3_run, "w1_w3", l);
        hb_hb2_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(s->hb, hb_hb2_map, hidden_dim * sizeof(float));
        std::memcpy(s->hb2, hb_hb2_map + hidden_dim, hidden_dim * sizeof(float));

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        // quantize(&s->hq, s->hb, hidden_dim);
        // matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        quantize2(&s->hq, s->hb, hidden_dim); // hidden_dim = 5632
        std::memcpy(hq_map, s->hq.q, hidden_dim * sizeof(int8_t));
        std::memcpy(w2_scale_map, s->hq.s, 22 * sizeof(float));     
        hq_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // w2 and its ws were synced during the w1/w3 kernel; only the 22 xs change here
        w2_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, K2_NGROUPS * sizeof(float), 0);
        auto w2_run = matmul2_krnl(xb_bo, hq_bo, w2_bo, w2_scale_bo, dim);
        // prepare buffer for next layer's w1, w3
        if (l != p->n_layers - 1) {
            std::memcpy(w1_w3_map, (w->w1 + l + 1)->q, dim * hidden_dim * sizeof(int8_t)); // copy w1 (dim * hidden_dim = 2048 * 5632)
            std::memcpy(w1_w3_map + dim * hidden_dim, (w->w3 + l + 1)->q, (1024 * hidden_dim * sizeof(int8_t))); // copy half of w3 // copy w3 (dim * hidden_dim = 2048 * 5632)
            std::memcpy(w1_w3_scale_map + 8, w1_ws + (l + 1) * 8 * hidden_dim, 8 * hidden_dim * sizeof(float)); // w1_ws (8 * hidden_dim = 8 * 5632)
            std::memcpy(w1_w3_scale_map + 8 + 8 * hidden_dim, w3_ws + (l + 1) * 8 * hidden_dim, 8 * hidden_dim * sizeof(float)); // w3_ws (8 * hidden_dim = 8 * 5632)
            // flush w1 + the first half of w3 (the second half is copied and synced during wo)
            w1_w3_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, (dim + 1024) * hidden_dim * sizeof(int8_t), 0);
            w1_w3_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        wait_run(w2_run, "w2", l);
        xb_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(s->xb, xb_map, dim * sizeof(float));
        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    // quantize(&s->xq, x, dim);
    //std::cout << "Quantize2!!" << std::endl;
    quantize2(&s->xq, x, dim);
    // matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    // matmul2(s->logits, &s->xq, w->wcls, wcls_ws, dim, p->vocab_size);
    //std::cout << "Allocate Buffer in Global Memory!!" << std::endl;
    // Map the contents of the buffer object into host memory
    //std::cout << "Map the contents of the buffer object into host memory" << std::endl;
    std::memcpy(xq_map, s->xq.q, 2048 * sizeof(int8_t));
    std::memcpy(logits_scale_map, s->xq.s, 8 * sizeof(float));
    // Synchronize buffer content with device side
    //std::cout << "synchronize input buffer data to device global memory" << std::endl;
    xq_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // wcls_bo and the ws part of logits_scale_bo never change after generate() loads and
    // syncs them once. Only the 8 xs scales at the start change per token, so sync just those.
    // If wcls or its ws are ever modified here, sync the whole buffer again.
    logits_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 8 * sizeof(float), 0);

    //std::cout << "Execution of the kernel\n";
    auto logits_run = krnl(logits_bo, xq_bo, wcls_bo, logits_scale_bo, p->vocab_size);
    // qkv
    std::memcpy(wq_wk_wv_map, w->wq->q, dim * dim * sizeof(int8_t)); // copy wq (dim * dim = 2048 * 2048)
    std::memcpy(wq_wk_wv_map + dim * dim, w->wk->q, dim * kv_dim * sizeof(int8_t)); // copy wk (dim * kv_dim = 2048 * 256)
    std::memcpy(wq_wk_wv_map + dim * (dim + kv_dim), w->wv->q, dim * kv_dim * sizeof(int8_t)); // copy wv (dim * kv_dim = 2048 * 256)
    std::memcpy(wq_wk_wv_scale_map + 8, wq_ws, 8 * dim * sizeof(float)); // wq_ws (32 * dim = 32 * 2048)
    std::memcpy(wq_wk_wv_scale_map + 8 + 8 * dim, wk_ws, 8 * kv_dim * sizeof(float)); // wk_ws (32 * kv_dim = 32 * 256)
    std::memcpy(wq_wk_wv_scale_map + 8 * (1 + dim + kv_dim), wv_ws, 8 * kv_dim * sizeof(float)); // wv_ws (32 * kv_dim = 32 * 256)    
    // wo
    std::memcpy(wo_map, w->wo->q, 2048 * 2048 * sizeof(int8_t));
    std::memcpy(wo_scale_map + 8, wo_ws, 2048 * 8 * sizeof(float));    
    // w1/w3
    std::memcpy(w1_w3_map, w->w1->q, dim * hidden_dim * sizeof(int8_t)); // copy w1 (dim * hidden_dim = 2048 * 5632)
    std::memcpy(w1_w3_map + dim * hidden_dim, w->w3->q, dim * hidden_dim * sizeof(int8_t)); // copy w3 (dim * hidden_dim = 2048 * 5632)
    std::memcpy(w1_w3_scale_map + 8, w1_ws, 8 * hidden_dim * sizeof(float)); // w1_ws (8 * hidden_dim = 8 * 5632)
    std::memcpy(w1_w3_scale_map + 8 + 8 * hidden_dim, w3_ws, 8 * hidden_dim * sizeof(float)); // w3_ws (8 * hidden_dim = 8 * 5632)
    // w2
    std::memcpy(w2_map, (w->w2)->q, hidden_dim * dim * sizeof(int8_t));
    std::memcpy(w2_scale_map + 22, w2_ws, 22 * 2048 * sizeof(float));
    // layer 0 weights for the next token: flush them while the kernel runs / once at init
    wq_wk_wv_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    wq_wk_wv_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    wo_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w1_w3_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w1_w3_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w2_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    wait_run(logits_run, "logits", -1);  // classifier, after the last layer

    // Get the output;
    //std::cout << "Get the output data from the device" << std::endl;
    logits_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    char *input = "<0x0A>";
    if (strcmp(str, "\\n") != 0)
    {
        input = str;
    }
    TokenIndex tok = {.str = input}; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char *)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    std::string binaryFile = "./vadd.xclbin";
    int device_index = 0;
    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin!! " << binaryFile << std::endl;
    auto uuid = device.load_xclbin("./vadd.xclbin");
    auto krnl = xrt::kernel(device, uuid, "matmul1", xrt::kernel::cu_access_mode::exclusive);
    auto matmul2_krnl = xrt::kernel(device, uuid, "matmul2", xrt::kernel::cu_access_mode::exclusive);
    std::cout << "Kernel loaded!!" << std::endl;
    // init bo/map
    xb2_bo = xrt::bo(device, sizeof(float) * 2048, krnl.group_id(0));    
    xq_bo = xrt::bo(device, sizeof(int8_t) * 2048, krnl.group_id(1));
    wo_bo = xrt::bo(device, sizeof(int8_t) * 2048 * 2048, krnl.group_id(2)); // each layer has 2048x2048 weights
    xb2_map = xb2_bo.map<float*>();
    xq_map = xq_bo.map<int8_t*>();
    wo_map = wo_bo.map<int8_t*>();
    // init qkv bo/map
    qkv_bo = xrt::bo(device, sizeof(float) * (2048 + 256 + 256), krnl.group_id(0));  // q (dim) + k (kv_dim) + v (kv_dim)
    wq_wk_wv_bo = xrt::bo(device, sizeof(int8_t) * 2048 * (2048 + 256 + 256), krnl.group_id(2)); // w1 (dim * hidden_dim) + w3 (dim * hidden_dim)
    wq_wk_wv_scale_bo = xrt::bo(device, sizeof(float) * K1_NGROUPS * (2048 + 256 + 256 + 1), krnl.group_id(3)); // xs (8) + w1_ws (5632x8) + w3_ws (5632x8)
    qkv_map = qkv_bo.map<float*>();
    wq_wk_wv_map = wq_wk_wv_bo.map<int8_t*>();
    wq_wk_wv_scale_map = wq_wk_wv_scale_bo.map<float*>();    
    // init wo bo/map
    wo_scale_bo = xrt::bo(device, sizeof(float) * 8 * (2048 + 1), krnl.group_id(3)); // xs (8) + wo_ws (2048x8)
    wo_scale_map = wo_scale_bo.map<float*>();
    // init w1/w3 bo/map
    hb_hb2_bo = xrt::bo(device, sizeof(float) * 5632 * 2, krnl.group_id(0));  // hb (hidden_dim) + hb2 (hidden_dim)
    w1_w3_bo = xrt::bo(device, sizeof(int8_t) * 2048 * 5632 * 2, krnl.group_id(2)); // w1 (dim * hidden_dim) + w3 (dim * hidden_dim)
    w1_w3_scale_bo = xrt::bo(device, sizeof(float) * 8 * (5632 * 2 + 1), krnl.group_id(3)); // xs (8) + w1_ws (5632x8) + w3_ws (5632x8)
    hb_hb2_map = hb_hb2_bo.map<float*>();
    w1_w3_map = w1_w3_bo.map<int8_t*>();
    w1_w3_scale_map = w1_w3_scale_bo.map<float*>();
    // init w2 bo/map
    xb_bo = xrt::bo(device, sizeof(float) * 2048, matmul2_krnl.group_id(0)); // xb (dim)
    hq_bo = xrt::bo(device, sizeof(int8_t) * 5632, matmul2_krnl.group_id(1));  // hq (hidden_dim)
    w2_bo = xrt::bo(device, sizeof(int8_t) * 5632 * 2048, matmul2_krnl.group_id(2)); // w2 (hidden_dim x dim)
    w2_scale_bo = xrt::bo(device, sizeof(float) * 22 * (2048 + 1), matmul2_krnl.group_id(3)); // xs (22) + w2_ws (22 * 2048)
    xb_map = xb_bo.map<float*>();
    hq_map = hq_bo.map<int8_t*>();
    w2_map = w2_bo.map<int8_t*>();
    w2_scale_map = w2_scale_bo.map<float*>();
    // init logits
    logits_bo = xrt::bo(device, sizeof(float) * 32003, krnl.group_id(0));
    wcls_bo = xrt::bo(device, sizeof(int8_t) * 2048 * 32003, krnl.group_id(2));
    logits_scale_bo = xrt::bo(device, sizeof(float) * 8 * (32003 + 1), krnl.group_id(3));
    logits_map = logits_bo.map<float*>();
    wcls_map = wcls_bo.map<int8_t*>();
    logits_scale_map = logits_scale_bo.map<float*>();
    std::memcpy(wcls_map, transformer->weights.wcls->q, 2048 * 32003 * sizeof(int8_t));
    std::memcpy(logits_scale_map + 8, wcls_ws, 32003 * 8 * sizeof(float));
    // the classifier weights and their scales are constant: sync them to the device once here
    wcls_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    logits_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // preload 1st layer
    // qkv
    int dim = 2048;
    int kv_dim = 256;
    int hidden_dim = 5632;
    std::memcpy(wq_wk_wv_map, transformer->weights.wq->q, dim * dim * sizeof(int8_t)); // copy wq (dim * dim = 2048 * 2048)
    std::memcpy(wq_wk_wv_map + dim * dim, transformer->weights.wk->q, dim * kv_dim * sizeof(int8_t)); // copy wk (dim * kv_dim = 2048 * 256)
    std::memcpy(wq_wk_wv_map + dim * (dim + kv_dim), transformer->weights.wv->q, dim * kv_dim * sizeof(int8_t)); // copy wv (dim * kv_dim = 2048 * 256)
    std::memcpy(wq_wk_wv_scale_map + 8, wq_ws, 8 * dim * sizeof(float)); // wq_ws (32 * dim = 32 * 2048)
    std::memcpy(wq_wk_wv_scale_map + 8 + 8 * dim, wk_ws, 8 * kv_dim * sizeof(float)); // wk_ws (32 * kv_dim = 32 * 256)
    std::memcpy(wq_wk_wv_scale_map + 8 * (1 + dim + kv_dim), wv_ws, 8 * kv_dim * sizeof(float)); // wv_ws (32 * kv_dim = 32 * 256)    
    // wo
    std::memcpy(wo_map, transformer->weights.wo->q, 2048 * 2048 * sizeof(int8_t));
    std::memcpy(wo_scale_map + 8, wo_ws, 2048 * 8 * sizeof(float));    
    // w1/w3
    std::memcpy(w1_w3_map, transformer->weights.w1->q, dim * hidden_dim * sizeof(int8_t)); // copy w1 (dim * hidden_dim = 2048 * 5632)
    std::memcpy(w1_w3_map + dim * hidden_dim, transformer->weights.w3->q, dim * hidden_dim * sizeof(int8_t)); // copy w3 (dim * hidden_dim = 2048 * 5632)
    std::memcpy(w1_w3_scale_map + 8, w1_ws, 8 * hidden_dim * sizeof(float)); // w1_ws (8 * hidden_dim = 8 * 5632)
    std::memcpy(w1_w3_scale_map + 8 + 8 * hidden_dim, w3_ws, 8 * hidden_dim * sizeof(float)); // w3_ws (8 * hidden_dim = 8 * 5632)        
    // w2
    std::memcpy(w2_map, transformer->weights.w2->q, hidden_dim * dim * sizeof(int8_t));
    std::memcpy(w2_scale_map + 22, w2_ws, 22 * 2048 * sizeof(float));
    // layer 0 weights for the next token: flush them while the kernel runs / once at init
    wq_wk_wv_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    wq_wk_wv_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    wo_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w1_w3_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w1_w3_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w2_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        forward(transformer, token, pos, device, krnl, matmul2_krnl);

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            // copy logits out of the device buffer first: sample() modifies them in place
            std::memcpy(transformer->state.logits, logits_map, sampler->vocab_size * sizeof(float));
            next = sample(sampler, transformer->state.logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tok_tl-chat.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    wcls_ws = (float*)malloc(32003 * K1_NGROUPS * sizeof(float));
    wq_ws = (float*)malloc(22 * K1_NGROUPS * 2048 * sizeof(float)); // (layer = 22, dim/GS = 8, dim = 2048)
    wk_ws = (float*)malloc(22 * K1_NGROUPS * 256 * sizeof(float)); // (layer = 22, dim/GS = 8, kv_dim = 256)
    wv_ws = (float*)malloc(22 * K1_NGROUPS * 256 * sizeof(float)); // (layer = 22, dim/GS = 8, kv_dim = 256)    
    wo_ws = (float*)malloc(22 * K1_NGROUPS * 2048 * sizeof(float)); // (layer = 22, dim/GS = 8, dim = 2048)
    w1_ws = (float*)malloc(22 * K1_NGROUPS * 5632 * sizeof(float)); // (layer = 22, dim/GS = 8, hidden_dim = 5632)    
    w2_ws = (float*)malloc(22 * K2_NGROUPS * 2048 * sizeof(float)); // (layer = 22, hidden_dim/GS = 22, dim = 2048)
    w3_ws = (float*)malloc(22 * K1_NGROUPS * 5632 * sizeof(float)); // (layer = 22, dim/GS = 8, hidden_dim = 5632)    
    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif