/*
 * train/mccfr.c - External-sampling MCCFR (Lanctot et al. 2009) for the
 * card-abstracted heads-up Short Deck PLO5 game, with the techniques used
 * by Pluribus / RoboPoker:
 *   - linear weighting: regret and average-strategy increments are weighted
 *     by w = 1 + t / interval  (piecewise-linear CFR, Brown & Sandholm 2019)
 *   - regret-based pruning: after a warm-up, 95% of traversals skip actions
 *     whose regret is below PRUNE_T (they are never pruned at prob > 0)
 *   - hogwild multi-threading over shared regret tables
 *
 * The betting tree is the EXACT game tree (exported by train/train_blueprint.py
 * from plo5bot/tree.py); only cards are abstracted. Chance is sampled from a
 * precomputed dataset of deals: buckets[d][hand][street] and the showdown
 * result of hand 0 vs hand 1. Each iteration samples a deal and randomly
 * assigns its two hands to the button / big blind.
 *
 * Info set = (street, decision node's index within the street, own bucket on
 * that street): perfect recall of the betting, imperfect recall of cards.
 *
 * Build:  gcc -O3 -march=native -pthread -o train/mccfr train/mccfr.c -lm
 * Usage:  train/mccfr TREE DEALS OUT ITERS THREADS SEED [INIT] [T_OFFSET]
 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int8_t player, street, kind, folder;
    int32_t child[4];
    int32_t c0, c1, local;
} node_t;

static node_t *nodes;
static int n_nodes, n_local[4], NB[4];
static double *R[4], *S[4];
static size_t sz[4];

/* ---- Restricted Nash Response mode (Johanson, Zinkevich, Bowling 2008) ----
 * "us" = R/S above (the counter-strategy being learned, both positions).
 * The opponent is, per iteration, either the FIXED model (prob p_fixed) or a
 * FREE learner (RO/SO) that best-responds to us. The model is the blueprint
 * average strategy tilted per street group g (0 preflop, 1 postflop):
 *     sigma_m[a] ~ max(0, sigma_bp[a] * mult[g][a] + add[g][a])  (legal a)
 * Our position u is sampled each iteration. */
static int rnr = 0;
static double *RO[4], *SO[4], *BP[4];       /* opponent free regrets / avg, blueprint avg */
static double p_fixed = 0.0, mult[2][4], add_[2][4];
static uint8_t *bk;      /* [N][2][4] */
static int8_t *res;      /* [N] */
static long n_deals;
static long long iters_total;
static int n_threads;
static long long interval = 2000000;
static long long prune_start;
static long long t_offset = 0;
static const double PRUNE_T = -2.0e5;
static volatile long long global_done = 0;

typedef struct { uint64_t s; } rng_t;
static inline uint64_t rnext(rng_t *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double runif(rng_t *r) { return (rnext(r) >> 11) * (1.0 / 9007199254740992.0); }
static inline long rbelow(rng_t *r, long n) { return (long)(((rnext(r) >> 32) * (uint64_t)n) >> 32) % n; }

static inline void regret_match(const double *r, const node_t *nd, double *sig) {
    double pos = 0; int nl = 0;
    for (int a = 0; a < 4; a++) if (nd->child[a] >= 0) { nl++; if (r[a] > 0) pos += r[a]; }
    for (int a = 0; a < 4; a++) {
        if (nd->child[a] < 0) sig[a] = 0;
        else sig[a] = pos > 0 ? (r[a] > 0 ? r[a] / pos : 0) : 1.0 / nl;
    }
}

/* b[p*4+s] = bucket of player p on street s; res0 = showdown result for player 0 */
static double traverse(int n, int trav, const uint8_t *b, int res0, rng_t *rng, double w, int prune) {
    const node_t *nd = &nodes[n];
    if (nd->kind == 1) {
        double pay0 = nd->folder == 0 ? -(double)nd->c0 : (double)nd->c1;
        return trav == 0 ? pay0 : -pay0;
    }
    if (nd->kind == 2) {
        double pay0 = (double)res0 * nd->c0;
        return trav == 0 ? pay0 : -pay0;
    }
    int p = nd->player, s = nd->street;
    size_t base = ((size_t)nd->local * NB[s] + b[p * 4 + s]) * 4;
    double *r = R[s] + base;
    double sig[4];
    regret_match(r, nd, sig);
    if (p == trav) {
        double v[4] = {0, 0, 0, 0}, ev = 0;
        int explored[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; a++) {
            if (nd->child[a] < 0) continue;
            if (prune && sig[a] == 0 && r[a] < PRUNE_T) continue;
            explored[a] = 1;
            v[a] = traverse(nd->child[a], trav, b, res0, rng, w, prune);
            ev += sig[a] * v[a];
        }
        for (int a = 0; a < 4; a++) if (explored[a]) r[a] += w * (v[a] - ev);
        return ev;
    } else {
        double *sm = S[s] + base;
        for (int a = 0; a < 4; a++) if (nd->child[a] >= 0) sm[a] += w * sig[a];
        double u = runif(rng), acc = 0;
        int pick = -1;
        for (int a = 0; a < 4; a++) {
            if (nd->child[a] < 0) continue;
            pick = a; acc += sig[a];
            if (u < acc) break;
        }
        return traverse(nd->child[pick], trav, b, res0, rng, w, prune);
    }
}

static inline void model_sigma(const node_t *nd, const double *bp, double *sig) {
    int g = nd->street > 0;
    double tot = 0; int nl = 0;
    for (int a = 0; a < 4; a++) {
        if (nd->child[a] < 0) { sig[a] = 0; continue; }
        nl++;
        double v = bp[a] * mult[g][a] + add_[g][a];
        sig[a] = v > 0 ? v : 0;
        tot += sig[a];
    }
    for (int a = 0; a < 4; a++) {
        if (nd->child[a] < 0) continue;
        sig[a] = tot > 0 ? sig[a] / tot : 1.0 / nl;
    }
}

/* RNR traversal. trav = position being updated; u = our position; fixed =
 * opponent plays the model this iteration. Returns value for trav. */
static double traverse_rnr(int n, int trav, int u, int fixed, const uint8_t *b, int res0, rng_t *rng,
                           double w, int prune) {
    const node_t *nd = &nodes[n];
    if (nd->kind == 1) {
        double pay0 = nd->folder == 0 ? -(double)nd->c0 : (double)nd->c1;
        return trav == 0 ? pay0 : -pay0;
    }
    if (nd->kind == 2) {
        double pay0 = (double)res0 * nd->c0;
        return trav == 0 ? pay0 : -pay0;
    }
    int p = nd->player, s = nd->street;
    size_t base = ((size_t)nd->local * NB[s] + b[p * 4 + s]) * 4;
    int is_us = (p == u);
    double sig[4];
    double *r = 0, *sm = 0;
    if (is_us) { r = R[s] + base; sm = S[s] + base; regret_match(r, nd, sig); }
    else if (fixed) { model_sigma(nd, BP[s] + base, sig); }
    else { r = RO[s] + base; sm = SO[s] + base; regret_match(r, nd, sig); }
    if (p == trav) {
        double v[4] = {0, 0, 0, 0}, ev = 0;
        int explored[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; a++) {
            if (nd->child[a] < 0) continue;
            if (r && prune && sig[a] == 0 && r[a] < PRUNE_T) continue;
            if (!r && sig[a] == 0) continue;
            explored[a] = 1;
            v[a] = traverse_rnr(nd->child[a], trav, u, fixed, b, res0, rng, w, prune);
            ev += sig[a] * v[a];
        }
        if (r) for (int a = 0; a < 4; a++) if (explored[a]) r[a] += w * (v[a] - ev);
        return ev;
    }
    if (sm) for (int a = 0; a < 4; a++) if (nd->child[a] >= 0) sm[a] += w * sig[a];
    double x = runif(rng), acc = 0;
    int pick = -1;
    for (int a = 0; a < 4; a++) {
        if (nd->child[a] < 0) continue;
        pick = a; acc += sig[a];
        if (x < acc) break;
    }
    return traverse_rnr(nd->child[pick], trav, u, fixed, b, res0, rng, w, prune);
}

typedef struct { int id; uint64_t seed; long long iters; double ev_sum; } targ_t;

static void *worker(void *arg) {
    targ_t *ta = (targ_t *)arg;
    rng_t rng = {ta->seed};
    uint8_t b[8];
    for (long long i = 0; i < ta->iters; i++) {
        long long t = t_offset + i * n_threads + ta->id;
        double w = 1.0 + (double)(t / interval);
        long d = rbelow(&rng, n_deals);
        int swap = (int)(rnext(&rng) & 1);
        const uint8_t *src = bk + (size_t)d * 8;
        int h0 = swap, h1 = 1 - swap;     /* which dataset hand is the button */
        for (int s = 0; s < 4; s++) { b[s] = src[h0 * 4 + s]; b[4 + s] = src[h1 * 4 + s]; }
        int res0 = swap ? -res[d] : res[d];
        int prune = (t > prune_start) && (runif(&rng) < 0.95);
        if (rnr) {
            int u = (int)(rnext(&rng) & 1);
            int fixed = runif(&rng) < p_fixed;
            for (int trav = 0; trav < 2; trav++) {
                double v = traverse_rnr(0, trav, u, fixed, b, res0, &rng, w, prune);
                if (trav == u) ta->ev_sum += v;
            }
        } else {
            for (int trav = 0; trav < 2; trav++) ta->ev_sum += traverse(0, trav, b, res0, &rng, w, prune);
        }
        if ((i & 0xFFFF) == 0) __sync_fetch_and_add(&global_done, 0x10000);
    }
    return NULL;
}

static void *xread(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(*len);
    if (fread(buf, 1, *len, f) != *len) { perror("read"); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s TREE DEALS OUT ITERS THREADS SEED [INIT] [T_OFFSET]\n", argv[0]);
        return 1;
    }
    size_t len;
    /* tree file: int32 n_nodes, n_local[4], NB[4], then n_nodes node_t */
    int32_t *tf = (int32_t *)xread(argv[1], &len);
    n_nodes = tf[0];
    for (int s = 0; s < 4; s++) { n_local[s] = tf[1 + s]; NB[s] = tf[5 + s]; }
    nodes = (node_t *)(tf + 9);
    if (len != 9 * 4 + (size_t)n_nodes * sizeof(node_t)) { fprintf(stderr, "bad tree file\n"); return 1; }
    /* deals file: int64 n, then n*8 bucket bytes, then n result bytes */
    uint8_t *df = (uint8_t *)xread(argv[2], &len);
    n_deals = (long)*(int64_t *)df;
    bk = df + 8;
    res = (int8_t *)(df + 8 + (size_t)n_deals * 8);
    iters_total = atoll(argv[4]);
    n_threads = atoi(argv[5]);
    uint64_t seed = strtoull(argv[6], 0, 10);
    if (argc > 8) t_offset = atoll(argv[8]);
    prune_start = t_offset + iters_total / 10;
    for (int s = 0; s < 4; s++) {
        sz[s] = (size_t)n_local[s] * NB[s] * 4;
        R[s] = calloc(sz[s], sizeof(double));
        S[s] = calloc(sz[s], sizeof(double));
    }
    if (argc > 7 && strcmp(argv[7], "-") != 0) {
        FILE *f = fopen(argv[7], "rb");
        if (!f) { perror(argv[7]); return 1; }
        for (int s = 0; s < 4; s++) {
            if (fread(R[s], sizeof(double), sz[s], f) != sz[s] || fread(S[s], sizeof(double), sz[s], f) != sz[s]) {
                fprintf(stderr, "bad init file\n"); return 1;
            }
        }
        fclose(f);
    }
    if (argc > 9) {
        /* RNR: params file "p_fixed m0[4] a0[4] m1[4] a1[4]"; INIT (blueprint raw) required */
        FILE *pf = fopen(argv[9], "r");
        if (!pf || argc <= 7 || strcmp(argv[7], "-") == 0) { fprintf(stderr, "RNR needs params + INIT\n"); return 1; }
        int ok = fscanf(pf, "%lf", &p_fixed);
        for (int g = 0; g < 2; g++) {
            for (int a = 0; a < 4; a++) ok += fscanf(pf, "%lf", &mult[g][a]);
            for (int a = 0; a < 4; a++) ok += fscanf(pf, "%lf", &add_[g][a]);
        }
        fclose(pf);
        if (ok != 17) { fprintf(stderr, "bad params file\n"); return 1; }
        rnr = 1;
        for (int s = 0; s < 4; s++) {
            RO[s] = malloc(sz[s] * sizeof(double)); memcpy(RO[s], R[s], sz[s] * sizeof(double));
            SO[s] = calloc(sz[s], sizeof(double));
            BP[s] = malloc(sz[s] * sizeof(double));
            for (size_t i = 0; i < sz[s]; i += 4) {
                double tot = S[s][i] + S[s][i + 1] + S[s][i + 2] + S[s][i + 3];
                for (int a = 0; a < 4; a++) BP[s][i + a] = tot > 0 ? S[s][i + a] / tot : 0.25;
            }
            /* our average restarts so the counter-strategy is not dominated by the blueprint's history */
            memset(S[s], 0, sz[s] * sizeof(double));
        }
        fprintf(stderr, "RNR mode: p_fixed %.3f\n", p_fixed);
    }
    fprintf(stderr, "nodes %d, local %d/%d/%d/%d, buckets %d/%d/%d/%d, deals %ld, iters %lld, threads %d\n",
            n_nodes, n_local[0], n_local[1], n_local[2], n_local[3], NB[0], NB[1], NB[2], NB[3],
            n_deals, iters_total, n_threads);
    pthread_t th[64];
    targ_t ta[64];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < n_threads; i++) {
        ta[i].id = i; ta[i].seed = seed * 1000003ULL + i * 7919ULL + 1; ta[i].ev_sum = 0;
        ta[i].iters = iters_total / n_threads;
        pthread_create(&th[i], 0, worker, &ta[i]);
    }
    /* progress */
    for (;;) {
        struct timespec ts = {10, 0};
        nanosleep(&ts, 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double el = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        long long done = global_done;
        fprintf(stderr, "  %.0fs: ~%lld iters (%.0f/s)\n", el, done, done / el);
        if (done >= iters_total - 0x10000LL * n_threads) break;
    }
    for (int i = 0; i < n_threads; i++) pthread_join(th[i], 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    fprintf(stderr, "done %lld iters in %.1fs (%.0f/s)\n", iters_total, el, iters_total / el);
    FILE *f = fopen(argv[3], "wb");
    for (int s = 0; s < 4; s++) { fwrite(R[s], sizeof(double), sz[s], f); fwrite(S[s], sizeof(double), sz[s], f); }
    fclose(f);
    return 0;
}
