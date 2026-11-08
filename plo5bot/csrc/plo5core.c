/*
 * plo5core.c - fast Short Deck PLO5 evaluation and Monte Carlo hand features.
 *
 * Card encoding is identical to evaluator.py: rank = c >> 2 (0 = six .. 8 = ace),
 * suit = c & 3.  36 cards.
 *
 * Five-card scoring uses an ADDITIVE rank key: key = sum over the 5 cards of
 * 5^rank.  Each rank appears at most 4 times, so the key uniquely identifies
 * the rank multiset (base-5 digits) and the key of a 2-card hole pair plus the
 * key of a 3-card board triple is simply their sum - no sorting needed.
 *   NF[key] = score ordinal of the multiset when the 5 cards are NOT a flush
 *   FL[key] = score ordinal when all 5 cards share a suit (distinct ranks)
 * Both tables are generated in Python from evaluator.py's own scoring rules
 * (see plo5bot/core.py) and handed over with p5_init(), so the ordering is
 * exactly the referee's ordering.  Ordinals: bigger = better, equal = tie.
 *
 * No libc calls besides memcpy/memset, no malloc: the library has no
 * dependencies beyond the oldest glibc symbols, so it loads in any
 * x86-64 Linux image.
 */
#include <stdint.h>
#include <string.h>

#define NKEYS 1953125 /* 5^9 */
#define NCARDS 36

static uint16_t NF[NKEYS];
static uint16_t FL[NKEYS];
static int g_ready = 0;

static const int POW5[9] = {1, 5, 25, 125, 625, 3125, 15625, 78125, 390625};
static const int PAIRS[10][2] = {{0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}};
static const int TRIPS[10][3] = {{0,1,2},{0,1,3},{0,1,4},{0,2,3},{0,2,4},{0,3,4},
                                  {1,2,3},{1,2,4},{1,3,4},{2,3,4}};

int p5_init(const uint16_t *nf, const uint16_t *fl) {
    memcpy(NF, nf, sizeof(NF));
    memcpy(FL, fl, sizeof(FL));
    g_ready = 1;
    return NKEYS;
}

int p5_ready(void) { return g_ready; }

/* ---------------------------------------------------------------- RNG ---- */
typedef struct { uint64_t s; } rng_t;
static inline uint64_t rng_next(rng_t *r) { /* splitmix64 */
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline int rng_below(rng_t *r, int n) {
    return (int)(((rng_next(r) >> 32) * (uint64_t)n) >> 32);
}

/* -------------------------------------------------- precomputed pieces ---- */
typedef struct { int key[10]; int suit[10]; } hole_t;   /* suit = -1 if offsuit */
typedef struct { int key[10]; int suit[10]; } board_t;  /* suit = -1 if not mono */

static inline void prep_hole(const int *h, hole_t *o) {
    for (int i = 0; i < 10; i++) {
        int a = h[PAIRS[i][0]], b = h[PAIRS[i][1]];
        o->key[i] = POW5[a >> 2] + POW5[b >> 2];
        o->suit[i] = ((a & 3) == (b & 3)) ? (a & 3) : -1;
    }
}
static inline void prep_board(const int *bd, board_t *o) {
    for (int j = 0; j < 10; j++) {
        int a = bd[TRIPS[j][0]], b = bd[TRIPS[j][1]], c = bd[TRIPS[j][2]];
        o->key[j] = POW5[a >> 2] + POW5[b >> 2] + POW5[c >> 2];
        o->suit[j] = ((a & 3) == (b & 3) && (b & 3) == (c & 3)) ? (a & 3) : -1;
    }
}
static inline int eval_prepped(const hole_t *h, const board_t *b) {
    int best = 0;
    for (int i = 0; i < 10; i++) {
        const int hk = h->key[i], hs = h->suit[i];
        for (int j = 0; j < 10; j++) {
            int k = hk + b->key[j];
            int v = (hs >= 0 && hs == b->suit[j]) ? FL[k] : NF[k];
            if (v > best) best = v;
        }
    }
    return best;
}

/* Best Omaha score ordinal: exactly 2 of 5 hole cards + exactly 3 of 5 board. */
int p5_omaha(const int *hole, const int *board) {
    hole_t h; board_t b;
    prep_hole(hole, &h); prep_board(board, &b);
    return eval_prepped(&h, &b);
}

/* Score of exactly five cards (for tests). */
int p5_five(const int *c) {
    int key = 0, s = c[0] & 3, fl = 1;
    for (int i = 0; i < 5; i++) { key += POW5[c[i] >> 2]; if ((c[i] & 3) != s) fl = 0; }
    return fl ? FL[key] : NF[key];
}

/* Batch: n hands (n x 5) against one board (5) -> out[n]. */
void p5_omaha_batch(const int *holes, int n, const int *board, int *out) {
    board_t b; prep_board(board, &b);
    for (int i = 0; i < n; i++) { hole_t h; prep_hole(holes + 5 * i, &h); out[i] = eval_prepped(&h, &b); }
}

/* Build the list of cards not in `used` (bitmask). Returns count. */
static int remaining(uint64_t used, int *rem) {
    int n = 0;
    for (int c = 0; c < NCARDS; c++) if (!((used >> c) & 1ULL)) rem[n++] = c;
    return n;
}

/*
 * Hand-strength features against a uniformly random opponent hand.
 *   hole[5], board[nb] with nb in {0,3,4,5}; extra dead cards dead[nd].
 *   R runouts of the missing board cards; for each, K random opponent hands.
 *   HS_r = (wins + ties/2) / K on runout r.
 * out[0] = mean HS_r          (= equity vs a random hand)
 * out[1] = mean HS_r^2        (potential: high for drawing hands)
 * out[2] = fraction of runouts with HS_r >= 0.9  (nut potential)
 * Total evaluations: R * (K + 1).
 */
void p5_hs_features(const int *hole, const int *board, int nb, const int *dead, int nd,
                    int R, int K, uint64_t seed, double *out) {
    rng_t rng = {seed * 0x2545F4914F6CDD1DULL + 0x1234567ULL};
    uint64_t used = 0;
    for (int i = 0; i < 5; i++) used |= 1ULL << hole[i];
    for (int i = 0; i < nb; i++) used |= 1ULL << board[i];
    for (int i = 0; i < nd; i++) used |= 1ULL << dead[i];
    int rem[NCARDS]; int n = remaining(used, rem);
    int need = 5 - nb;
    if (need == 0) R = 1;
    hole_t me; prep_hole(hole, &me);
    double s1 = 0, s2 = 0, snut = 0;
    int full[5];
    for (int i = 0; i < nb; i++) full[i] = board[i];
    for (int r = 0; r < R; r++) {
        for (int i = 0; i < need; i++) {          /* partial Fisher-Yates */
            int j = i + rng_below(&rng, n - i);
            int t = rem[i]; rem[i] = rem[j]; rem[j] = t;
            full[nb + i] = rem[i];
        }
        board_t b; prep_board(full, &b);
        int mine = eval_prepped(&me, &b);
        int wins2 = 0;                               /* 2 per win, 1 per tie */
        int m = n - need;
        int *pool = rem + need;
        for (int k = 0; k < K; k++) {
            int opp[5];
            for (int i = 0; i < 5; i++) {
                int j = i + rng_below(&rng, m - i);
                int t = pool[i]; pool[i] = pool[j]; pool[j] = t;
                opp[i] = pool[i];
            }
            hole_t oh; prep_hole(opp, &oh);
            int theirs = eval_prepped(&oh, &b);
            wins2 += (mine > theirs) ? 2 : (mine == theirs ? 1 : 0);
        }
        double hs = wins2 / (2.0 * K);
        s1 += hs; s2 += hs * hs; snut += (hs >= 0.9);
    }
    out[0] = s1 / R; out[1] = s2 / R; out[2] = snut / R;
}

/*
 * Equity of `hole` against a weighted list of opponent hands opp[n*5] given a
 * partial board, averaging over R random runouts per opponent hand.  Opponent
 * hands that collide with known cards get weight 0.  Returns the weighted
 * equity; also writes per-hand equities into per_hand (may be NULL).
 */
double p5_equity_vs_hands(const int *hole, const int *board, int nb, const int *opp,
                          const double *w, int n, int R, uint64_t seed, double *per_hand) {
    rng_t rng = {seed * 0x9E3779B97F4A7C15ULL + 77ULL};
    uint64_t used = 0;
    for (int i = 0; i < 5; i++) used |= 1ULL << hole[i];
    for (int i = 0; i < nb; i++) used |= 1ULL << board[i];
    hole_t me; prep_hole(hole, &me);
    int need = 5 - nb;
    if (need == 0) R = 1;
    double num = 0, den = 0;
    board_t fixed; int full[5];
    for (int i = 0; i < nb; i++) full[i] = board[i];
    if (need == 0) prep_board(full, &fixed);
    for (int h = 0; h < n; h++) {
        const int *o = opp + 5 * h;
        uint64_t om = 0;
        for (int i = 0; i < 5; i++) om |= 1ULL << o[i];
        if ((om & used) || w[h] <= 0) { if (per_hand) per_hand[h] = -1; continue; }
        hole_t oh; prep_hole(o, &oh);
        double eq;
        if (need == 0) {
            int a = eval_prepped(&me, &fixed), b = eval_prepped(&oh, &fixed);
            eq = a > b ? 1.0 : (a == b ? 0.5 : 0.0);
        } else {
            int rem[NCARDS]; int m = remaining(used | om, rem);
            int acc = 0;
            for (int r = 0; r < R; r++) {
                for (int i = 0; i < need; i++) {
                    int j = i + rng_below(&rng, m - i);
                    int t = rem[i]; rem[i] = rem[j]; rem[j] = t;
                    full[nb + i] = rem[i];
                }
                board_t b; prep_board(full, &b);
                int a = eval_prepped(&me, &b), c = eval_prepped(&oh, &b);
                acc += a > c ? 2 : (a == c ? 1 : 0);
            }
            eq = acc / (2.0 * R);
        }
        if (per_hand) per_hand[h] = eq;
        num += w[h] * eq; den += w[h];
    }
    return den > 0 ? num / den : 0.5;
}

/*
 * Offline dataset generation for blueprint training.
 * For deal d (seeded by seed + d) deal 15 cards: hole0 = c[0..4], hole1 = c[5..9],
 * board = c[10..14].  For each player p and street s in {flop, turn} write the
 * 3 features of p5_hs_features (with Rf,Kf / Rt,Kt samples), for river write HS
 * against Kr random hands (R=1).  Preflop is looked up from a table in Python.
 *   cards_out[d*15 + i]           the 15 cards
 *   feat_out [d*2*7 + p*7 + f]    f: 0..2 flop, 3..5 turn, 6 river HS
 *   res_out  [d]                  +1 if hole0 wins at showdown, -1 loses, 0 tie
 */
void p5_gen_deals(int n, uint64_t seed, int Rf, int Kf, int Rt, int Kt, int Kr,
                  uint8_t *cards_out, float *feat_out, int8_t *res_out) {
    for (int d = 0; d < n; d++) {
        rng_t rng = {(seed + (uint64_t)d) * 0xD1B54A32D192ED03ULL + 99ULL};
        int deck[NCARDS];
        for (int i = 0; i < NCARDS; i++) deck[i] = i;
        for (int i = 0; i < 15; i++) {
            int j = i + rng_below(&rng, NCARDS - i);
            int t = deck[i]; deck[i] = deck[j]; deck[j] = t;
        }
        int *c = deck;
        for (int i = 0; i < 15; i++) cards_out[d * 15 + i] = (uint8_t)c[i];
        const int *board = c + 10;
        for (int p = 0; p < 2; p++) {
            const int *hole = c + 5 * p;
            const int *other = c + 5 * (1 - p);   /* NOT used as dead: features vs random hand */
            (void)other;
            double f[3];
            uint64_t s0 = rng_next(&rng);
            p5_hs_features(hole, board, 3, 0, 0, Rf, Kf, s0, f);
            float *o = feat_out + (size_t)d * 14 + p * 7;
            o[0] = (float)f[0]; o[1] = (float)f[1]; o[2] = (float)f[2];
            p5_hs_features(hole, board, 4, 0, 0, Rt, Kt, rng_next(&rng), f);
            o[3] = (float)f[0]; o[4] = (float)f[1]; o[5] = (float)f[2];
            p5_hs_features(hole, board, 5, 0, 0, 1, Kr, rng_next(&rng), f);
            o[6] = (float)f[0];
        }
        int a = p5_omaha(c, board), b = p5_omaha(c + 5, board);
        res_out[d] = (int8_t)(a > b ? 1 : (a < b ? -1 : 0));
    }
}

/* Preflop equity vs random hand for many hands: holes[n*5] -> out[n]. */
void p5_preflop_equity(const int *holes, int n, int R, uint64_t seed, double *out) {
    for (int i = 0; i < n; i++) {
        double f[3];
        p5_hs_features(holes + 5 * i, 0, 0, 0, 0, R, 1, seed + (uint64_t)i * 7919ULL, f);
        out[i] = f[0];
    }
}
