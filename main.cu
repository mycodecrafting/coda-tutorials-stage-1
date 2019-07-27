#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;

__constant__
const uint8_t mnt4_alpha[bytes_per_elem] = {13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__constant__
const uint8_t mnt4_modulus[bytes_per_elem] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

using namespace std;
using namespace cuFIXNUM;

struct fp2_pp {
    std::vector<uint8_t*> c0;
    std::vector<uint8_t*> c1;
};

struct fp2_result {
    vector<uint8_t*> r0;
    vector<uint8_t*> r1;
};

template< typename fixnum >
struct mul_and_convert {
  // redc may be worth trying over cios
  typedef modnum_monty_cios<fixnum> modnum;
  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum ac0, fixnum ac1, fixnum bc0, fixnum bc1) {
      fixnum n = array_to_fixnum(mnt4_alpha);
      fixnum my_mod = array_to_fixnum(mnt4_modulus);

      modnum mod = modnum(my_mod);

      fixnum a0_b0;
      fixnum a1_b1;
      fixnum a0_plus_a1;
      fixnum b0_plus_b1;
      fixnum c;

      fixnum a0;
      fixnum a1;
      fixnum b0;
      fixnum b1;
      fixnum alpha;
      mod.to_modnum(a0, ac0);
      mod.to_modnum(a1, ac1);
      mod.to_modnum(b0, bc0);
      mod.to_modnum(b1, bc1);
      mod.to_modnum(alpha, n);

      mod.mul(a0_b0, a0, b0);
      mod.mul(a1_b1, a1, b1);
      mod.add(a0_plus_a1, a0, a1);
      mod.add(b0_plus_b1, b0, b1);
      mod.mul(c, a0_plus_a1, b0_plus_b1);

      fixnum ra0;
      fixnum ra0m;
      mod.mul(ra0m, a1_b1, alpha);
      mod.add(ra0, a0_b0, ra0m);

      fixnum s0;
      mod.from_modnum(s0, ra0);
      r0 = s0;

      fixnum ra1;
      fixnum ra1m;
      mod.sub(ra1m, c, a0_b0);
      mod.sub(ra1, ra1m, a1_b1);

      fixnum s1;
      mod.from_modnum(s1, ra1);
      r1 = s1;
  }
  __device__ fixnum array_to_fixnum(const uint8_t* arr) {
      return fixnum(((fixnum*)arr)[fixnum::layout::laneIdx()]);
  }
};

template< int fn_bytes, typename fixnum_array >
vector<uint8_t*> get_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t* local_results = new uint8_t[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);
    vector<uint8_t*> res_v;
    for (int n = 0; n < nelts; n++) {
      uint8_t* a = (uint8_t*)malloc(fn_bytes*sizeof(uint8_t));
      for (int i = 0; i < fn_bytes; i++) {
        a[i] = local_results[n*fn_bytes + i];
      }
      res_v.emplace_back(a);
    }
    delete local_results;
    return res_v;
}

template< int fn_bytes, typename word_fixnum, template <typename> class Func >
fp2_result compute_fp2_product(fp2_pp a, fp2_pp b) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    int nelts = a.c0.size();

    uint8_t *input_ac0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_ac1 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_bc0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_bc1 = new uint8_t[fn_bytes * nelts];

    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_ac0[i] = a.c0[i/fn_bytes][i%fn_bytes];
      input_ac1[i] = a.c1[i/fn_bytes][i%fn_bytes];
      input_bc0[i] = b.c0[i/fn_bytes][i%fn_bytes];
      input_bc1[i] = b.c1[i/fn_bytes][i%fn_bytes];
    }

    fixnum_array *res0, *res1, *in_ac0, *in_ac1, *in_bc0, *in_bc1;
    in_ac0 = fixnum_array::create(input_ac0, fn_bytes * nelts, fn_bytes);
    in_ac1 = fixnum_array::create(input_ac1, fn_bytes * nelts, fn_bytes);
    in_bc0 = fixnum_array::create(input_bc0, fn_bytes * nelts, fn_bytes);
    in_bc1 = fixnum_array::create(input_bc1, fn_bytes * nelts, fn_bytes);
    res0 = fixnum_array::create(nelts);
    res1 = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res0, res1, in_ac0, in_ac1, in_bc0, in_bc1);

    fp2_result res;
    res.r0 = get_fixnum_array<fn_bytes, fixnum_array>(res0, nelts);
    res.r1 = get_fixnum_array<fn_bytes, fixnum_array>(res1, nelts);

    delete in_ac0;
    delete in_ac1;
    delete in_bc0;
    delete in_bc1;
    delete res0;
    delete res1;
    delete[] input_ac0;
    delete[] input_ac1;
    delete[] input_bc0;
    delete[] input_bc1;

    return res;
}

uint8_t* read_mnt_fq(FILE* inputs) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem, sizeof(uint8_t));
  if (buf == NULL) {
    printf("read_mnt_fq: could not allocate memory\n");
    exit(-1);
  }
  fread((void*)buf, io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

void write_mnt_fq(uint8_t* fq, FILE* outputs) {
  fwrite((void *) fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  auto inputs = fopen(argv[2], "r");
  auto outputs = fopen(argv[3], "w");

  size_t n;

  while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }

    fp2_pp x;
    for (size_t i = 0; i < n; ++i) {
      x.c0.emplace_back(read_mnt_fq(inputs));
      x.c1.emplace_back(read_mnt_fq(inputs));
    }

    fp2_pp y;
    for (size_t i = 0; i < n; ++i) {
      y.c0.emplace_back(read_mnt_fq(inputs));
      y.c1.emplace_back(read_mnt_fq(inputs));
    }

    fp2_result res = compute_fp2_product<bytes_per_elem, u64_fixnum, mul_and_convert>(x, y);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq(res.r0[i], outputs);
      write_mnt_fq(res.r1[i], outputs);
    }

    for (size_t i = 0; i < n; ++i) {
      free(x.c0[i]);
      free(x.c1[i]);
      free(y.c0[i]);
      free(y.c1[i]);
      free(res.r0[i]);
      free(res.r1[i]);
    }
  }

  fclose(outputs);

  return 0;
}
