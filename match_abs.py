from tinygrad import Tensor
from tinygrad.uop.ops import UPat, PatternMatcher, graph_rewrite, renderer as uop_renderer, Ops

print("[match_abs] Demo: simplify (abs(t))**2 to t*t")

# Helper to render a UOp into a compact string
def render_uop(u):
  r = graph_rewrite(u, uop_renderer)
  if r.op is Ops.NOOP and isinstance(r.arg, str):
    return r.arg
  # fallback
  return f"{u.op}({', '.join(s.op.name for s in u.src)})"

# Build the example graph
t = Tensor([11])
exp_pow = t.abs()**2
exp_mul = t.abs()*t.abs()

# Define a pattern for sign(t):
# sign(t) == where(t!=0, where(t<0, -1, 1), 0) [optionally + t*0]
_t = UPat.var("t")
# allow CONST or RESHAPE(CONST) for broadcast scalars
_zero_c = UPat.const(None, 0)
_neg1_c = UPat.const(None, -1)
_pos1_c = UPat.const(None, 1)
zero = UPat.any(_zero_c, UPat(Ops.RESHAPE, src=(_zero_c,)))
neg_one = UPat.any(_neg1_c, UPat(Ops.RESHAPE, src=(_neg1_c,)))
pos_one = UPat.any(_pos1_c, UPat(Ops.RESHAPE, src=(_pos1_c,)))

is_nonzero = UPat(Ops.CMPNE, src=(_t, zero))
is_neg = UPat(Ops.CMPLT, src=(_t, zero))
base_sign = UPat(Ops.WHERE, src=(is_neg, neg_one, pos_one))
sign_core = UPat(Ops.WHERE, src=(is_nonzero, base_sign, zero))
# allow optional + t*0 in sign definition
opt_zero_term = UPat(Ops.MUL, src=(_t, zero))
sign_pat = UPat.any(sign_core, UPat(Ops.ADD, src=(sign_core, opt_zero_term)))

# abs(t) == t * sign(t)
abs_pat = UPat(Ops.MUL, src=[_t, sign_pat])  # list -> allow commutative order

# Two possible shapes to simplify:
# 1) pow(abs(t), 2) where 2 may be a scalar or a broadcast RESHAPE of 2
exp2_const = UPat.const(None, 2)
exp2_bcast = UPat(Ops.RESHAPE, src=(exp2_const,))
pow2_abs_pat = UPat(Ops.POW, src=(abs_pat, UPat.any(exp2_const, exp2_bcast)))
# 2) abs(t) * abs(t)
mul_abs_abs_pat = UPat(Ops.MUL, src=[abs_pat, abs_pat])

# Rewrite to t*t
def rewrite_abs2(t):
  return t * t

pm_abs2 = PatternMatcher([
  (pow2_abs_pat, rewrite_abs2),
  (mul_abs_abs_pat, rewrite_abs2),
])

def show_case(name, u):
  print(f"\n{name}:")
  print("  Before:", u)#render_uop(u))
  nu = graph_rewrite(u, pm_abs2, name="abs2->sq")
  print("  After: ", nu)# render_uop(nu))
  print("  Matched:", nu is not u)
  print("  Is square mul:", nu.op is Ops.MUL and len(nu.src)==2 and nu.src[0] is nu.src[1])
  print("  Result:", Tensor(nu, device=t.device).tolist())

show_case("pow(abs(t),2)", exp_pow.uop)
show_case("abs(t)*abs(t)", exp_mul.uop)
