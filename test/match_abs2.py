from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UPat, PatternMatcher, graph_rewrite, renderer as uop_renderer, Ops, UOp

print("[match_abs] Demo: simplify (abs(t))**2 to t*t")

# Helper to render a UOp into a compact string
def render_uop(u):
  r = graph_rewrite(u, uop_renderer)
  if r.op is Ops.NOOP and isinstance(r.arg, str):
    return r.arg
  # fallback
  return f"{u.op}({', '.join(s.op.name for s in u.src)})"

# Build the example graph
t = Tensor([11, 12])
exp_pow = t.abs()**2
exp_mul = t.abs()*t.abs()

# Define a pattern for sign(t):
# sign(t) == where(t!=0, where(t<0, -1, 1), 0) [optionally + t*0]
_t = UPat.var("t")
zero = UPat.const(None, 0)
neg_one = UPat.const(None, -1)
pos_one = UPat.const(None, 1)

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

def show_case(name, pm, u):
  print(f"\n{name}:")
  print("  Before:", render_uop(u))
  nu = graph_rewrite(u, pm, name="abs2->sq")
  print("  After: ", render_uop(nu))
  print("  Matched:", nu is not u)
  print("  Is square mul:", nu.op is Ops.MUL and len(nu.src)==2 and nu.src[0] is nu.src[1])
  print("  Result:", Tensor(nu, device=t.device).tolist())

#show_case("pow(abs(t),2)", pm_abs2, exp_pow.uop)
#show_case("abs(t)*abs(t)", pm_abs2, exp_mul.uop)


def _is_const_val(u:UOp, val:float) -> bool:
  # matches both CONST and VCONST with all elements equal to val
  if u.op is Ops.CONST: return float(u.arg) == float(val)
  if u.op is Ops.VCONST:
    try: return all(float(v) == float(val) for v in u.arg)
    except TypeError: return False
  return False

def _is_sign_of(s:UOp, x:UOp) -> bool:
  # sign(x) without dtype-fix term: x.ne(0).where((x<0).where(-1,1), 0)
  if s.op is not Ops.WHERE or len(s.src) != 3: return False
  cond_nz, tbranch, fbranch = s.src
  # false branch must be 0
  if not _is_const_val(fbranch, 0): return False
  # cond must be x != 0 OR a bool cast of x
  cond_is_nz = (cond_nz.op is Ops.CMPNE and len(cond_nz.src) == 2 and cond_nz.src[0] is x and _is_const_val(cond_nz.src[1], 0))
  cond_is_bool_cast = (cond_nz.op is Ops.CAST and cond_nz.dtype.scalar() == dtypes.bool and len(cond_nz.src) == 1 and cond_nz.src[0] is x)
  if not (cond_is_nz or cond_is_bool_cast): return False
  # true branch must be (x<0).where(-1, 1)
  if tbranch.op is not Ops.WHERE or len(tbranch.src) != 3: return False
  cond_lt0, tneg, tpos = tbranch.src
  if not (cond_lt0.op is Ops.CMPLT and len(cond_lt0.src) == 2 and cond_lt0.src[0] is x and _is_const_val(cond_lt0.src[1], 0)): return False
  if not (_is_const_val(tneg, -1) and _is_const_val(tpos, 1)): return False
  print(f"Matched sign off")
  return True

pm_abs2_0 = PatternMatcher([((UPat.var("a") * UPat.var("a")),
    lambda a: (y*y) if (a.op is Ops.MUL and len(a.src) == 2 and (
                        (_is_sign_of(a.src[0], y:=a.src[1])) or (_is_sign_of(a.src[1], y:=a.src[0])))) else None),])


show_case("pow(abs(t),2)", pm_abs2_0, exp_pow.uop)
show_case("abs(t)*abs(t)", pm_abs2_0, exp_mul.uop)
