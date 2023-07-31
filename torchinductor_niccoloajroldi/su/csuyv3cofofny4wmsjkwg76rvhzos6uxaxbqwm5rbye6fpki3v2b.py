
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    x0 = xindex % 1024
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp1 = tl.load(in_ptr1 + (r2 + (768*tmp0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp2 = x0
        tmp3 = tl.load(in_ptr2 + (r2 + (768*tmp2)), rmask, eviction_policy='evict_last', other=0)
        tmp4 = tmp1 + tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tl.store(out_ptr0 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp4, rmask & xmask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp6 = 768.0
    tmp7 = tmp5 / tmp6
    _tmp11 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp9 = tmp8 - tmp7
        tmp10 = tmp9 * tmp9
        _tmp11 = tl.where(rmask & xmask, _tmp11 + tmp10, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp20 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0)
        tmp13 = tmp12 - tmp7
        tmp14 = 768.0
        tmp15 = tmp11 / tmp14
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = tl.libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tl.store(out_ptr2 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp21, rmask & xmask)
