
from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


triton__0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 2, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


triton__1 = async_compile.triton('''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 2, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    x0 = xindex % 1024
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    _tmp5 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + 0
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp1 = tl.load(in_ptr1 + (r2 + (768*tmp0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr3 + (r2 + (768*tmp2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp4 = tmp1 + tmp3
        _tmp5 = tl.where(rmask & xmask, _tmp5 + tmp4, _tmp5)
        tl.store(out_ptr0 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp4, rmask & xmask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp6 = 768.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp7, xmask)
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
    tmp12 = 768.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.libdevice.rsqrt(tmp15)
    tl.store(in_out_ptr1 + (x3 + tl.zeros([XBLOCK, 1], tl.int32)), tmp16, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp17 = tl.load(out_ptr0 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0)
        tmp18 = tmp17 - tmp7
        tmp19 = tmp18 * tmp16
        tmp21 = tmp19 * tmp20
        tl.store(out_ptr2 + (r2 + (768*x3) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), tmp21, rmask & xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    with torch.cuda._DeviceGuard(2):
        torch.cuda.set_device(2) # no-op to ensure context
        buf0 = empty_strided((1, 1024), (1024, 1), device='cuda', dtype=torch.int64)
        stream2 = get_cuda_stream(2)
        triton__0.run(buf0, 1024, grid=grid(1024), stream=stream2)
        buf1 = empty_strided((12, 1024, 768), (786432, 768, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((12, 1024, 1), (1024, 1, 12288), device='cuda', dtype=torch.float32)
        buf3 = buf2; del buf2  # reuse
        buf5 = empty_strided((12, 1024, 1), (1024, 1, 1), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((12, 1024, 1), (1024, 1, 12288), device='cuda', dtype=torch.float32)
        buf6 = as_strided(buf4, (12, 1024, 1), (1024, 1, 1)); del buf4  # reuse
        buf7 = empty_strided((12, 1024, 768), (786432, 768, 1), device='cuda', dtype=torch.float32)
        triton__1.run(buf3, buf6, primals_3, primals_1, buf0, primals_2, primals_4, buf1, buf5, buf7, 12288, 768, grid=grid(12288), stream=stream2)
        del buf3
        del primals_1
        del primals_2
        return (buf7, buf1, primals_3, primals_4, buf0, buf1, buf5, buf6, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((50304, 768), (768, 1), device='cuda:2', dtype=torch.float32)
    primals_2 = rand_strided((1024, 768), (768, 1), device='cuda:2', dtype=torch.float32)
    primals_3 = rand_strided((12, 1024), (1024, 1), device='cuda:2', dtype=torch.int64)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:2', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4]))
