import pytest
import tilelang
import tilelang.ascend.language as T


def example(case, threads=32):
    @T.prim_func
    def run(team: T.comm.Team, data: T.comm.Window((128,), "int32"), result: T.Tensor((32,), "int32")):
        with T.Kernel(1), T.Vector(vector=1):
            queue = T.jetty.workspace()
            with T.SimtVF(threads=threads):
                lane = T.get_thread_binding()
                if case != "unbound":
                    T.jetty.init(team, 1 - T.comm.rank(team), queue)
                if case == "divergent":
                    if lane == 0:
                        T.jetty.publish(team, queue)
                else:
                    T.jetty.post_write(team, queue, src=data[lane : lane + 1], dst=data[64:66], segments=3 if case == "count" else 2)
                    T.jetty.publish(team, queue)
                    if case == "fused":
                        result[lane] = data[lane] * 2
                    T.jetty.drain(team, queue, timeout_cycles=0 if case == "timeout" else 1000000)

    return run


@pytest.mark.parametrize(
    ("case", "threads", "message"),
    [
        ("fused", 64, "threads=32"),
        ("unbound", 32, "same team"),
        ("count", 32, "equal-sized"),
        ("divergent", 32, "entire warp"),
        ("timeout", 32, "timeout_cycles"),
    ],
)
def test_invalid(case, threads, message):
    with pytest.raises(ValueError, match=message):
        example(case, threads)


@pytest.mark.parametrize("case", ["communication", "fused"])
def test_order_and_target(case):
    program = example(case)
    assert int(program.attrs["tl.ascend_jetty"]) == 1
    code = tilelang.lower(program, target="ascend").kernel_source
    vf = code[code.index("__simt_vf__ __launch_bounds__(32)") :]
    calls = ["tl_jetty::init(", "tl_jetty::post_write<2>(", "tl_jetty::publish("]
    if case == "fused":
        calls.append(" * 2)")
    calls.append("tl_jetty::drain<1000000>(")
    offsets = [vf.index(token) for token in calls]
    assert offsets == sorted(offsets)
    with pytest.raises(ValueError, match="plain AscendC"):
        tilelang.lower(program, target="pto")
