"""Levmar 2.6 nonlinear least-squares kernels and C ABI.

The callback trampoline is the only C code in this project. Optimization,
finite differencing, dense normal equations, damping, projection, and linear
constraint transforms are compiled Mojo.
"""

from std.ffi import external_call
from std.math import sqrt
from std.sys.info import simd_width_of


def ptr[
    dtype: DType
](addr: Int) -> UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]]:
    return UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]](
        unsafe_from_address=addr
    )


def scalar[dtype: DType](value: Float64) -> Scalar[dtype]:
    return Scalar[dtype](value)


def callback[
    dtype: DType
](fn_addr: Int, p_addr: Int, dst_addr: Int, m: Int, n: Int, data_addr: Int,):
    if dtype == DType.float64:
        external_call["mlm_call_f64", NoneType](
            fn_addr, p_addr, dst_addr, m, n, data_addr
        )
    else:
        external_call["mlm_call_f32", NoneType](
            fn_addr, p_addr, dst_addr, m, n, data_addr
        )


def finite[dtype: DType](value: Scalar[dtype]) -> Bool:
    if value != value:
        return False
    if dtype == DType.float64:
        return abs(value) <= scalar[dtype](1.7976931348623157e308)
    return abs(value) <= scalar[dtype](3.4028234663852886e38)


def machine_epsilon[dtype: DType]() -> Scalar[dtype]:
    if dtype == DType.float64:
        return scalar[dtype](2.220446049250313e-16)
    return scalar[dtype](1.1920928955078125e-7)


def real_min[dtype: DType]() -> Scalar[dtype]:
    if dtype == DType.float64:
        return scalar[dtype](2.2250738585072014e-308)
    return scalar[dtype](1.1754943508222875e-38)


# levmar: misc_core.c LEVMAR_L2NRMXMY
def residual_norm2[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    x: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    hx: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
) -> Scalar[dtype]:
    var total: Scalar[dtype] = scalar[dtype](0.0)
    for i in range(n):
        var value = x[i] - hx[i]
        dst[i] = value
        total += value * value
    return total


# levmar: lm_core.c LEVMAR_DER normal-equation accumulation
def normal_equations[
    dtype: DType
](
    jac: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    e: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    jtj: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    jte: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    m: Int,
):
    comptime W = simd_width_of[dtype]()
    if dtype == DType.float64 and n * m >= 100_000:
        external_call["cblas_dgemv", NoneType](
            101, 112, n, m, 1.0, jac, m, e, 1, 0.0, jte, 1
        )
        external_call["cblas_dgemm", NoneType](
            101,
            112,
            111,
            m,
            m,
            n,
            1.0,
            jac,
            m,
            jac,
            m,
            0.0,
            jtj,
            m,
        )
        return

    for i in range(m * m):
        jtj[i] = scalar[dtype](0.0)
    for i in range(m):
        jte[i] = scalar[dtype](0.0)
    for lr in range(n):
        var row = jac + lr * m
        var residual = SIMD[dtype, W](e[lr])
        var i = 0
        while i + W <= m:
            jte.store(
                i,
                jte.load[width=W](i)
                + row.load[width=W](i) * residual,
            )
            i += W
        while i < m:
            jte[i] += row[i] * e[lr]
            i += 1
        for i in range(m):
            var alpha = row[i]
            var alpha_vector = SIMD[dtype, W](alpha)
            var j = 0
            while j + W <= i + 1:
                var offset = i * m + j
                jtj.store(
                    offset,
                    jtj.load[width=W](offset)
                    + row.load[width=W](j) * alpha_vector,
                )
                j += W
            while j <= i:
                jtj[i * m + j] += row[j] * alpha
                j += 1
    for i in range(m):
        for j in range(i + 1, m):
            jtj[i * m + j] = jtj[j * m + i]


# levmar: Axb_core.c AX_EQ_B_LU (non-LAPACK path)
def solve_lu[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    rhs: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    solution: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    scratch: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
) -> Bool:
    for i in range(n * n):
        scratch[i] = matrix[i]
    for i in range(n):
        solution[i] = rhs[i]
    for col in range(n):
        var pivot = col
        var best = abs(scratch[col * n + col])
        for row in range(col + 1, n):
            var candidate = abs(scratch[row * n + col])
            if candidate > best:
                best = candidate
                pivot = row
        if best == scalar[dtype](0.0) or not finite[dtype](best):
            return False
        if pivot != col:
            for j in range(col, n):
                var temp = scratch[col * n + j]
                scratch[col * n + j] = scratch[pivot * n + j]
                scratch[pivot * n + j] = temp
            var btemp = solution[col]
            solution[col] = solution[pivot]
            solution[pivot] = btemp
        var diagonal = scratch[col * n + col]
        for row in range(col + 1, n):
            var factor = scratch[row * n + col] / diagonal
            scratch[row * n + col] = scalar[dtype](0.0)
            for j in range(col + 1, n):
                scratch[row * n + j] -= factor * scratch[col * n + j]
            solution[row] -= factor * solution[col]
    for rr in range(n):
        var i = n - 1 - rr
        var total = solution[i]
        for j in range(i + 1, n):
            total -= scratch[i * n + j] * solution[j]
        var diagonal = scratch[i * n + i]
        if diagonal == scalar[dtype](0.0):
            return False
        solution[i] = total / diagonal
    return True


# levmar: misc_core.c LEVMAR_FDIF_FORW_JAC_APPROX and LEVMAR_FDIF_CENT_JAC_APPROX
def finite_difference[
    dtype: DType
](
    func_addr: Int,
    data_addr: Int,
    p_addr: Int,
    p: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    base: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    minus: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    plus: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    minus_addr: Int,
    plus_addr: Int,
    jac: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    delta: Scalar[dtype],
    central: Bool,
    m: Int,
    n: Int,
):
    for j in range(m):
        var d = scalar[dtype](1.0e-4) * abs(p[j])
        if d < delta:
            d = delta
        var saved = p[j]
        if central:
            p[j] = saved - d
            callback[dtype](func_addr, p_addr, minus_addr, m, n, data_addr)
            p[j] = saved + d
            callback[dtype](func_addr, p_addr, plus_addr, m, n, data_addr)
            p[j] = saved
            var inverse = scalar[dtype](0.5) / d
            for i in range(n):
                jac[i * m + j] = (plus[i] - minus[i]) * inverse
        else:
            p[j] = saved + d
            callback[dtype](func_addr, p_addr, plus_addr, m, n, data_addr)
            p[j] = saved
            var inverse = scalar[dtype](1.0) / d
            for i in range(n):
                jac[i * m + j] = (plus[i] - base[i]) * inverse


def project_box[
    dtype: DType
](
    p: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    lower: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    upper: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    m: Int,
):
    for i in range(m):
        if p[i] < lower[i]:
            p[i] = lower[i]
        elif p[i] > upper[i]:
            p[i] = upper[i]


def evaluate[
    dtype: DType
](
    func_addr: Int,
    data_addr: Int,
    p_addr: Int,
    dst_addr: Int,
    dst: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    x: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    m: Int,
) -> Scalar[dtype]:
    callback[dtype](func_addr, p_addr, dst_addr, m, n, data_addr)
    var total: Scalar[dtype] = scalar[dtype](0.0)
    for i in range(n):
        dst[i] = x[i] - dst[i]
        total += dst[i] * dst[i]
    return total


# levmar: lm_core.c LEVMAR_DER and lmbc_core.c LEVMAR_BC_DER
def levmar_core[
    dtype: DType
](
    func_addr: Int,
    jac_addr: Int,
    data_addr: Int,
    p_addr: Int,
    x_addr: Int,
    lower_addr: Int,
    upper_addr: Int,
    opts_addr: Int,
    info_addr: Int,
    work_addr: Int,
    m: Int,
    n: Int,
    itmax: Int,
    bounded: Bool,
    numeric_jacobian: Bool,
) -> Int:
    if m <= 0 or n < m or itmax < 0:
        return -1
    comptime bytes = 8 if dtype == DType.float64 else 4
    var p = ptr[dtype](p_addr)
    var x = ptr[dtype](x_addr)
    var opts = ptr[dtype](opts_addr)
    var info = ptr[dtype](info_addr)
    var lower = ptr[dtype](lower_addr)
    var upper = ptr[dtype](upper_addr)
    var e_offset = 0
    var hx_offset = e_offset + n
    var jte_offset = hx_offset + n
    var jac_offset = jte_offset + m
    var jtj_offset = jac_offset + n * m
    var dp_offset = jtj_offset + m * m
    var diag_offset = dp_offset + m
    var pdp_offset = diag_offset + m
    var aux_offset = pdp_offset + m
    var solve_offset = aux_offset + 2 * n
    var e = ptr[dtype](work_addr + e_offset * bytes)
    var hx = ptr[dtype](work_addr + hx_offset * bytes)
    var jte = ptr[dtype](work_addr + jte_offset * bytes)
    var jac = ptr[dtype](work_addr + jac_offset * bytes)
    var jtj = ptr[dtype](work_addr + jtj_offset * bytes)
    var dp = ptr[dtype](work_addr + dp_offset * bytes)
    var diag = ptr[dtype](work_addr + diag_offset * bytes)
    var pdp = ptr[dtype](work_addr + pdp_offset * bytes)
    var aux_minus = ptr[dtype](work_addr + aux_offset * bytes)
    var aux_plus = ptr[dtype](work_addr + (aux_offset + n) * bytes)
    var solve_scratch = ptr[dtype](work_addr + solve_offset * bytes)
    var hx_addr = work_addr + hx_offset * bytes
    var pdp_addr = work_addr + pdp_offset * bytes
    var aux_minus_addr = work_addr + aux_offset * bytes
    var aux_plus_addr = work_addr + (aux_offset + n) * bytes

    if bounded:
        for i in range(m):
            if lower[i] > upper[i]:
                return -1
        project_box[dtype](p, lower, upper, m)

    callback[dtype](func_addr, p_addr, hx_addr, m, n, data_addr)
    var error2 = residual_norm2[dtype](e, x, hx, n)
    var initial_error2 = error2
    var nfev = 1
    var njev = 0
    var nlss = 0
    var stop = 0
    var iterations = 0
    var mu: Scalar[dtype] = scalar[dtype](0.0)
    var gradient_inf: Scalar[dtype] = scalar[dtype](0.0)
    var step2: Scalar[dtype] = scalar[dtype](0.0)
    var max_diagonal: Scalar[dtype] = real_min[dtype]()
    var nu = 2
    var tau = opts[0]
    var eps1 = opts[1]
    var eps2_sq = opts[2] * opts[2]
    var eps3 = opts[3]
    var delta = abs(opts[4])
    var central = opts[4] < scalar[dtype](0.0)
    if not finite[dtype](error2):
        stop = 7

    while iterations < itmax and stop == 0:
        if error2 <= eps3:
            stop = 6
            break
        if numeric_jacobian:
            if not central:
                callback[dtype](func_addr, p_addr, hx_addr, m, n, data_addr)
                nfev += 1
            finite_difference[dtype](
                func_addr,
                data_addr,
                p_addr,
                p,
                hx,
                aux_minus,
                aux_plus,
                aux_minus_addr,
                aux_plus_addr,
                jac,
                delta,
                central,
                m,
                n,
            )
            nfev += 2 * m if central else m
        else:
            callback[dtype](
                jac_addr,
                p_addr,
                work_addr + jac_offset * bytes,
                m,
                n,
                data_addr,
            )
        njev += 1
        normal_equations[dtype](jac, e, jtj, jte, n, m)
        gradient_inf = scalar[dtype](0.0)
        var p2: Scalar[dtype] = scalar[dtype](0.0)
        var active_ok = True
        max_diagonal = real_min[dtype]()
        for i in range(m):
            diag[i] = jtj[i * m + i]
            if diag[i] > max_diagonal:
                max_diagonal = diag[i]
            p2 += p[i] * p[i]
            if bounded and p[i] == upper[i]:
                if jte[i] <= scalar[dtype](0.0):
                    active_ok = False
            elif bounded and p[i] == lower[i]:
                if jte[i] >= scalar[dtype](0.0):
                    active_ok = False
            else:
                var magnitude = abs(jte[i])
                if magnitude > gradient_inf:
                    gradient_inf = magnitude
        if active_ok and gradient_inf <= eps1:
            step2 = scalar[dtype](0.0)
            stop = 1
            break
        if iterations == 0:
            mu = (
                scalar[dtype](0.5)
                * tau
                * error2 if bounded else tau
                * max_diagonal
            )

        var accepted = False
        var inner = 0
        while not accepted and stop == 0 and inner < 64:
            for i in range(m):
                jtj[i * m + i] = diag[i] + mu
            nlss += 1
            if solve_lu[dtype](jtj, jte, dp, solve_scratch, m):
                step2 = scalar[dtype](0.0)
                for i in range(m):
                    pdp[i] = p[i] + dp[i]
                if bounded:
                    project_box[dtype](pdp, lower, upper, m)
                for i in range(m):
                    dp[i] = pdp[i] - p[i]
                    step2 += dp[i] * dp[i]
                if step2 <= eps2_sq * p2:
                    stop = 2
                    break
                var huge_limit = (p2 + opts[2]) / (
                    machine_epsilon[dtype]() * machine_epsilon[dtype]()
                )
                if step2 >= huge_limit:
                    stop = 4
                    break
                var candidate2 = evaluate[dtype](
                    func_addr,
                    data_addr,
                    pdp_addr,
                    hx_addr,
                    hx,
                    x,
                    n,
                    m,
                )
                nfev += 1
                if not finite[dtype](candidate2):
                    stop = 7
                    break
                var predicted: Scalar[dtype] = scalar[dtype](0.0)
                for i in range(m):
                    predicted += dp[i] * (mu * dp[i] + jte[i])
                var actual = error2 - candidate2
                var good = candidate2 <= scalar[dtype](
                    0.99995
                ) * error2 if bounded else predicted > scalar[dtype](
                    0.0
                ) and actual > scalar[
                    dtype
                ](
                    0.0
                )
                if good:
                    if predicted > scalar[dtype](0.0):
                        var ratio = scalar[dtype](
                            2.0
                        ) * actual / predicted - scalar[dtype](1.0)
                        var factor = scalar[dtype](1.0) - ratio * ratio * ratio
                        if factor < scalar[dtype](1.0 / 3.0):
                            factor = scalar[dtype](1.0 / 3.0)
                        mu *= factor
                    elif bounded:
                        mu = min(mu, scalar[dtype](0.1) * candidate2)
                    nu = 2
                    for i in range(m):
                        p[i] = pdp[i]
                    for i in range(n):
                        e[i] = hx[i]
                    error2 = candidate2
                    accepted = True
                    break
                if bounded:
                    var t = scalar[dtype](0.9)
                    while t > scalar[dtype](1.0e-12):
                        for i in range(m):
                            pdp[i] = p[i] + t * dp[i]
                        project_box[dtype](pdp, lower, upper, m)
                        candidate2 = evaluate[dtype](
                            func_addr,
                            data_addr,
                            pdp_addr,
                            hx_addr,
                            hx,
                            x,
                            n,
                            m,
                        )
                        nfev += 1
                        if candidate2 < error2:
                            for i in range(m):
                                p[i] = pdp[i]
                            for i in range(n):
                                e[i] = hx[i]
                            error2 = candidate2
                            accepted = True
                            break
                        t *= scalar[dtype](0.9)
                    if accepted:
                        break
            mu *= Scalar[dtype](nu)
            if nu > 1_073_741_824:
                stop = 5
                break
            nu *= 2
            inner += 1
        if not accepted and stop == 0:
            stop = 5
        iterations += 1

    if iterations >= itmax and stop == 0:
        stop = 3
    for i in range(m):
        jtj[i * m + i] = diag[i]
    info[0] = initial_error2
    info[1] = error2
    info[2] = gradient_inf
    info[3] = step2
    info[4] = mu / max_diagonal if max_diagonal > scalar[dtype](0.0) else mu
    info[5] = Scalar[dtype](iterations)
    info[6] = Scalar[dtype](stop)
    info[7] = Scalar[dtype](nfev)
    info[8] = Scalar[dtype](njev)
    info[9] = Scalar[dtype](nlss)
    return -1 if stop == 4 or stop == 7 else iterations


# levmar: lm_core.c LEVMAR_DIF
def levmar_dif_core[
    dtype: DType
](
    func_addr: Int,
    data_addr: Int,
    p_addr: Int,
    x_addr: Int,
    opts_addr: Int,
    info_addr: Int,
    work_addr: Int,
    m: Int,
    n: Int,
    itmax: Int,
) -> Int:
    if m <= 0 or n < m or itmax < 0:
        return -1
    comptime bytes = 8 if dtype == DType.float64 else 4
    var p = ptr[dtype](p_addr)
    var x = ptr[dtype](x_addr)
    var opts = ptr[dtype](opts_addr)
    var info = ptr[dtype](info_addr)
    var hx_offset = n
    var jte_offset = 2 * n
    var jac_offset = jte_offset + m
    var jtj_offset = jac_offset + n * m
    var dp_offset = jtj_offset + m * m
    var diag_offset = dp_offset + m
    var pdp_offset = diag_offset + m
    var wrk_offset = pdp_offset + m
    var wrk2_offset = wrk_offset + n
    var solve_offset = wrk2_offset + n
    var e = ptr[dtype](work_addr)
    var hx = ptr[dtype](work_addr + hx_offset * bytes)
    var jte = ptr[dtype](work_addr + jte_offset * bytes)
    var jac = ptr[dtype](work_addr + jac_offset * bytes)
    var jtj = ptr[dtype](work_addr + jtj_offset * bytes)
    var dp = ptr[dtype](work_addr + dp_offset * bytes)
    var diag = ptr[dtype](work_addr + diag_offset * bytes)
    var pdp = ptr[dtype](work_addr + pdp_offset * bytes)
    var wrk = ptr[dtype](work_addr + wrk_offset * bytes)
    var wrk2 = ptr[dtype](work_addr + wrk2_offset * bytes)
    var solve_scratch = ptr[dtype](work_addr + solve_offset * bytes)
    var hx_addr = work_addr + hx_offset * bytes
    var pdp_addr = work_addr + pdp_offset * bytes
    var wrk_addr = work_addr + wrk_offset * bytes
    var wrk2_addr = work_addr + wrk2_offset * bytes

    callback[dtype](func_addr, p_addr, hx_addr, m, n, data_addr)
    var error2 = residual_norm2[dtype](e, x, hx, n)
    var initial_error2 = error2
    var tau = opts[0]
    var eps1 = opts[1]
    var eps2_sq = opts[2] * opts[2]
    var eps3 = opts[3]
    var delta = abs(opts[4])
    var central = opts[4] < scalar[dtype](0.0)
    var mu: Scalar[dtype] = scalar[dtype](0.0)
    var gradient_inf: Scalar[dtype] = scalar[dtype](0.0)
    var step2: Scalar[dtype] = scalar[dtype](
        1.7976931348623157e308
    ) if dtype == DType.float64 else scalar[dtype](3.4028234663852886e38)
    var p2: Scalar[dtype] = scalar[dtype](0.0)
    var max_diagonal: Scalar[dtype] = real_min[dtype]()
    var stop = 7 if not finite[dtype](error2) else 0
    var nfev = 1
    var njap = 0
    var nlss = 0
    var nu = 20
    var updated_jacobian = 0
    var updated_parameters = True
    var new_jacobian = False
    var refresh_after = max(m, 10)
    var iterations = 0

    while iterations < itmax and stop == 0:
        if error2 <= eps3:
            stop = 6
            break
        if updated_parameters and nu > 16 or updated_jacobian == refresh_after:
            finite_difference[dtype](
                func_addr,
                data_addr,
                p_addr,
                p,
                hx,
                wrk,
                wrk2,
                wrk_addr,
                wrk2_addr,
                jac,
                delta,
                central,
                m,
                n,
            )
            nfev += 2 * m if central else m
            njap += 1
            nu = 2
            updated_jacobian = 0
            updated_parameters = False
            new_jacobian = True
        if new_jacobian:
            new_jacobian = False
            normal_equations[dtype](jac, e, jtj, jte, n, m)
            gradient_inf = scalar[dtype](0.0)
            p2 = scalar[dtype](0.0)
            max_diagonal = real_min[dtype]()
            for i in range(m):
                var magnitude = abs(jte[i])
                if magnitude > gradient_inf:
                    gradient_inf = magnitude
                diag[i] = jtj[i * m + i]
                if diag[i] > max_diagonal:
                    max_diagonal = diag[i]
                p2 += p[i] * p[i]
        if gradient_inf <= eps1:
            step2 = scalar[dtype](0.0)
            stop = 1
            break
        if iterations == 0:
            mu = tau * max_diagonal
        for i in range(m):
            jtj[i * m + i] = diag[i] + mu
        nlss += 1
        if solve_lu[dtype](jtj, jte, dp, solve_scratch, m):
            step2 = scalar[dtype](0.0)
            for i in range(m):
                pdp[i] = p[i] + dp[i]
                step2 += dp[i] * dp[i]
            if step2 <= eps2_sq * p2:
                stop = 2
            elif step2 >= (p2 + opts[2]) / (
                machine_epsilon[dtype]() * machine_epsilon[dtype]()
            ):
                stop = 4
            else:
                callback[dtype](func_addr, pdp_addr, wrk_addr, m, n, data_addr)
                nfev += 1
                var candidate2: Scalar[dtype] = scalar[dtype](0.0)
                for i in range(n):
                    wrk2[i] = x[i] - wrk[i]
                    candidate2 += wrk2[i] * wrk2[i]
                if not finite[dtype](candidate2):
                    stop = 7
                else:
                    var actual = error2 - candidate2
                    if updated_parameters or actual > scalar[dtype](0.0):
                        for i in range(n):
                            var product: Scalar[dtype] = scalar[dtype](0.0)
                            for j in range(m):
                                product += jac[i * m + j] * dp[j]
                            var correction = (wrk[i] - hx[i] - product) / step2
                            for j in range(m):
                                jac[i * m + j] += correction * dp[j]
                        updated_jacobian += 1
                        new_jacobian = True
                    var predicted: Scalar[dtype] = scalar[dtype](0.0)
                    for i in range(m):
                        predicted += dp[i] * (mu * dp[i] + jte[i])
                    if predicted > scalar[dtype](0.0) and actual > scalar[
                        dtype
                    ](0.0):
                        var ratio = scalar[dtype](
                            2.0
                        ) * actual / predicted - scalar[dtype](1.0)
                        var factor = scalar[dtype](1.0) - ratio * ratio * ratio
                        if factor < scalar[dtype](1.0 / 3.0):
                            factor = scalar[dtype](1.0 / 3.0)
                        mu *= factor
                        nu = 2
                        for i in range(m):
                            p[i] = pdp[i]
                        for i in range(n):
                            e[i] = wrk2[i]
                            hx[i] = wrk[i]
                        error2 = candidate2
                        updated_parameters = True
                        iterations += 1
                        continue
        if stop == 0:
            mu *= Scalar[dtype](nu)
            if nu > 1_073_741_824:
                stop = 5
            else:
                nu *= 2
        for i in range(m):
            jtj[i * m + i] = diag[i]
        iterations += 1

    if iterations >= itmax and stop == 0:
        stop = 3
    for i in range(m):
        jtj[i * m + i] = diag[i]
    info[0] = initial_error2
    info[1] = error2
    info[2] = gradient_inf
    info[3] = step2
    info[4] = mu / max_diagonal if max_diagonal > scalar[dtype](0.0) else mu
    info[5] = Scalar[dtype](iterations)
    info[6] = Scalar[dtype](stop)
    info[7] = Scalar[dtype](nfev)
    info[8] = Scalar[dtype](njap)
    info[9] = Scalar[dtype](nlss)
    return -1 if stop == 4 or stop == 7 else iterations


# levmar: lmlec_core.c LMLEC_FUNC
def affine_map[
    dtype: DType
](
    free: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    particular: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    basis: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    dst: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    m: Int,
    free_m: Int,
):
    for i in range(m):
        var total = particular[i]
        for j in range(free_m):
            total += basis[i * free_m + j] * free[j]
        dst[i] = total


# levmar: lmlec_core.c LMLEC_JACF
def jacobian_times_basis[
    dtype: DType
](
    jac: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    basis: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    dst: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    m: Int,
    free_m: Int,
):
    comptime W = simd_width_of[dtype]()
    if n * m * free_m >= 1_000_000:
        if dtype == DType.float64:
            external_call["cblas_dgemm", NoneType](
                101,
                111,
                111,
                n,
                free_m,
                m,
                1.0,
                jac,
                m,
                basis,
                free_m,
                0.0,
                dst,
                free_m,
            )
        else:
            external_call["cblas_sgemm", NoneType](
                101,
                111,
                111,
                n,
                free_m,
                m,
                Float32(1.0),
                jac,
                m,
                basis,
                free_m,
                Float32(0.0),
                dst,
                free_m,
            )
        return

    for i in range(n):
        var j = 0
        while j + W <= free_m:
            var total = SIMD[dtype, W](scalar[dtype](0.0))
            for k in range(m):
                total += (
                    basis.load[width=W](k * free_m + j)
                    * SIMD[dtype, W](jac[i * m + k])
                )
            dst.store(i * free_m + j, total)
            j += W
        while j < free_m:
            var total: Scalar[dtype] = scalar[dtype](0.0)
            for k in range(m):
                total += jac[i * m + k] * basis[k * free_m + j]
            dst[i * free_m + j] = total
            j += 1


@export("mlm_der_f64")
def mlm_der_f64(
    func: Int,
    jac: Int,
    data: Int,
    p: Int,
    x: Int,
    opts: Int,
    info: Int,
    work: Int,
    m: Int,
    n: Int,
    itmax: Int,
) abi("C") -> Int:
    return levmar_core[DType.float64](
        func, jac, data, p, x, p, p, opts, info, work, m, n, itmax, False, False
    )


@export("mlm_dif_f64")
def mlm_dif_f64(
    func: Int,
    data: Int,
    p: Int,
    x: Int,
    opts: Int,
    info: Int,
    work: Int,
    m: Int,
    n: Int,
    itmax: Int,
) abi("C") -> Int:
    return levmar_dif_core[DType.float64](
        func, data, p, x, opts, info, work, m, n, itmax
    )


@export("mlm_bc_der_f64")
def mlm_bc_der_f64(
    func: Int,
    jac: Int,
    data: Int,
    p: Int,
    x: Int,
    lower: Int,
    upper: Int,
    opts: Int,
    info: Int,
    work: Int,
    m: Int,
    n: Int,
    itmax: Int,
) abi("C") -> Int:
    return levmar_core[DType.float64](
        func,
        jac,
        data,
        p,
        x,
        lower,
        upper,
        opts,
        info,
        work,
        m,
        n,
        itmax,
        True,
        False,
    )


@export("mlm_bc_dif_f64")
def mlm_bc_dif_f64(
    func: Int,
    data: Int,
    p: Int,
    x: Int,
    lower: Int,
    upper: Int,
    opts: Int,
    info: Int,
    work: Int,
    m: Int,
    n: Int,
    itmax: Int,
) abi("C") -> Int:
    return levmar_core[DType.float64](
        func,
        0,
        data,
        p,
        x,
        lower,
        upper,
        opts,
        info,
        work,
        m,
        n,
        itmax,
        True,
        True,
    )


@export("mlm_der_f32")
def mlm_der_f32(
    func: Int,
    jac: Int,
    data: Int,
    p: Int,
    x: Int,
    opts: Int,
    info: Int,
    work: Int,
    m: Int,
    n: Int,
    itmax: Int,
) abi("C") -> Int:
    return levmar_core[DType.float32](
        func, jac, data, p, x, p, p, opts, info, work, m, n, itmax, False, False
    )


@export("mlm_dif_f32")
def mlm_dif_f32(
    func: Int,
    data: Int,
    p: Int,
    x: Int,
    opts: Int,
    info: Int,
    work: Int,
    m: Int,
    n: Int,
    itmax: Int,
) abi("C") -> Int:
    return levmar_dif_core[DType.float32](
        func, data, p, x, opts, info, work, m, n, itmax
    )


@export("mlm_bc_der_f32")
def mlm_bc_der_f32(
    func: Int,
    jac: Int,
    data: Int,
    p: Int,
    x: Int,
    lower: Int,
    upper: Int,
    opts: Int,
    info: Int,
    work: Int,
    m: Int,
    n: Int,
    itmax: Int,
) abi("C") -> Int:
    return levmar_core[DType.float32](
        func,
        jac,
        data,
        p,
        x,
        lower,
        upper,
        opts,
        info,
        work,
        m,
        n,
        itmax,
        True,
        False,
    )


@export("mlm_bc_dif_f32")
def mlm_bc_dif_f32(
    func: Int,
    data: Int,
    p: Int,
    x: Int,
    lower: Int,
    upper: Int,
    opts: Int,
    info: Int,
    work: Int,
    m: Int,
    n: Int,
    itmax: Int,
) abi("C") -> Int:
    return levmar_core[DType.float32](
        func,
        0,
        data,
        p,
        x,
        lower,
        upper,
        opts,
        info,
        work,
        m,
        n,
        itmax,
        True,
        True,
    )


@export("mlm_affine_f64")
def mlm_affine_f64(
    free: Int, particular: Int, basis: Int, dst: Int, m: Int, free_m: Int
) abi("C"):
    affine_map[DType.float64](
        ptr[DType.float64](free),
        ptr[DType.float64](particular),
        ptr[DType.float64](basis),
        ptr[DType.float64](dst),
        m,
        free_m,
    )


@export("mlm_jac_basis_f64")
def mlm_jac_basis_f64(
    jac: Int, basis: Int, dst: Int, n: Int, m: Int, free_m: Int
) abi("C"):
    jacobian_times_basis[DType.float64](
        ptr[DType.float64](jac),
        ptr[DType.float64](basis),
        ptr[DType.float64](dst),
        n,
        m,
        free_m,
    )


@export("mlm_affine_f32")
def mlm_affine_f32(
    free: Int, particular: Int, basis: Int, dst: Int, m: Int, free_m: Int
) abi("C"):
    affine_map[DType.float32](
        ptr[DType.float32](free),
        ptr[DType.float32](particular),
        ptr[DType.float32](basis),
        ptr[DType.float32](dst),
        m,
        free_m,
    )


@export("mlm_jac_basis_f32")
def mlm_jac_basis_f32(
    jac: Int, basis: Int, dst: Int, n: Int, m: Int, free_m: Int
) abi("C"):
    jacobian_times_basis[DType.float32](
        ptr[DType.float32](jac),
        ptr[DType.float32](basis),
        ptr[DType.float32](dst),
        n,
        m,
        free_m,
    )
