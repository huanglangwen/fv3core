from typing import Optional, Tuple

import gt4py
import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    asin,
    computation,
    cos,
    horizontal,
    interval,
    region,
    sin,
    sqrt,
)
from fv3core.utils.grid import axis_offsets
import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.typing import FloatField, FloatFieldI, FloatFieldIJ


# compact 4-pt cubic interpolation
c1 = 2.0 / 3.0
c2 = -1.0 / 6.0
d1 = 0.375
d2 = -1.0 / 24.0
# PPM volume mean form
b1 = 7.0 / 12.0
b2 = -1.0 / 12.0
# 4-pt Lagrange interpolation
a1 = 9.0 / 16.0
a2 = -1.0 / 16.0


@gtscript.function
def ppm_volume_mean_x(qin: FloatField):
    return b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)


@gtscript.function
def ppm_volume_mean_y(qin: FloatField):
    return b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)


@gtscript.function
def lagrange_y(qx: FloatField):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@gtscript.function
def lagrange_x(qy: FloatField):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


@gtscript.function
def cubic_interpolation_south(qx: FloatField, qout: FloatField, qxx: FloatField):
    return c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx[0, 1, 0])


@gtscript.function
def cubic_interpolation_north(qx: FloatField, qout: FloatField, qxx: FloatField):
    return c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx[0, -1, 0])


@gtscript.function
def cubic_interpolation_west(qy: FloatField, qout: FloatField, qyy: FloatField):
    return c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy[1, 0, 0])


@gtscript.function
def cubic_interpolation_east(qy: FloatField, qout: FloatField, qyy: FloatField):
    return c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy[-1, 0, 0])


@gtscript.function
def qout_x_edge(edge_w: FloatFieldIJ, q2: FloatField):
    return edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


@gtscript.function
def qout_y_edge(edge_s: FloatFieldI, q1: FloatField):
    return edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1


@gtscript.function
def qx_edge_west(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[1, 0] / dxa
    g_ou = dxa[-2, 0] / dxa[-1, 0]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qx_edge_west2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatFieldIJ):
    g_in = dxa / dxa[-1, 0]
    return (
        3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qx[-1, 0, 0] + qx[1, 0, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qx_edge_east(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[-2, 0] / dxa[-1, 0]
    g_ou = dxa[1, 0] / dxa
    return 0.5 * (
        ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qx_edge_east2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    g_in = dxa[-1, 0] / dxa
    return (
        3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qx[1, 0, 0] + qx[-1, 0, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qy_edge_south(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, 1] / dya
    g_ou = dya[0, -2] / dya[0, -1]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qy_edge_south2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    g_in = dya / dya[0, -1]
    return (
        3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy[0, -1, 0] + qy[0, 1, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qy_edge_north(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, -2] / dya[0, -1]
    g_ou = dya[0, 1] / dya
    return 0.5 * (
        ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qy_edge_north2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    g_in = dya[0, -1] / dya
    return (
        3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy[0, 1, 0] + qy[0, -1, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def great_circle_dist_noradius(p1a, p1b, p2a, p2b):
    tb = sin((p1b - p2b) / 2.0) ** 2.0
    ta = sin((p1a - p2a) / 2.0) ** 2.0
    return asin(sqrt(tb + cos(p1b) * cos(p2b) * ta)) * 2.0


@gtscript.function
def extrap_corner(
    p0a,
    p0b,
    p1a,
    p1b,
    p2a,
    p2b,
    qa,
    qb,
):
    x1 = great_circle_dist_noradius(p1a, p1b, p0a, p0b)
    x2 = great_circle_dist_noradius(p2a, p2b, p0a, p0b)
    return qa + x1 / (x2 - x1) * (qa - qb)


def _sw_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):

    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )

        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


def _nw_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


def _ne_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


def _se_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


def _a2b_ord4_stencil(
    qin: FloatField,
    qout: FloatField,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    edge_n: FloatFieldI,
    edge_s: FloatFieldI,
    edge_e: FloatFieldIJ,
    edge_w: FloatFieldIJ,
    q1: FloatField,
    q2: FloatField,
    qx: FloatField,
    qy: FloatField,
    qxx: FloatField,
    qyy: FloatField,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # {
        with horizontal(
            region[i_start - 1 : i_start + 1, :], region[i_end : i_end + 2, :]
        ):
            q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)

        with horizontal(
            region[:, j_start - 1 : j_start + 1], region[:, j_end : j_end + 2]
        ):
            q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)

        with horizontal(region[i_start, j_start + 1 : j_end + 1]):
            # qout = qout_x_edge(edge_w, q2)
            qout = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2
        with horizontal(region[i_end + 1, j_start + 1 : j_end + 1]):
            # qout = qout_x_edge(edge_e, q2)
            qout = edge_e * q2[0, -1, 0] + (1.0 - edge_e) * q2
        with horizontal(region[i_start + 1 : i_end + 1, j_start]):
            # qout = qout_y_edge(edge_s, q1)
            qout = edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1
        with horizontal(region[i_start + 1 : i_end + 1, j_end + 1]):
            # qout = qout_y_edge(edge_n, q1)
            qout = edge_n * q1[-1, 0, 0] + (1.0 - edge_n) * q1

        # compute_qx
        qx = ppm_volume_mean_x(qin)
        g_in = 0.0
        g_ou = 0.0
        with horizontal(region[i_start, :]):
            # qx = qx_edge_west(qin, dxa)
            g_in = dxa[1, 0] / dxa
            g_ou = dxa[-2, 0] / dxa[-1, 0]
            qx = 0.5 * (
                ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
                + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
            )
        with horizontal(region[i_start + 1, :]):
            # qx = qx_edge_west2(qin, dxa, qx)
            g_in = dxa / dxa[-1, 0]
            qx = (
                3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qx[-1, 0, 0] + qx[1, 0, 0])
            ) / (2.0 + 2.0 * g_in)
        with horizontal(region[i_end + 1, :]):
            # qx = qx_edge_east(qin, dxa)
            g_in = dxa[-2, 0] / dxa[-1, 0]
            g_ou = dxa[1, 0] / dxa
            qx = 0.5 * (
                ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
                + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)
            )
        with horizontal(region[i_end, :]):
            # qx = qx_edge_east2(qin, dxa, qx)
            g_in = dxa[-1, 0] / dxa
            qx = (
                3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qx[1, 0, 0] + qx[-1, 0, 0])
            ) / (2.0 + 2.0 * g_in)
        # compute_qy
        qy = ppm_volume_mean_y(qin)
        with horizontal(region[:, j_start]):
            # qy = qy_edge_south(qin, dya)
            g_in = dya[0, 1] / dya
            g_ou = dya[0, -2] / dya[0, -1]
            qy = 0.5 * (
                ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
                + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
            )

        with horizontal(region[:, j_start + 1]):
            # qy = qy_edge_south2(qin, dya, qy)
            g_in = dya / dya[0, -1]
            qy = (
                3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy[0, -1, 0] + qy[0, 1, 0])
            ) / (2.0 + 2.0 * g_in)

        with horizontal(region[:, j_end + 1]):
            # qy = qy_edge_north(qin, dya)
            g_in = dya[0, -2] / dya[0, -1]
            g_ou = dya[0, 1] / dya
            qy = 0.5 * (
                ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
                + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
            )

        with horizontal(region[:, j_end]):
            # qy = qy_edge_north2(qin, dya, qy)
            g_in = dya[0, -1] / dya
            qy = (
                3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy[0, 1, 0] + qy[0, -1, 0])
            ) / (2.0 + 2.0 * g_in)
        # compute_qxx
        qxx = lagrange_y(qx)
        with horizontal(region[:, j_start + 1]):
            # qxx = cubic_interpolation_south(qx, qout, qxx)
            qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx[0, 1, 0])
        with horizontal(region[:, j_end]):
            # qxx = cubic_interpolation_north(qx, qout, qxx)
            qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx[0, -1, 0])
        # compute_qyy
        qyy = lagrange_x(qy)
        with horizontal(region[i_start + 1, :]):
            # qyy = cubic_interpolation_west(qy, qout, qyy)
            qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy[1, 0, 0])
        with horizontal(region[i_end, :]):
            # qyy = cubic_interpolation_east(qy, qout, qyy)
            qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy[-1, 0, 0])

        with horizontal(region[i_start + 1 : i_end + 1, j_start + 1 : j_end + 1]):
            qout = 0.5 * (qxx + qyy)
        # }

        if replace:
            qin = qout

# TODO                                                                                                        
# within regions, the edge_w and edge_w variables that are singleton in the                                    
# I dimension error, workaround is repeating the data, but the longterm                                        
# fix should happen in regions                                                                                 
def _j_storage_repeat_over_i(grid_array: gt4py.storage.Storage, shape: Tuple[int, int] ):
    dup = utils.repeat(grid_array, shape[1], axis=0)
    return utils.make_storage_data(dup, shape, (0, 0))

class AGrid2BGridFourthOrder:
    """
    Fortran name is a2b_ord4, test module is A2B_Ord4
    """
    def __init__(self, namelist):
        assert namelist.grid_type < 3
        self.grid = spec.grid
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        full_origin= self.grid.full_origin()
        edge_e = _j_storage_repeat_over_i(self.grid.edge_e, shape[0:2])
        edge_w = _j_storage_repeat_over_i(self.grid.edge_w, shape[0:2])
        self._tmp_q1 = utils.make_storage_from_shape(shape, full_origin)
        self._tmp_q2 = utils.make_storage_from_shape(shape, full_origin)
        self._tmp_qx = utils.make_storage_from_shape(shape, full_origin)
        self._tmp_qy = utils.make_storage_from_shape(shape, full_origin)
        self._tmp_qxx = utils.make_storage_from_shape(shape, full_origin)
        self._tmp_qyy = utils.make_storage_from_shape(shape, full_origin)
        ax_offsets = axis_offsets(
            self.grid, self.grid.full_origin(), self.grid.domain_shape_full()
        )
        stencil_kwargs = {
            "backend": global_config.get_backend(),
            "rebuild": global_config.get_rebuild(),
            "externals": ax_offsets,
        }
        self.stencil_runtime_args = {"validate_args": global_config.get_validate_args()}
        stencil_wrapper = gtscript.stencil(**stencil_kwargs)
        self.stencil=stencil_wrapper(_a2b_ord4_stencil)
        self._sw_corner = stencil_wrapper(_sw_corner)
        self._se_corner = stencil_wrapper(_se_corner)
        self._nw_corner = stencil_wrapper(_nw_corner)
        self._ne_corner = stencil_wrapper(_ne_corner)
        def __call__(
                self,
                qin: FloatField,
                qout: FloatField,
                kstart: int = 0,
                nk: Optional[int] = None,
                replace: bool = False,
        ):
            """
            Transfers qin from A-grid to B-grid.

            Args:
            qin: Input on A-grid (in)
            qout: Output on B-grid (out)
            kstart: Starting level
            nk: Number of levels
            replace: If True, sets `qout = qin` as the last step
            """
            grid = self.grid
            if nk is None:
                nk = grid.npz - kstart
            corner_domain = (1, 1, nk)
            self._sw_corner(
                qin,
                qout,
                grid.agrid1,
                grid.agrid2,
                grid.bgrid1,
                grid.bgrid2,
                origin=(grid.is_, grid.js, kstart),
                domain=corner_domain,
                **self.stencil_runtime_args
            )

            self._nw_corner(
                qin,
                qout,
                grid.agrid1,
                grid.agrid2,
                grid.bgrid1,
                grid.bgrid2,
                origin=(grid.ie + 1, grid.js, kstart),
                domain=corner_domain,**self.stencil_runtime_args
            )
            self._ne_corner(
                qin,
                qout,
                grid.agrid1,
                grid.agrid2,
                grid.bgrid1,
                grid.bgrid2,
                origin=(grid.ie + 1, grid.je + 1, kstart),
                domain=corner_domain,**self.stencil_runtime_args
            )
            self._se_corner(
                qin,
                qout,
                grid.agrid1,
                grid.agrid2,
                grid.bgrid1,
                grid.bgrid2,
                origin=(grid.is_, grid.je + 1, kstart),
                domain=corner_domain,**self.stencil_runtime_args
            )

            self.stencil(
                qin,
                qout,
                grid.dxa,
                grid.dya,
                grid.edge_n,
                grid.edge_s,
                edge_e,
                edge_w,
                q1,
                q2,
                qx,
                qy,
                qxx,
                qyy,
                replace,
                origin=(grid.is_, grid.js, kstart),
                domain=(grid.nic + 1, grid.njc + 1, nk),**self.stencil_runtime_args
            )
            
