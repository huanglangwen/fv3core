#!/usr/bin/env python3
from types import SimpleNamespace

import click

import fv3core
import fv3core._config as spec
import fv3core.testing
import fv3core.utils.global_config as global_config
import serialbox
from fv3core.stencils.dyn_core import AcousticDynamics
from gt4py.storage import from_array
from time import time
import cupy

def set_up_namelist(data_directory: str) -> None:
    spec.set_namelist(data_directory + "/input.nml")


def initialize_serializer(data_directory: str, rank: int = 0) -> serialbox.Serializer:
    return serialbox.Serializer(
        serialbox.OpenModeKind.Read,
        data_directory,
        "Generator_rank" + str(rank),
    )


def read_grid(
    serializer: serialbox.Serializer, rank: int = 0
) -> fv3core.testing.TranslateGrid:
    grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
    grid_data = {}
    grid_fields = serializer.fields_at_savepoint(grid_savepoint)
    for field in grid_fields:
        grid_data[field] = serializer.read(field, grid_savepoint)
        if len(grid_data[field].flatten()) == 1:
            grid_data[field] = grid_data[field][0]
    return fv3core.testing.TranslateGrid(grid_data, rank).python_grid()


def initialize_fv3core(backend: str, do_halo_updates: bool) -> None:
    fv3core.set_backend(backend)
    fv3core.set_rebuild(False)
    fv3core.set_validate_args(False)
    global_config.set_do_halo_exchange(do_halo_updates)


def read_input_data(grid, serializer):
    driver_object = fv3core.testing.TranslateDynCore([grid])
    savepoint_in = serializer.get_savepoint("DynCore-In")[0]
    return driver_object.collect_input_data(serializer, savepoint_in)


def get_state_from_input(grid, input_data):
    driver_object = fv3core.testing.TranslateDynCore([grid])
    driver_object._base.make_storage_data_input_vars(input_data)

    inputs = driver_object.inputs
    for name, properties in inputs.items():
        grid.quantity_dict_update(
            input_data, name, dims=properties["dims"], units=properties["units"]
        )

    statevars = SimpleNamespace(**input_data)
    return {"state": statevars}


@click.command()
@click.argument("data_directory", required=True, nargs=1)
@click.argument("time_steps", required=False, default="1")
@click.argument("backend", required=False, default="gtc:gt:cpu_ifirst")
@click.option("--halo_update/--no-halo_update", default=False)
def driver(
    data_directory: str,
    time_steps: str,
    backend: str,
    halo_update: bool,
):
    set_up_namelist(data_directory)
    serializer = initialize_serializer(data_directory)
    initialize_fv3core(backend, halo_update)
    if backend == "gtc:cuda":
        from gt4py.gtgraph import AsyncContext
        async_context = AsyncContext(50, name="acoustics", graph_record=False, concurrent=True, blocking=False, region_analysis=True)
        global_config.set_async_context(async_context)
    grid = read_grid(serializer)
    spec.set_grid(grid)

    input_data = read_input_data(grid, serializer)

    acoutstics_object = AcousticDynamics(
        None,
        spec.namelist,
        from_array(input_data["ak"], backend, (0,0,0), mask=(False, False, True)),
        from_array(input_data["bk"], backend, (0,0,0), mask=(False, False, True)),
        from_array(input_data["pfull"], backend, (0,0,0), mask=(False, False, True)),
        from_array(input_data["phis"], backend, (0,0,0)),
    )

    state = get_state_from_input(grid, input_data)
    state["state"].__dict__.update(acoutstics_object._temporaries)

    # Testing dace infrastucture
    #output_field = acoutstics_object.dace_dummy(input_data["omga"])
    #output_field = acoutstics_object.dace_dummy(state["state"].omga)
    #print(output_field)

    # @Linus: make this call a dace program
    
    acoutstics_object(state["state"], insert_temporaries=False)
    async_context.wait()
    t0 = time()
    with cupy.cuda.profile():
        for _ in range(int(time_steps)-1):
            acoutstics_object(state["state"], insert_temporaries=False)
        if backend == "gtc:cuda":
            async_context.wait()
            #async_context.graph_save()
    t1 = time()
    print(f"Elapsed time: {t1 - t0} s")


if __name__ == "__main__":
    driver()

