"""Autogenerated MCP server for Pipefunc pipelines."""

import json
import operator
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import fastmcp
import pydantic
from rich.console import Console

from pipefunc._pipeline._autodoc import PipelineDocumentation, format_pipeline_docs
from pipefunc._pipeline._base import Pipeline
from pipefunc._utils import requires
from pipefunc.map import load_all_outputs, load_outputs
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run_eager_async import AsyncMap
from pipefunc.map._run_info import RunInfo
from pipefunc.map._shapes import shape_is_resolved
from pipefunc.map._storage_array._base import StorageBase
from pipefunc.map._storage_array._file import FileArray


@dataclass
class JobInfo:
    """Information about an async pipeline job."""

    runner: AsyncMap
    started_at: datetime
    run_folder: str
    status: Literal["running", "completed", "cancelled"]
    pipeline_name: str


# Global job registry to track async pipeline executions (UUID -> JobInfo)
job_registry: dict[str, JobInfo] = {}

_DEFAULT_PIPELINE_NAME = "Unnamed Pipeline"
_DEFAULT_PIPELINE_DESCRIPTION = "No description provided."

_PIPEFUNC_INSTRUCTIONS = """\
This MCP server executes pipefunc computational pipelines.
pipefunc creates function pipelines as DAGs where functions are automatically connected based on input/output dependencies.
See https://pipefunc.readthedocs.io/en/latest/ and https://github.com/pipefunc/pipefunc for more information.

<general>
## CORE CONCEPTS:
- Pipeline: A sequence of interconnected Python functions forming a computational workflow. Dependencies are handled automatically.
- MapSpec: A string (e.g., "x[i] -> y[i]") that defines how arrays are mapped between functions, enabling powerful, parallel parameter sweeps.

## EXECUTION MODES:
Two execution modes are available:

1. Synchronous (execute_pipeline_sync):
   - Blocks until completion and returns results immediately
   - Ideal for single calculations, small datasets, and interactive use.

2. Asynchronous (execute_pipeline_async):
   - Returns a `job_id` immediately for tracking background execution.
   - Ideal for large datasets, long-running computations, and parameter sweeps.
   - Workflow: Start job -> Get `job_id` -> Check status.
   - IMPORTANT: Always IMMEDIATELY check the job status with check_job_status after starting a new job!

## JOB MANAGEMENT (for Async):
- `check_job_status(job_id)`: Monitor progress and get results when complete.
- `list_jobs()`: See all running/completed jobs.
- `cancel_job(job_id)`: Stop a running job.

## EXECUTION PARAMETERS:
- inputs: Dictionary with parameter values (single values or arrays)
- parallel: Boolean (default true) - enables parallel execution
- run_folder: Optional string - directory to save intermediate results

## MAPSPEC REFERENCE:
`MapSpec` is the key to unlocking parallel execution and parameter sweeps.

Basic Syntax:
`"input1[index1], input2[index2] -> output1[index3]"`

Index Rules (How to control sweeps):
- Different Indices (`a[i]`, `b[j]`): Creates a cross-product. The pipeline runs for every combination of elements from `a` and `b`.
- Same Index (`x[i]`, `y[i]`): Zips the inputs. The pipeline pairs elements `x[0]` with `y[0]`, `x[1]` with `y[1]`, etc. The arrays must have the same length.
- No Index: A single value is treated as a constant and used in every computation of the sweep.

Common `MapSpec` Patterns & Examples:
- `"x[i] -> y[i]"`: Element-wise. Processes each element of `x` independently.
  - *Example Input*: `{"x": [1, 2, 3]}`
- `"x[i], y[i] -> z[i]"`: Zipped. Pairs elements from `x` and `y`.
  - *Example Input*: `{"x": [1, 2], "y": [10, 20]}`
- `"a[i], b[j] -> c[i, j]"`: Cross-product. Combines every element of `a` with every element of `b`.
  - *Example Input*: `{"a": [1, 2], "b": [10, 20]}` results in 4 runs.
- `"x[i, :] -> y[i]"`: Reduction. Aggregates data across a dimension (the `:`). This is typically an internal pipeline step.
- `"... -> x[i]"`: Dynamic Axis Generation. A function that generates an array from a scalar input. This is typically an internal pipeline step.

## OUTPUT FORMAT:
Returns dictionary with all pipeline outputs. Each output contains:
- "output": Computed result (converted to JSON-compatible format)
- "shape": Array dimensions (if applicable)

## JOB MANAGEMENT:
- check_job_status: Monitor progress and get results when complete
- list_jobs: See all running/completed jobs
- cancel_job: Stop a running job

</general>
"""

_PIPELINE_EXECUTE_DESCRIPTION_TEMPLATE = """\
Executes the '{pipeline_name}' pipeline.

## 1. Pipeline Purpose

{pipeline_description}

## 2. Pipeline Information

{pipeline_info}

## 3. How to Provide Inputs

{input_format_section}

## 4. How Arrays are Processed (`MapSpec` rules)

{mapspec_section}

## 5. Detailed Parameter and Output Reference

{documentation}
"""


_PIPELINE_ASYNC_EXECUTE_DESCRIPTION_EXTRA = """\
This tool returns a job ID and run folder.

Use the job ID to:
- check_job_status: Monitor progress and get results when complete
- list_jobs: See all running/completed jobs
- cancel_job: Stop a running job

IMPORTANT:
- Whenever starting a new job, ALWAYS immediately check the job status with check_job_status.
"""


def _get_pipeline_documentation(pipeline: Pipeline) -> str:
    """Generate formatted pipeline documentation tables using Rich."""
    requires("rich", "griffe", reason="mcp", extras="autodoc")

    doc = PipelineDocumentation.from_pipeline(pipeline)
    tables = format_pipeline_docs(doc, print_table=False, emojis=False)
    assert tables is not None

    console = Console(force_terminal=False)
    with console.capture() as capture:
        for table in tables:
            console.print(table, "\n")
    return capture.get()


def _get_pipeline_info_summary(pipeline_name: str, pipeline: Pipeline) -> str:
    """Generate a summary of pipeline information."""
    info = pipeline.info()
    assert info is not None

    def _format(key: str) -> str:
        return ", ".join(info[key]) if info[key] else "None"

    lines = [
        f"Pipeline Name: {pipeline_name}",
        f"Required Inputs: {_format('required_inputs')}",
        f"Optional Inputs: {_format('optional_inputs')}",
        f"Outputs: {_format('outputs')}",
        f"Intermediate Outputs: {_format('intermediate_outputs')}",
    ]
    return "\n".join(lines)


def _get_input_format_section(pipeline: Pipeline) -> str:
    """Dynamically generate the input format and usage section."""
    root_args = set(pipeline.root_args())
    mapspec_inputs = pipeline.mapspec_names
    required_array_inputs = sorted(root_args & mapspec_inputs)

    example_dict: dict[str, Any] = {}
    type_defaults = {int: 0, float: 0.0, str: "example", bool: False}
    param_ann = pipeline.parameter_annotations

    for param in sorted(root_args):
        param_type = param_ann.get(param)
        if isinstance(param_type, str):  # pragma: no cover
            param_type = None

        if param in required_array_inputs:
            item_type_default = type_defaults.get(param_type, "item")  # type: ignore[arg-type]
            example_dict[param] = [item_type_default, item_type_default]
        else:
            example_dict[param] = type_defaults.get(param_type, "value")  # type: ignore[arg-type]

    example_str = json.dumps(example_dict)

    lines = []
    if required_array_inputs:
        scalar_inputs = sorted(root_args - mapspec_inputs)
        lines.append("This pipeline is designed for array-based parameter sweeps.")

        lines.append("\nRequired Array Inputs:")
        lines.append("The following parameters MUST be provided as arrays (or nested lists):")
        lines.extend([f"- `{name}`" for name in required_array_inputs])

        if scalar_inputs:
            lines.append("\nConstant Inputs:")
            lines.append("The following parameters are provided as single, constant values:")
            lines.extend([f"- `{name}`" for name in scalar_inputs])
    else:
        # If no root arguments are in a mapspec, then all inputs must be scalars.
        lines.append("This pipeline accepts single values for all its inputs.")

    lines.append("\nExample `inputs` JSON:")
    lines.append(f"```json\n{example_str}\n```")

    return "\n".join(lines)


def _is_root_mapspec(mapspec: MapSpec, root_args: tuple[str, ...]) -> bool:
    """Check if a mapspec is a root mapspec."""
    return any(arg in mapspec.input_names for arg in root_args)


def _get_mapspec_section(pipeline: Pipeline) -> str:
    """Generate the pipeline-specific mapspec information section."""
    mapspecs = pipeline.mapspecs(ordered=True)
    if not mapspecs:
        return "This pipeline does not use array processing (`mapspec`). It only accepts single values for inputs."

    lines = ["The following rules define how arrays are processed in this pipeline:"]
    root_args = pipeline.root_args()
    for i, mapspec in enumerate(mapspecs, 1):
        context = (
            " (driven by your inputs)"
            if _is_root_mapspec(mapspec, root_args)
            else " (internal processing step)"
        )
        lines.append(f"  {i}. `{mapspec}`{context}")

    lines.append(
        "\nFor general `MapSpec` syntax and patterns, please refer to the main server instructions.",
    )
    return "\n".join(lines)


def _format_tool_description(pipeline: Pipeline) -> str:
    """Format a complete tool description using the template."""
    pipeline_name = pipeline.name or _DEFAULT_PIPELINE_NAME
    pipeline_description = pipeline.description or _DEFAULT_PIPELINE_DESCRIPTION
    documentation = _get_pipeline_documentation(pipeline)
    pipeline_info = _get_pipeline_info_summary(pipeline_name, pipeline)
    input_format_section = _get_input_format_section(pipeline)
    mapspec_section = _get_mapspec_section(pipeline)
    return _PIPELINE_EXECUTE_DESCRIPTION_TEMPLATE.format(
        pipeline_name=pipeline_name,
        pipeline_description=pipeline_description,
        pipeline_info=pipeline_info,
        input_format_section=input_format_section,
        mapspec_section=mapspec_section,
        documentation=documentation,
    )


def build_mcp_server(pipeline: Pipeline, **fast_mcp_kwargs: Any) -> fastmcp.FastMCP:
    """Build an MCP (Model Context Protocol) server for a Pipefunc pipeline.

    This function creates a FastMCP server that exposes your Pipefunc pipeline as an
    MCP tool, allowing AI assistants and other MCP clients to execute your computational
    workflows. The server automatically generates parameter validation, documentation,
    and provides parallel execution capabilities.

    Parameters
    ----------
    pipeline
        A Pipefunc Pipeline object containing the computational workflow to expose.
        The pipeline's functions, parameters, and mapspecs will be automatically
        analyzed to generate the MCP tool interface.
    **fast_mcp_kwargs
        Additional keyword arguments to pass to the FastMCP server.
        See {class}`fastmcp.FastMCP` for more details.

    Returns
    -------
    fastmcp.FastMCP
        A configured FastMCP server instance ready to run. The server includes:

        - Automatic parameter validation using Pydantic models
        - Documentation based on the pipeline's functions doc-strings and parameter annotations
        - Synchronous and asynchronous execution capabilities
        - Job management for async pipeline execution with progress tracking
        - JSON-serializable output formatting

    Examples
    --------
    **Basic Usage:**

    Create and run an MCP server from a pipeline::

        # my_mcp.py
        from physics_pipeline import pipeline_charge  # import from module to enable correct serialization
        from pipefunc.mcp import build_mcp_server

        if __name__ == "__main__":  # Important to use this 'if' for parallel execution!
            mcp = build_mcp_server(pipeline_charge)
            mcp.run(path="/charge", port=8000, transport="streamable-http")

    **Client Configuration:**

    Register the server with an MCP client (e.g., Cursor IDE ``.cursor/mcp.json``)::

        {
          "mcpServers": {
            "physics-simulation": {
              "url": "http://127.0.0.1:8000/charge"
            }
          }
        }

    **Alternative Transport Methods:**

    .. code-block:: python

        # HTTP server (recommended for development)
        mcp = build_mcp_server(pipeline)
        mcp.run(path="/api", port=8000, transport="streamable-http")

        # Standard I/O (for CLI integration)
        mcp = build_mcp_server(pipeline)
        mcp.run(transport="stdio")

        # Server-Sent Events
        mcp = build_mcp_server(pipeline)
        mcp.run(transport="sse")

    **Pipeline Requirements:**

    Your pipeline should be properly configured with JSON serializable inputs
    with proper type annotations::

        from pipefunc import pipefunc, Pipeline

        @pipefunc(output_name="result")
        def calculate(x: float, y: float) -> float:
            return x * y + 2

        pipeline = Pipeline([calculate])
        mcp = build_mcp_server(pipeline)

    **Async Pipeline Execution:**

    The server provides tools for asynchronous pipeline execution with job management::

        # Start an async job
        execute_pipeline_async(inputs={"x": [1, 2, 3], "y": [4, 5, 6]})
        # Returns: {"job_id": "uuid-string", "run_folder": "runs/job_uuid-string"}

        # Check job status and progress
        check_job_status(job_id="uuid-string")
        # Returns status, progress, and results when complete

        # Cancel a running job
        cancel_job(job_id="uuid-string")

        # List all tracked jobs
        list_jobs()
        # Returns summary of all jobs with their status

    **Execution Modes:**

    The server provides two execution patterns:

    1. **Synchronous execution** (``execute_pipeline_sync``):
       Uses ``pipeline.map()`` - blocks until completion, returns results immediately.
       Best for small-to-medium pipelines when you need results right away.

    2. **Asynchronous execution** (``execute_pipeline_async``):
       Uses ``pipeline.map_async()`` - returns immediately with job tracking.
       Best for long-running pipelines, background processing, and when you need
       progress monitoring or cancellation capabilities.

    Notes
    -----
    - The server automatically handles type validation using the pipeline's Pydantic model
    - Output arrays are converted to JSON-compatible lists
    - Parallel execution is enabled by default but can be disabled per request
    - Async execution provides job management with progress tracking and cancellation capabilities
    - Job registry is maintained globally across all MCP tool calls

    See Also
    --------
    run_mcp_server : Convenience function to build and run server in one call
    Pipeline.map : The underlying method used to execute pipeline workflows

    """
    requires("mcp", "rich", "griffe", reason="mcp", extras="mcp")

    pipeline_name = pipeline.name or _DEFAULT_PIPELINE_NAME
    server_instructions = _PIPEFUNC_INSTRUCTIONS
    execute_pipeline_tool_description = _format_tool_description(pipeline)
    async_execute_pipeline_tool_description = (
        execute_pipeline_tool_description + _PIPELINE_ASYNC_EXECUTE_DESCRIPTION_EXTRA
    )

    Model = pipeline.pydantic_model()  # noqa: N806
    Model.model_rebuild()  # Ensure all type references are resolved
    mcp = fastmcp.FastMCP(
        name=pipeline_name,
        instructions=server_instructions,
        **fast_mcp_kwargs,
    )

    @mcp.tool(name="execute_pipeline_sync", description=execute_pipeline_tool_description)
    async def execute_pipeline_sync(
        ctx: fastmcp.Context,
        inputs: Model,  # type: ignore[valid-type]
        parallel: bool = True,  # noqa: FBT002
        run_folder: str | None = None,
    ) -> str:
        """Execute pipeline synchronously and return results immediately.

        Blocks until completion. Best for small-to-medium pipelines.
        """
        return await _execute_pipeline_sync(pipeline, ctx, inputs, parallel, run_folder)

    @mcp.tool(name="execute_pipeline_async", description=async_execute_pipeline_tool_description)
    async def execute_pipeline_async(
        ctx: fastmcp.Context,
        inputs: Model,  # type: ignore[valid-type]
        run_folder: str | None = None,
    ) -> str:
        """Start pipeline execution asynchronously and return job_id immediately.

        Returns job_id for tracking. Best for long-running pipelines and parameter sweeps.
        """
        return await _execute_pipeline_async(pipeline, ctx, inputs, run_folder)

    @mcp.tool(
        name="check_job_status",
        description="Check status of an active pipeline job started by execute_pipeline_async in this MCP session. Only works for jobs in the current session's registry, not previous sessions.",
    )
    async def check_job_status(job_id: str) -> str:
        """Check status of active jobs from current session only."""
        return await _check_job_status(job_id)

    @mcp.tool(
        name="cancel_job",
        description="Cancel an active pipeline job from this MCP session."
        " Only works for jobs started by execute_pipeline_async in the current session.",
    )
    async def cancel_job(ctx: fastmcp.Context, job_id: str) -> str:
        """Cancel active jobs from current session only."""
        return await _cancel_job(ctx, job_id)

    @mcp.tool(
        name="list_jobs",
        description="List all active pipeline jobs tracked in this MCP session."
        " Only shows jobs from execute_pipeline_async in the current session, not previous sessions or other executions.",
    )
    async def list_jobs() -> str:
        """List active jobs from current session only."""
        return await _list_jobs()

    @mcp.tool(
        name="run_info",
        description="Get information about ANY pipeline run folder on disk, including runs"
        " from previous sessions, other executions, or direct pipeline.map() calls."
        " Universal inspection tool.",
    )
    def run_info(run_folder: str) -> dict[str, Any]:
        """Inspect any run folder on disk, works across sessions."""
        return _run_info(run_folder)

    @mcp.tool(
        name="list_historical_runs",
        description="List all historical pipeline runs from disk. Scans a folder for run directories"
        " and shows run information sorted by modification time. Works across all sessions and executions.",
    )
    def list_historical_runs(folder: str = "runs", max_runs: int | None = None) -> _Runs:
        """List all historical pipeline runs from disk, sorted by modification time."""
        return _list_historical_runs(folder, max_runs)

    @mcp.tool(
        name="load_outputs",
        description="Load outputs from a pipeline run folder. Works across all sessions and executions.",
    )
    def load_outputs(run_folder: str, output_names: list[str] | None = None) -> dict[str, Any]:
        """Load outputs from a pipeline run folder."""
        return _load_outputs(run_folder, output_names)

    return mcp


async def _execute_pipeline_sync(
    pipeline: Pipeline,
    ctx: fastmcp.Context,
    inputs: pydantic.BaseModel,  # type: ignore[valid-type]
    parallel: bool = True,  # noqa: FBT002
    run_folder: str | None = None,
) -> str:
    await ctx.info(f"Executing pipeline {pipeline.name=} with inputs: {inputs}")
    result = pipeline.map(
        inputs=inputs,
        parallel=parallel,
        run_folder=run_folder,
    )
    await ctx.info(f"Pipeline {pipeline.name=} executed")
    # Convert ResultDict to a more readable format
    output = {}
    for key, result_obj in result.items():
        output[key] = {
            "output": result_obj.output.tolist()
            if hasattr(result_obj.output, "tolist")
            else result_obj.output,
            "shape": getattr(result_obj.output, "shape", None),
        }
    return str(output)


async def _execute_pipeline_async(
    pipeline: Pipeline,
    ctx: fastmcp.Context,
    inputs: pydantic.BaseModel,  # type: ignore[valid-type]
    run_folder: str | None = None,
) -> str:
    job_id = str(uuid.uuid4())
    actual_run_folder = run_folder or f"runs/job_{job_id}"

    await ctx.info(
        f"Starting async pipeline {pipeline.name=} with job_id={job_id} and run_folder={actual_run_folder}",
    )

    # Store the AsyncMap object in the global registry
    async_map = pipeline.map_async(
        inputs=inputs,
        run_folder=actual_run_folder,
        show_progress="headless",
    )
    job_registry[job_id] = JobInfo(
        runner=async_map,
        started_at=datetime.now(tz=timezone.utc),
        run_folder=actual_run_folder,
        status="running",
        pipeline_name=pipeline.name or _DEFAULT_PIPELINE_NAME,
    )

    await ctx.info(f"Started async job {job_id} in folder {actual_run_folder}")
    return str({"job_id": job_id, "run_folder": actual_run_folder})


async def _check_job_status(job_id: str) -> str:
    job = job_registry.get(job_id)
    if not job:
        return str({"error": "Job not found"})

    task = job.runner.task
    is_done = task.done()

    # Get progress information if available
    assert job.runner.progress is not None
    progress_dict = job.runner.progress.progress_dict
    progress_info = {}
    for output_name, status in progress_dict.items():
        elapsed_time = status.elapsed_time()
        remaining_time = status.remaining_time(elapsed_time=elapsed_time)
        progress_info[str(output_name)] = {
            "progress": status.progress,
            "n_completed": status.n_completed,
            "n_total": status.n_total,
            "n_failed": status.n_failed,
            "elapsed_time": elapsed_time if elapsed_time is not None else None,
            "remaining_time": remaining_time if remaining_time is not None else None,
        }

    result_info = {
        "job_id": job_id,
        "pipeline_name": job.pipeline_name,
        "status": "completed" if is_done else "running",
        "progress": progress_info,
        "run_folder": job.run_folder,
        "started_at": job.started_at.isoformat(),
        "error": str(task.exception()) if is_done and task.exception() else None,
    }

    # If job is completed, get the results
    if is_done and not task.exception():
        pipeline_result = task.result()
        output = {}
        for key, result_obj in pipeline_result.items():
            output[key] = {
                "output": result_obj.output.tolist()
                if hasattr(result_obj.output, "tolist")
                else result_obj.output,
                "shape": getattr(result_obj.output, "shape", None),
            }
        result_info["results"] = output

    return str(result_info)


async def _cancel_job(ctx: fastmcp.Context, job_id: str) -> str:
    job = job_registry.get(job_id)
    if not job:
        return str({"error": "Job not found"})

    task = job.runner.task
    if not task.done():
        task.cancel()
        job.status = "cancelled"
        await ctx.info(f"Cancelled job {job_id}")
        return str({"status": "cancelled", "job_id": job_id})
    return str({"error": "Job not found or already completed", "job_id": job_id})


async def _list_jobs() -> str:
    if not job_registry:
        return str({"jobs": [], "total_count": 0})

    jobs_info = []
    for job_id, job in job_registry.items():
        task = job.runner.task
        is_done = task.done()
        job_info = {
            "job_id": job_id,
            "pipeline_name": job.pipeline_name,
            "status": job.status,
            "run_folder": job.run_folder,
            "started_at": job.started_at.isoformat(),
            "has_error": is_done and task.exception() is not None,
        }
        jobs_info.append(job_info)

    return str({"jobs": jobs_info, "total_count": len(jobs_info)})


def _progress_info_from_disk(run_info: RunInfo) -> tuple[dict[str, Any], bool]:
    outputs = {}
    store = run_info.init_store()
    all_complete = True
    for output_name in run_info.all_output_names:
        data = store[output_name]
        if isinstance(data, StorageBase):
            assert isinstance(data, FileArray)
            size: int | str
            progress: float | str
            if shape_is_resolved(data.shape):
                size = data.size
                progress = 1.0 - sum(data.mask_linear()) / size
            else:  # pragma: no cover
                size = "unknown"
                progress = "unknown"
            nbytes = sum(f.stat().st_size for f in data.folder.rglob("*") if f.is_file())
        elif isinstance(data, Path):
            if data.exists():
                progress = 1.0
                nbytes = data.stat().st_size
            else:
                progress = 0.0
                nbytes = 0
        else:  # pragma: no cover
            msg = f"Should not happen: {type(data)}"
            raise RuntimeError(msg)  # noqa: TRY004
        complete = progress == 1.0
        all_complete = all_complete and complete
        outputs[output_name] = {
            "progress": progress,
            "complete": complete,
            "bytes": nbytes,
        }
    return outputs, all_complete


class _RunEntry(TypedDict):
    last_modified: str
    run_folder: str
    all_complete: bool
    total_outputs: int
    completed_outputs: int
    pipefunc_version: str


class _Runs(TypedDict):
    runs: list[_RunEntry]
    total_count: int
    folder: str
    scanned_directories: int
    error: str | None


def _list_historical_runs(folder: str = "runs", max_runs: int | None = None) -> _Runs:
    """List all historical pipeline runs from disk, sorted by modification time."""
    runs_folder = Path(folder)
    runs: _Runs = {
        "runs": [],
        "total_count": 0,
        "folder": folder,
        "scanned_directories": 0,
        "error": None,
    }
    if not runs_folder.exists():  # pragma: no cover
        runs["error"] = f"Folder '{folder}' does not exist"
        return runs
    if not runs_folder.is_dir():  # pragma: no cover
        runs["error"] = f"'{folder}' is not a directory"
        return runs

    for run_folder in runs_folder.iterdir():
        runs["scanned_directories"] += 1
        if not run_folder.is_dir():  # pragma: no cover
            continue
        run_info_path = RunInfo.path(str(run_folder))
        if not run_info_path.exists():  # pragma: no cover
            continue
        mod_time = datetime.fromtimestamp(run_info_path.stat().st_mtime)  # noqa: DTZ006
        run_entry: _RunEntry = {  # type: ignore[typeddict-item]
            "last_modified": mod_time.isoformat(),
            "run_folder": str(run_folder),
        }
        runs["runs"].append(run_entry)

    # Sort by modification time (newest first)
    runs["runs"].sort(key=operator.itemgetter("last_modified"), reverse=True)

    if max_runs is not None and max_runs > 0:
        runs["runs"] = runs["runs"][:max_runs]

    for run_entry in runs["runs"]:
        try:
            run_info = RunInfo.load(run_entry["run_folder"])
        except Exception:  # noqa: BLE001, S112  # pragma: no cover
            continue  # Skip directories that don't contain valid run info
        runs["total_count"] += 1
        run_info_json = json.loads(run_info_path.read_text())
        outputs, all_complete = _progress_info_from_disk(run_info)
        run_entry["all_complete"] = all_complete
        run_entry["total_outputs"] = len(outputs)
        run_entry["completed_outputs"] = sum(1 for output in outputs.values() if output["complete"])
        run_entry["pipefunc_version"] = run_info_json.get("pipefunc_version", "unknown")

    return runs


def _run_info(run_folder: str) -> dict[str, Any]:
    try:
        run_info = RunInfo.load(run_folder)
    except Exception as e:  # noqa: BLE001  # pragma: no cover
        return {"error": str(e)}
    assert isinstance(run_info, RunInfo)
    outputs, all_complete = _progress_info_from_disk(run_info)
    run_info_json = json.loads(run_info.path(run_folder).read_text())
    return {"run_info": run_info_json, "outputs": outputs, "all_complete": all_complete}


def _load_outputs(run_folder: str, output_names: list[str] | None = None) -> dict[str, Any]:
    try:
        if output_names is None:
            return load_all_outputs(run_folder=run_folder)
        return load_outputs(*output_names, run_folder=run_folder)
    except Exception as e:  # noqa: BLE001  # pragma: no cover
        return {"error": str(e)}
