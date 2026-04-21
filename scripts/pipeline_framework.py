"""
Unified pipeline orchestration framework for ODesignBench tasks.

This module standardizes stage scheduling while keeping task-specific
implementations in small plugins/specs.
"""

from __future__ import annotations

import json
import os
import sys
import shutil
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

from hydra.utils import get_original_cwd


# Allow imports from ODesignBench root when running under scripts/.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocess.preprocess import Preprocess
from refold.refold_api import ReFold
from evaluation.evaluation_api import Evaluation
from evaluation.motif_scaffolding.motif_scaffolding_evaluation import _discover_motifs, _prepare_input_dir


StageFn = Callable[["PipelineContext"], None]
EvaluationPluginFn = Callable[["PipelineContext"], None]


@dataclass
class PipelineContext:
    cfg: object
    task_name: str
    pipeline_dir: Path
    design_dir: str
    origin_cwd: str
    gpu_list: list[str]
    preprocess_model: Preprocess
    inversefold_model: Optional[Any]
    refold_model: ReFold
    evaluation_model: Evaluation
    runtime: dict


@dataclass
class TaskSpec:
    preprocess_stage: Optional[StageFn]
    inversefold_stage: Optional[StageFn]
    refold_prepare_stage: Optional[StageFn]
    refold_stage: Optional[StageFn]
    evaluation_plugins: list[EvaluationPluginFn]


def _gpus_to_list(gpus: object) -> list[str]:
    """Normalize gpus config to ['0'] / ['0','1'] for downstream APIs."""
    if isinstance(gpus, (list, tuple)):
        return [str(x).strip().strip("[]'\"") for x in gpus]

    s = str(gpus).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if "," in s:
        return [x.strip().strip("[]'\"") for x in s.split(",") if x.strip()]
    return [s.strip().strip("[]'\"")]


def _resolve_pipeline_dir(cfg: object) -> Path:
    root = Path(str(getattr(cfg, "root")))
    if root.is_absolute():
        return root
    return Path(get_original_cwd()) / root


def _resolve_design_dir(cfg: object) -> str:
    design_dir = str(getattr(cfg, "design_dir"))
    if os.path.isabs(design_dir):
        return os.path.normpath(design_dir)
    return os.path.normpath(os.path.join(get_original_cwd(), design_dir))


def _stage_enabled(cfg: object, stage_key: str) -> bool:
    unified_cfg = cfg.get("unified", None) if hasattr(cfg, "get") else None
    if unified_cfg is None:
        return True
    steps_cfg = unified_cfg.get("steps", None) if hasattr(unified_cfg, "get") else None
    if steps_cfg is None:
        return True
    return bool(steps_cfg.get(stage_key, True))


def _run_stage(ctx: PipelineContext, stage_key: str, fn: Optional[StageFn]) -> None:
    if fn is None:
        return
    if not _stage_enabled(ctx.cfg, stage_key):
        print(f"[unified] Skip stage '{stage_key}' (disabled by config)")
        return
    print(f"[unified] Running stage: {stage_key}")
    t0 = perf_counter()
    error_msg = None
    try:
        fn(ctx)
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        elapsed = perf_counter() - t0
        timings = ctx.runtime.setdefault("timings", {})
        timings[stage_key] = {
            "elapsed_seconds": round(elapsed, 3),
            "success": error_msg is None,
        }
        if error_msg is not None:
            timings[stage_key]["error"] = error_msg
        print(
            f"[unified] Stage '{stage_key}' finished in {elapsed:.2f}s"
            f"{' (failed)' if error_msg is not None else ''}"
        )


def _get_cfg_value(cfg: object, key: str, default: object = None) -> object:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_interface_pocket_cutoff(ctx: PipelineContext) -> float:
    interface_cfg = _get_cfg_value(ctx.cfg, "interface", {}) or {}
    if hasattr(interface_cfg, "get"):
        cutoff = interface_cfg.get("pocket_cutoff", 3.5)
    else:
        cutoff = getattr(interface_cfg, "pocket_cutoff", 3.5)
    return float(cutoff)


def _infer_motif_name_from_pdbs(pdb_paths: list[Path], fallback_name: str) -> str:
    """
    Infer motif identifier from sample PDB names.

    Examples:
      - 01_1LDB_0.pdb -> 01_1LDB
      - 01_1LDB-7.pdb -> 01_1LDB
    """
    if not pdb_paths:
        return fallback_name
    stem = pdb_paths[0].stem
    inferred = re.sub(r"([_-])\d+$", "", stem)
    return inferred or fallback_name


def _resolve_metadata_csv(
    ctx: PipelineContext,
    *,
    cfg_key: str,
    preferred_names: list[str],
    csv_label: str,
) -> str:
    """Resolve metadata CSV path from config override or design_dir defaults."""
    csv_path = _get_cfg_value(ctx.cfg, cfg_key, None)
    if csv_path is not None and str(csv_path).strip():
        if not os.path.isabs(str(csv_path)):
            csv_path = os.path.join(ctx.origin_cwd, str(csv_path))
        csv_path = os.path.normpath(str(csv_path))
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_label} not found: {csv_path}")
        print(f"[unified] Using {csv_label} from config override: {csv_path}")
        return csv_path

    design_dir = Path(ctx.design_dir)
    for filename in preferred_names:
        candidate = design_dir / filename
        if candidate.exists():
            return os.path.normpath(str(candidate))

    csv_files = sorted(design_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return os.path.normpath(str(csv_files[0]))

    if len(csv_files) == 0:
        preferred_text = ", ".join(preferred_names)
        raise FileNotFoundError(
            f"{csv_label} not found under design_dir={design_dir}. "
            f"Expected one of [{preferred_text}] or exactly one *.csv file."
        )

    matched = ", ".join(str(p) for p in csv_files)
    raise ValueError(
        f"Multiple CSV files found under design_dir={design_dir}; cannot infer {csv_label}. "
        f"Please keep only one metadata CSV or use a standard filename ({', '.join(preferred_names)}). "
        f"Found: {matched}"
    )


def _write_runtime_summary(
    ctx: PipelineContext,
    total_elapsed: float,
    success: bool,
    error_msg: Optional[str] = None,
) -> Path:
    """Persist stage timings to disk for reliable post-run auditing."""
    timings = ctx.runtime.get("timings", {})
    stage_elapsed_seconds = {}
    for key, value in timings.items():
        if isinstance(value, dict) and "elapsed_seconds" in value:
            stage_elapsed_seconds[key] = value["elapsed_seconds"]

    payload = {
        "task_name": ctx.task_name,
        "pipeline_dir": str(ctx.pipeline_dir),
        "success": success,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "stage_elapsed_seconds": stage_elapsed_seconds,
        "timings": timings,
    }
    if error_msg is not None:
        payload["error"] = error_msg

    summary_path = ctx.pipeline_dir / "runtime_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return summary_path


def _plugin_protein_eval(ctx: PipelineContext) -> None:
    ctx.runtime["evaluation_protein"] = ctx.evaluation_model.run(
        task="protein",
        pipeline_dir=str(ctx.pipeline_dir),
        output_csv=str(ctx.pipeline_dir / "raw_data.csv"),
    )


def _plugin_pbp_eval(ctx: PipelineContext) -> None:
    kwargs = {}
    pbp_info_csv = ctx.runtime.get("pbp_info_csv", None)
    if not pbp_info_csv:
        pbp_info_csv = _resolve_metadata_csv(
            ctx,
            cfg_key="pbp_info_csv",
            preferred_names=["pbp_info.csv"],
            csv_label="PBP info CSV",
        )
        ctx.runtime["pbp_info_csv"] = pbp_info_csv
    if pbp_info_csv:
        kwargs["pbp_info_csv"] = pbp_info_csv
    ctx.runtime["evaluation_pbp"] = ctx.evaluation_model.run(
        task="pbp",
        pipeline_dir=str(ctx.pipeline_dir),
        output_csv=str(ctx.pipeline_dir / "raw_data.csv"),
        **kwargs,
    )


def _plugin_lbp_eval(ctx: PipelineContext) -> None:
    lbp_info_csv = _get_or_resolve_lbp_info_csv(ctx)
    ctx.runtime["evaluation_lbp"] = ctx.evaluation_model.run(
        task="lbp",
        pipeline_dir=str(ctx.pipeline_dir),
        cands=str(ctx.pipeline_dir / "refold" / "chai1_out" / "chai1_cands.pkl"),
        output_csv=str(ctx.pipeline_dir / "raw_data.csv"),
        lbp_info_csv=lbp_info_csv,
    )


def _plugin_interface_eval(ctx: PipelineContext) -> None:
    kwargs = {
        "pipeline_dir": str(ctx.pipeline_dir),
        "cands": str(ctx.pipeline_dir / "refold" / "chai1_out" / "chai1_cands.pkl"),
        "output_csv": str(ctx.pipeline_dir / "raw_data.csv"),
        "pocket_cutoff": _get_interface_pocket_cutoff(ctx),
    }
    interface_info_csv = ctx.runtime.get("interface_info_csv", None)
    if interface_info_csv:
        kwargs["interface_info_csv"] = interface_info_csv
    kwargs["lbp_info_csv"] = _get_or_resolve_lbp_info_csv(ctx)
    ctx.runtime["evaluation_interface"] = ctx.evaluation_model.run(
        task="interface",
        **kwargs,
    )


def _plugin_nuc_eval(ctx: PipelineContext) -> None:
    ctx.runtime["evaluation_nuc"] = ctx.evaluation_model.run(
        task="nuc",
        pipeline_dir=str(ctx.pipeline_dir),
        output_csv=str(ctx.pipeline_dir / "raw_data.csv"),
    )


def _plugin_nbl_eval(ctx: PipelineContext) -> None:
    ctx.runtime["evaluation_nbl"] = ctx.evaluation_model.run(
        task="nbl",
        pipeline_dir=str(ctx.pipeline_dir),
        output_csv=str(ctx.pipeline_dir / "raw_data.csv"),
    )


def _plugin_pbn_eval(ctx: PipelineContext) -> None:
    ctx.runtime["evaluation_pbn"] = ctx.evaluation_model.run(
        task="pbn",
        pipeline_dir=str(ctx.pipeline_dir),
        output_csv=str(ctx.pipeline_dir / "raw_data.csv"),
    )


def _plugin_pbl_eval(ctx: PipelineContext) -> None:
    # Legacy PBL path uses pre-refold inputs for evaluation.
    eval_input = ctx.pipeline_dir / "inversefold_formatted_designs_for_evaluation"
    ctx.runtime["evaluation_pbl"] = ctx.evaluation_model.run(
        task="pbl",
        input_dir=str(eval_input),
        output_dir=str(ctx.pipeline_dir / "inversefold_formatted_designs_for_evaluation_metrics"),
    )


def _get_or_resolve_ame_csv(ctx: PipelineContext) -> str:
    if "ame_csv_path" in ctx.runtime:
        return ctx.runtime["ame_csv_path"]
    
    ame_csv_path = _resolve_metadata_csv(
        ctx,
        cfg_key="ame_csv",
        preferred_names=["ame.csv", "ame_info.csv"],
        csv_label="AME metadata CSV",
    )
    ctx.runtime["ame_csv_path"] = ame_csv_path
    
    from inversefold.ame_csv_utils import load_ame_csv
    ctx.runtime["ame_csv_df"] = load_ame_csv(ame_csv_path)
    
    return ame_csv_path


def _get_or_resolve_lbp_info_csv(ctx: PipelineContext) -> str:
    if "lbp_info_csv" in ctx.runtime:
        return ctx.runtime["lbp_info_csv"]

    lbp_info_csv = _resolve_metadata_csv(
        ctx,
        cfg_key="lbp_info_csv",
        preferred_names=["lbp_info.csv"],
        csv_label="LBP info CSV",
    )
    ctx.runtime["lbp_info_csv"] = lbp_info_csv

    from inversefold.lbp_csv_utils import load_lbp_info_csv

    ctx.runtime["lbp_info_df"] = load_lbp_info_csv(lbp_info_csv)
    return lbp_info_csv


def _plugin_ame_eval(ctx: PipelineContext) -> None:
    chai1_cands = ctx.pipeline_dir / "refold" / "chai1_out" / "chai1_cands.pkl"
    if not chai1_cands.exists():
        raise FileNotFoundError(f"AME refold output not found: {chai1_cands}")
    ame_cfg = _get_cfg_value(ctx.cfg, "ame", {}) or {}
    pocket_cutoff = 8.0
    if hasattr(ame_cfg, "get"):
        pocket_cutoff = float(ame_cfg.get("pocket_cutoff", 8.0))
    ctx.runtime["evaluation_ame"] = ctx.evaluation_model.run(
        task="ame",
        pipeline_dir=str(ctx.pipeline_dir),
        cands=str(chai1_cands),
        output_csv=str(ctx.pipeline_dir / "raw_data.csv"),
        pocket_cutoff=pocket_cutoff,
        ame_csv=_get_or_resolve_ame_csv(ctx),
    )


def _plugin_ame_statistics(ctx: PipelineContext) -> None:
    raw_data_csv = ctx.pipeline_dir / "raw_data.csv"
    if not raw_data_csv.exists():
        return
    try:
        from evaluation.metrics.ame_statistics import generate_ame_statistics

        generate_ame_statistics(
            csv_path=str(raw_data_csv),
            output_dir=str(ctx.pipeline_dir),
            success_col="success",
            save_plots=True,
            save_stats=True,
        )
    except Exception as exc:
        print(f"[unified] Warning: AME statistics generation failed: {exc}")


def _preprocess_protein(ctx: PipelineContext) -> None:
    ctx.runtime["preprocess"] = ctx.preprocess_model.run(
        action="format_output_pdb",
        input_dir=ctx.design_dir,
        output_dir=str(ctx.pipeline_dir / "formatted_designs"),
    )


def _preprocess_ligand(ctx: PipelineContext) -> None:
    ctx.runtime["preprocess"] = ctx.preprocess_model.run(
        action="format_output_ligand",
        input_dir=ctx.design_dir,
        output_dir=str(ctx.pipeline_dir / "formatted_designs"),
    )


def _preprocess_interface(ctx: PipelineContext) -> None:
    _preprocess_ligand(ctx)

    from inversefold.interface_utils import build_interface_info_csv

    formatted_dir = ctx.pipeline_dir / "formatted_designs"
    pocket_cutoff = _get_interface_pocket_cutoff(ctx)
    struct_paths = sorted(formatted_dir.glob("*.cif")) + sorted(formatted_dir.glob("*.pdb"))
    if not struct_paths:
        raise FileNotFoundError(f"No formatted interface structures found under {formatted_dir}")

    lbp_info_csv = _get_or_resolve_lbp_info_csv(ctx)
    lbp_info_df = ctx.runtime["lbp_info_df"]
    from inversefold.lbp_csv_utils import match_pdb_to_lbp_info

    design_chain_map = {}
    for struct_path in struct_paths:
        info = match_pdb_to_lbp_info(struct_path, lbp_info_df)
        if info is None:
            raise ValueError(
                f"No LBP info row found for '{struct_path.name}'. "
                "Ensure design_name in lbp_info.csv matches each formatted design filename."
            )
        design_chain_map[struct_path.stem] = info["design_chain"]

    interface_info_csv = build_interface_info_csv(
        formatted_dir,
        ctx.pipeline_dir / "interface_info.csv",
        pocket_cutoff=pocket_cutoff,
        design_chain_map=design_chain_map,
    )
    ctx.runtime["interface_info_csv"] = str(interface_info_csv)

    preprocess_result = ctx.runtime.get("preprocess", {})
    details = preprocess_result.setdefault("details", {})
    details["interface_pocket_cutoff"] = pocket_cutoff
    details["interface_design_count"] = len(struct_paths)
    details["interface_info_csv"] = str(interface_info_csv)
    details["lbp_info_csv"] = str(lbp_info_csv)


def _preprocess_cif(ctx: PipelineContext) -> None:
    ctx.runtime["preprocess"] = ctx.preprocess_model.run(
        action="format_output_cif",
        input_dir=ctx.design_dir,
        output_dir=str(ctx.pipeline_dir / "formatted_designs"),
    )


def _require_inversefold_model(ctx: PipelineContext) -> Any:
    if ctx.inversefold_model is None:
        raise RuntimeError(
            "inversefold model is not initialized for this task. "
            "Use a task spec with inversefold stage disabled/passthrough only."
        )
    return ctx.inversefold_model


def _inversefold_passthrough_formatted_designs(ctx: PipelineContext) -> None:
    """
    Nucleic-acid eval path: use ODesign outputs directly.
    No sequence redesign is performed in ODesignBench.
    """
    formatted_dir = ctx.pipeline_dir / "formatted_designs"
    inverse_fold_dir = ctx.pipeline_dir / "inverse_fold"
    inverse_fold_dir.mkdir(parents=True, exist_ok=True)

    backbone_paths = sorted(formatted_dir.glob("*.cif")) + sorted(formatted_dir.glob("*.pdb"))
    if not backbone_paths:
        raise FileNotFoundError(
            f"No formatted structures found under {formatted_dir}. "
            "Expected at least one *.cif or *.pdb from preprocess stage."
        )

    copied = 0
    for src in backbone_paths:
        dst = inverse_fold_dir / src.name
        shutil.copy2(src, dst)
        copied += 1

    ctx.runtime["inversefold"] = {
        "success": True,
        "stage": "inversefold.passthrough_from_formatted_designs",
        "outputs": {"output_dir": str(inverse_fold_dir)},
        "details": {"num_inputs": len(backbone_paths), "num_copied": copied},
    }


def _inversefold_proteinmpnn(ctx: PipelineContext) -> None:
    ctx.runtime["inversefold"] = _require_inversefold_model(ctx).run(
        action="proteinmpnn_distributed",
        input_dir=ctx.pipeline_dir / "formatted_designs",
        output_dir=str(ctx.pipeline_dir / "inverse_fold"),
        gpu_list=ctx.gpu_list,
        origin_cwd=ctx.origin_cwd,
    )


def _inversefold_pbp(ctx: PipelineContext) -> None:
    pbp_info_csv = _resolve_metadata_csv(
        ctx,
        cfg_key="pbp_info_csv",
        preferred_names=["pbp_info.csv"],
        csv_label="PBP info CSV",
    )

    from inversefold.pbp_csv_utils import load_pbp_info_csv

    pbp_info_df = load_pbp_info_csv(pbp_info_csv)
    ctx.runtime["pbp_info_csv"] = pbp_info_csv
    ctx.runtime["pbp_info_df"] = pbp_info_df

    ctx.runtime["inversefold"] = _require_inversefold_model(ctx).run(
        action="proteinmpnn_distributed",
        input_dir=ctx.pipeline_dir / "formatted_designs",
        output_dir=str(ctx.pipeline_dir / "inverse_fold"),
        gpu_list=ctx.gpu_list,
        origin_cwd=ctx.origin_cwd,
        pbp_info_csv=pbp_info_csv,
        pbp_info_df=pbp_info_df,
    )


def _inversefold_ligandmpnn(ctx: PipelineContext) -> None:
    ctx.runtime["inversefold"] = _require_inversefold_model(ctx).run(
        action="ligandmpnn_distributed",
        input_dir=ctx.pipeline_dir / "formatted_designs",
        output_dir=str(ctx.pipeline_dir / "inverse_fold"),
        gpu_list=ctx.gpu_list,
        origin_cwd=ctx.origin_cwd,
    )


def _inversefold_lbp(ctx: PipelineContext) -> None:
    _get_or_resolve_lbp_info_csv(ctx)
    lbp_info_df = ctx.runtime["lbp_info_df"]

    from inversefold.lbp_csv_utils import (
        calculate_fixed_residues_from_chain_roles,
        match_pdb_to_lbp_info,
    )

    def _fixed_residue_calculator(struct_path: Path, _row: Any) -> list[str]:
        info = match_pdb_to_lbp_info(struct_path, lbp_info_df)
        if info is None:
            raise ValueError(
                f"No LBP info row found for '{struct_path.name}'. "
                "Ensure design_name in lbp_info.csv matches each formatted design filename."
            )
        return calculate_fixed_residues_from_chain_roles(
            struct_path,
            design_chain=info["design_chain"],
            target_chain=info["target_chain"],
        )

    ctx.runtime["inversefold"] = _require_inversefold_model(ctx).run(
        action="ligandmpnn_distributed",
        input_dir=ctx.pipeline_dir / "formatted_designs",
        output_dir=str(ctx.pipeline_dir / "inverse_fold"),
        gpu_list=ctx.gpu_list,
        origin_cwd=ctx.origin_cwd,
        fixed_residues_calculator=_fixed_residue_calculator,
    )


def _inversefold_interface(ctx: PipelineContext) -> None:
    from inversefold.interface_utils import calculate_fixed_residues_from_ligand_proximity
    from inversefold.lbp_csv_utils import match_pdb_to_lbp_info

    pocket_cutoff = _get_interface_pocket_cutoff(ctx)
    _get_or_resolve_lbp_info_csv(ctx)
    lbp_info_df = ctx.runtime["lbp_info_df"]

    def _fixed_residue_calculator(struct_path: Path, _row: Any) -> list[str]:
        info = match_pdb_to_lbp_info(struct_path, lbp_info_df)
        if info is None:
            raise ValueError(
                f"No LBP info row found for '{struct_path.name}'. "
                "Ensure design_name in lbp_info.csv matches each formatted design filename."
            )
        return calculate_fixed_residues_from_ligand_proximity(
            struct_path,
            pocket_cutoff=pocket_cutoff,
            design_chain=info["design_chain"],
        )

    ctx.runtime["inversefold"] = _require_inversefold_model(ctx).run(
        action="ligandmpnn_distributed",
        input_dir=ctx.pipeline_dir / "formatted_designs",
        output_dir=str(ctx.pipeline_dir / "inverse_fold"),
        gpu_list=ctx.gpu_list,
        origin_cwd=ctx.origin_cwd,
        fixed_residues_calculator=_fixed_residue_calculator,
    )


def _inversefold_odesign_to_inverse_fold(ctx: PipelineContext) -> None:
    ctx.runtime["inversefold"] = _require_inversefold_model(ctx).run(
        action="odesignmpnn",
        input_root=ctx.pipeline_dir / "formatted_designs",
        inverse_fold_root=ctx.pipeline_dir / "inverse_fold",
    )


def _inversefold_odesign_to_inversefold(ctx: PipelineContext) -> None:
    # Keep compatibility with existing PBL output directory naming.
    ctx.runtime["inversefold"] = _require_inversefold_model(ctx).run(
        action="odesignmpnn",
        input_root=ctx.pipeline_dir / "formatted_designs",
        inverse_fold_root=ctx.pipeline_dir / "inversefold",
    )


def _prepare_refold_esm(ctx: PipelineContext) -> None:
    ctx.runtime["refold_prepare"] = ctx.refold_model.run(
        action="make_esmfold_json_multi_process",
        backbone_dir=str(ctx.pipeline_dir / "inverse_fold" / "backbones"),
        output_dir=str(ctx.pipeline_dir / "refold" / "esmfold_inputs.json"),
    )


def _run_refold_esm(ctx: PipelineContext) -> None:
    ctx.runtime["refold"] = ctx.refold_model.run(
        action="run_esmfold",
        sequences_file_json=str(ctx.pipeline_dir / "refold" / "esmfold_inputs.json"),
        output_dir=str(ctx.pipeline_dir / "refold" / "esmfold_out"),
    )


def _prepare_refold_af3_from_backbones(ctx: PipelineContext) -> None:
    ctx.runtime["refold_prepare"] = ctx.refold_model.run(
        action="make_af3_json_multi_process",
        backbone_dir=str(ctx.pipeline_dir / "inverse_fold" / "backbones"),
        output_path=str(ctx.pipeline_dir / "refold" / "af3_inputs"),
    )


def _prepare_refold_af3_pbp_target_msa(ctx: PipelineContext) -> None:
    pbp_info_df = ctx.runtime.get("pbp_info_df", None)
    if pbp_info_df is None:
        pbp_info_csv = _resolve_metadata_csv(
            ctx,
            cfg_key="pbp_info_csv",
            preferred_names=["pbp_info.csv"],
            csv_label="PBP info CSV",
        )
        from inversefold.pbp_csv_utils import load_pbp_info_csv
        pbp_info_df = load_pbp_info_csv(pbp_info_csv)
        ctx.runtime["pbp_info_csv"] = pbp_info_csv
        ctx.runtime["pbp_info_df"] = pbp_info_df

    ctx.runtime["refold_prepare"] = ctx.refold_model.run(
        action="make_af3_json_pbp_target_msa",
        backbone_dir=str(ctx.pipeline_dir / "inverse_fold" / "backbones"),
        output_path=str(ctx.pipeline_dir / "refold" / "af3_inputs"),
        pbp_info_df=pbp_info_df,
    )


def _prepare_refold_af3_pbp_dual(ctx: PipelineContext) -> None:
    pbp_info_df = ctx.runtime.get("pbp_info_df", None)
    if pbp_info_df is None:
        pbp_info_csv = _resolve_metadata_csv(
            ctx,
            cfg_key="pbp_info_csv",
            preferred_names=["pbp_info.csv"],
            csv_label="PBP info CSV",
        )
        from inversefold.pbp_csv_utils import load_pbp_info_csv
        pbp_info_df = load_pbp_info_csv(pbp_info_csv)
        ctx.runtime["pbp_info_csv"] = pbp_info_csv
        ctx.runtime["pbp_info_df"] = pbp_info_df

    complex_prepare = ctx.refold_model.run(
        action="make_af3_json_pbp_target_msa",
        backbone_dir=str(ctx.pipeline_dir / "inverse_fold" / "backbones"),
        output_path=str(ctx.pipeline_dir / "refold" / "af3_inputs"),
        pbp_info_df=pbp_info_df,
    )
    unbound_prepare = ctx.refold_model.run(
        action="make_af3_json_pbp_design_only",
        backbone_dir=str(ctx.pipeline_dir / "inverse_fold" / "backbones"),
        output_path=str(ctx.pipeline_dir / "refold" / "af3_unbound_inputs"),
        pbp_info_df=pbp_info_df,
    )
    ctx.runtime["refold_prepare"] = {
        "complex": complex_prepare,
        "unbound": unbound_prepare,
    }


def _prepare_refold_af3_from_inverse_fold(ctx: PipelineContext) -> None:
    ctx.runtime["refold_prepare"] = ctx.refold_model.run(
        action="make_af3_json_multi_process",
        backbone_dir=str(ctx.pipeline_dir / "inverse_fold"),
        output_path=str(ctx.pipeline_dir / "refold" / "af3_inputs"),
    )


def _run_refold_af3_for_paths(
    ctx: PipelineContext,
    *,
    input_json_path: Path,
    output_base_dir: Path,
) -> dict:
    """
    Run AlphaFold3 refold stage. Supports multi-GPU parallel execution.

    When multiple GPUs are available, splits the AF3 input JSON into N parts
    (one per GPU) and runs N parallel AF3 instances, each on a single GPU.
    """
    gpu_list = ctx.gpu_list

    # Load AF3 input JSON
    with open(input_json_path, "r") as f:
        all_jobs = json.load(f)
    
    if not isinstance(all_jobs, list):
        all_jobs = [all_jobs]
    
    num_gpus = len(gpu_list)
    num_jobs = len(all_jobs)
    
    if num_gpus <= 1 or num_jobs <= 1:
        # Single GPU or single job: run directly
        return ctx.refold_model.run(
            action="run_alphafold3",
            input_json=str(input_json_path),
            output_dir=str(output_base_dir),
        )
    
    # Multi-GPU parallel execution: split jobs across GPUs
    print(f"[refold] Multi-GPU mode: {num_gpus} GPUs, {num_jobs} jobs, distributing...")
    
    # Split jobs evenly across GPUs
    jobs_per_gpu = [[] for _ in range(num_gpus)]
    for i, job in enumerate(all_jobs):
        gpu_idx = i % num_gpus
        jobs_per_gpu[gpu_idx].append(job)
    
    # Create subdirectories for each GPU
    subdirs = []
    for i, gpu in enumerate(gpu_list):
        subdir = output_base_dir / f"gpu_{gpu}"
        subdir.mkdir(parents=True, exist_ok=True)
        subdirs.append((gpu, subdir))
    
    # Write split JSON files
    split_json_files = []
    for gpu_idx, (gpu_id, subdir) in enumerate(subdirs):
        split_json = subdir / "af3_inputs_split.json"
        with open(split_json, "w") as f:
            json.dump(jobs_per_gpu[gpu_idx], f, indent=2)
        split_json_files.append((gpu_id, split_json, subdir))
        print(f"[refold] GPU {gpu_id}: {len(jobs_per_gpu[gpu_idx])} jobs -> {subdir}")
    
    # Run AF3 in parallel on each GPU
    def run_af3_on_gpu(gpu_id: str, split_json: Path, output_dir: Path) -> dict:
        """Run AF3 on a single GPU."""
        print(f"[refold] Starting AF3 on GPU {gpu_id}...")
        t0 = perf_counter()
        
        # Call run_alphafold3 with single GPU
        result = ctx.refold_model.run(
            action="run_alphafold3_single_gpu",
            input_json=str(split_json),
            output_dir=str(output_dir),
            gpu=gpu_id,
        )
        
        elapsed = perf_counter() - t0
        print(f"[refold] GPU {gpu_id} finished in {elapsed:.1f}s")
        return result
    
    results = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(run_af3_on_gpu, gpu_id, split_json, output_dir): gpu_id
            for gpu_id, split_json, output_dir in split_json_files
        }
        
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                result = future.result()
                results.append((gpu_id, result))
            except Exception as e:
                print(f"[refold] ERROR: GPU {gpu_id} failed: {e}")
                raise
    
    # Consolidate results
    total_cif_files = 0
    for gpu_id, result in results:
        if isinstance(result, dict) and "details" in result:
            total_cif_files += result["details"].get("num_cif_files", 0)
    
    # Move all CIF files to the main output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    for gpu_id, subdir in subdirs:
        for cif_file in subdir.rglob("*.cif"):
            dest = output_base_dir / cif_file.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(cif_file), str(dest))
            sample_name = cif_file.stem.replace("_model", "")
            for suffix in ["_confidences.json", "_summary_confidences.json"]:
                sidecar = cif_file.parent / f"{sample_name}{suffix}"
                if sidecar.exists():
                    sidecar_dest = output_base_dir / sidecar.name
                    if sidecar_dest.exists():
                        sidecar_dest.unlink()
                    shutil.move(str(sidecar), str(sidecar_dest))
    
    result = {
        "stage": "refold.run_alphafold3_multi_gpu",
        "success": True,
        "outputs": {"output_dir": str(output_base_dir)},
        "num_gpus": num_gpus,
        "num_jobs": num_jobs,
        "results": results,
        "details": {"total_cif_files": total_cif_files},
    }
    print(f"[refold] Multi-GPU AF3 completed: {total_cif_files} CIF files in {output_base_dir}")
    return result


def _run_refold_af3(ctx: PipelineContext) -> None:
    ctx.runtime["refold"] = _run_refold_af3_for_paths(
        ctx,
        input_json_path=ctx.pipeline_dir / "refold" / "af3_inputs",
        output_base_dir=ctx.pipeline_dir / "refold" / "af3_out",
    )


def _run_refold_af3_pbp_dual(ctx: PipelineContext) -> None:
    complex_run = _run_refold_af3_for_paths(
        ctx,
        input_json_path=ctx.pipeline_dir / "refold" / "af3_inputs",
        output_base_dir=ctx.pipeline_dir / "refold" / "af3_out",
    )
    unbound_run = _run_refold_af3_for_paths(
        ctx,
        input_json_path=ctx.pipeline_dir / "refold" / "af3_unbound_inputs",
        output_base_dir=ctx.pipeline_dir / "refold" / "af3_unbound_out",
    )
    ctx.runtime["refold"] = {
        "complex": complex_run,
        "unbound": unbound_run,
    }


def _prepare_refold_chai1(ctx: PipelineContext) -> None:
    refold_cfg = _get_cfg_value(ctx.cfg, "refold", None)
    ctx.runtime["refold_prepare"] = ctx.refold_model.run(
        action="make_chai1_fasta_from_backbone_dir",
        backbone_dir=str(ctx.pipeline_dir / "inverse_fold" / "backbones"),
        output_dir=str(ctx.pipeline_dir / "refold" / "chai1_inputs"),
        ccd_path=getattr(refold_cfg, "ccd_component", None) if refold_cfg is not None else None,
        inverse_fold_dir=str(ctx.pipeline_dir / "inverse_fold"),
    )


def _run_refold_chai1(ctx: PipelineContext) -> None:
    fasta_dir = ctx.pipeline_dir / "refold" / "chai1_inputs"
    ctx.runtime["refold"] = ctx.refold_model.run(
        action="run_chai1",
        fasta_list=list(fasta_dir.glob("*.fasta")),
        output_dir=str(ctx.pipeline_dir / "refold" / "chai1_out"),
    )


def _pbl_prepare_evaluation_inputs(ctx: PipelineContext) -> None:
    # PBL inputs may already be inverse-folded CIFs. Prefer the freshly
    # preprocessed structures and fall back to the legacy inversefold dir.
    input_dir = ctx.pipeline_dir / "formatted_designs"
    legacy_input_dir = ctx.pipeline_dir / "inversefold"
    if not input_dir.exists() and legacy_input_dir.exists():
        input_dir = legacy_input_dir

    ctx.runtime["refold_prepare"] = ctx.preprocess_model.run(
        action="format_output_ligand_for_pbl_eval",
        input_dir=str(input_dir),
        output_dir=str(ctx.pipeline_dir / "inversefold_formatted_designs_for_evaluation"),
    )


def _preprocess_ame(ctx: PipelineContext) -> None:
    ame_csv_path = _resolve_metadata_csv(
        ctx,
        cfg_key="ame_csv",
        preferred_names=["ame.csv", "ame_info.csv"],
        csv_label="AME metadata CSV",
    )

    from inversefold.ame_csv_utils import load_ame_csv, print_ame_summary

    ame_csv_df = load_ame_csv(ame_csv_path)
    print_ame_summary(ame_csv_df, Path(ctx.design_dir))

    ctx.runtime["preprocess"] = ctx.preprocess_model.run(
        action="format_output_pdb",
        input_dir=ctx.design_dir,
        output_dir=str(ctx.pipeline_dir / "formatted_designs"),
    )

    ctx.runtime["ame_csv_path"] = ame_csv_path
    ctx.runtime["ame_csv_df"] = ame_csv_df


def _inversefold_ame(ctx: PipelineContext) -> None:
    ame_csv_path = _get_or_resolve_ame_csv(ctx)
    ctx.runtime["inversefold"] = _require_inversefold_model(ctx).run(
        action="ligandmpnn_distributed",
        input_dir=ctx.pipeline_dir / "formatted_designs",
        output_dir=str(ctx.pipeline_dir / "inverse_fold"),
        gpu_list=ctx.gpu_list,
        origin_cwd=ctx.origin_cwd,
        ame_csv=ame_csv_path,
        ame_csv_df=ctx.runtime["ame_csv_df"],
    )


def _prepare_refold_ame(ctx: PipelineContext) -> None:
    ame_csv_path = _get_or_resolve_ame_csv(ctx)
    ctx.runtime["refold_prepare"] = ctx.refold_model.run(
        action="make_chai1_fasta_multi_process",
        backbone_dir=str(ctx.pipeline_dir / "inverse_fold" / "backbones"),
        output_dir=str(ctx.pipeline_dir / "refold" / "chai1_inputs"),
        origin_cwd=ctx.origin_cwd,
        inverse_fold_dir=str(ctx.pipeline_dir / "inverse_fold"),
        ame_csv=ame_csv_path,
    )


def _preprocess_motif_scaffolding(ctx: PipelineContext) -> None:
    motif_cfg = _get_cfg_value(ctx.cfg, "motif_scaffolding", {}) or {}

    motif_list = None
    max_samples_per_motif = None
    if hasattr(motif_cfg, "get"):
        motif_list = motif_cfg.get("motif_list", None)
        max_samples_per_motif = motif_cfg.get("max_samples_per_motif", None)
    if motif_list is not None and isinstance(motif_list, list):
        motif_list = [str(m) for m in motif_list]
    if max_samples_per_motif is not None:
        max_samples_per_motif = int(max_samples_per_motif)

    design_dir = Path(ctx.design_dir)
    pdbs_in_root = list(design_dir.glob("*.pdb"))
    scaffold_in_root = (design_dir / "scaffold_info.csv").exists()

    if pdbs_in_root and scaffold_in_root:
        inferred_name = _infer_motif_name_from_pdbs(pdbs_in_root, design_dir.name)
        motifs_to_run = [motif_list[0]] if motif_list and len(motif_list) > 0 else [inferred_name]
        input_dirs = {m: design_dir for m in motifs_to_run}
    else:
        motifs_to_run = [str(m) for m in motif_list] if motif_list else _discover_motifs(design_dir)
        if not motifs_to_run:
            raise ValueError(
                f"No motif subdirs (with *.pdb and scaffold_info.csv) found in {design_dir}. "
                "Pass motif_list or use a single motif dir with PDBs and scaffold_info.csv."
            )
        input_dirs = {m: design_dir / m for m in motifs_to_run}
        for motif_name, motif_dir in input_dirs.items():
            if not motif_dir.exists():
                raise FileNotFoundError(f"Motif dir not found: {motif_dir}")

    motif_items = []
    for motif_name in motifs_to_run:
        source_dir = input_dirs[motif_name]
        if max_samples_per_motif is not None:
            staged_input_dir = ctx.pipeline_dir / "motif_scaffolding_inputs" / motif_name
            if staged_input_dir.exists():
                shutil.rmtree(staged_input_dir)
            input_dir = _prepare_input_dir(
                source_dir=source_dir,
                dest_dir=staged_input_dir,
                max_samples=max_samples_per_motif,
                motif_name=motif_name,
            )
        else:
            input_dir = source_dir

        motif_output_dir = ctx.pipeline_dir / "motif_scaffolding" / motif_name
        motif_output_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = input_dir / "scaffold_info.csv"
        motif_items.append(
            {
                "motif_name": motif_name,
                "input_dir": str(input_dir),
                "metadata_file": str(metadata_file) if metadata_file.exists() else None,
                "motif_output_dir": str(motif_output_dir),
            }
        )

    ctx.runtime["motif_items"] = motif_items
    ctx.runtime["preprocess"] = {"motif_count": len(motif_items)}


def _restore_motif_scaffolding_items(ctx: PipelineContext) -> list[dict]:
    motif_items = ctx.runtime.get("motif_items", [])
    if motif_items:
        return motif_items

    motif_cfg = _get_cfg_value(ctx.cfg, "motif_scaffolding", {}) or {}
    motif_list = None
    max_samples_per_motif = None
    if hasattr(motif_cfg, "get"):
        motif_list = motif_cfg.get("motif_list", None)
        max_samples_per_motif = motif_cfg.get("max_samples_per_motif", None)

    design_dir = Path(ctx.design_dir)
    pdbs_in_root = list(design_dir.glob("*.pdb"))
    scaffold_in_root = (design_dir / "scaffold_info.csv").exists()

    if pdbs_in_root and scaffold_in_root:
        motifs_to_run = [motif_list[0]] if motif_list and len(motif_list) > 0 else [design_dir.name]
        input_dirs = {m: design_dir for m in motifs_to_run}
    else:
        if motif_list:
            motifs_to_run = [str(m) for m in motif_list]
        else:
            motifs_to_run = _discover_motifs(design_dir)
        if not motifs_to_run:
            raise RuntimeError(
                f"No motif inputs found under design_dir={design_dir}; cannot restore motif items."
            )
        input_dirs = {m: design_dir / m for m in motifs_to_run}

    restored_items = []
    for motif_name in motifs_to_run:
        source_dir = input_dirs[motif_name]
        staged_input_dir = ctx.pipeline_dir / "motif_scaffolding_inputs" / motif_name
        if max_samples_per_motif is not None and staged_input_dir.exists():
            input_dir = staged_input_dir
        else:
            input_dir = source_dir

        motif_output_dir = ctx.pipeline_dir / "motif_scaffolding" / motif_name
        metadata_file = input_dir / "scaffold_info.csv"
        item = {
            "motif_name": motif_name,
            "input_dir": str(input_dir),
            "metadata_file": str(metadata_file) if metadata_file.exists() else None,
            "motif_output_dir": str(motif_output_dir),
        }

        inverse_fold_dir = motif_output_dir / "inverse_fold"
        if inverse_fold_dir.exists():
            item["inverse_fold_dir"] = str(inverse_fold_dir)

        refold_input_json = motif_output_dir / "refold" / "esmfold_inputs.json"
        if refold_input_json.exists():
            item["refold_input_json"] = str(refold_input_json)

        refold_output_dir = motif_output_dir / "refold" / "esmfold_out"
        if refold_output_dir.exists():
            item["refold_output_dir"] = str(refold_output_dir)

        restored_items.append(item)

    ctx.runtime["motif_items"] = restored_items
    return restored_items


def _inversefold_motif_scaffolding(ctx: PipelineContext) -> None:
    motif_items = _restore_motif_scaffolding_items(ctx)
    if not motif_items:
        raise RuntimeError("motif_scaffolding preprocess outputs not found; run preprocess first")

    results = {}
    for item in motif_items:
        motif_name = item["motif_name"]
        motif_output_dir = Path(item["motif_output_dir"])
        inverse_fold_dir = motif_output_dir / "inverse_fold"
        run_kwargs = {
            "action": "proteinmpnn_distributed",
            "input_dir": Path(item["input_dir"]),
            "output_dir": str(inverse_fold_dir),
            "gpu_list": ctx.gpu_list,
            "origin_cwd": ctx.origin_cwd,
        }
        metadata_file = item.get("metadata_file")
        if metadata_file:
            metadata_path = Path(str(metadata_file))
            if metadata_path.exists():
                run_kwargs["scaffold_info_csv"] = str(metadata_path)
        results[motif_name] = _require_inversefold_model(ctx).run(
            **run_kwargs,
        )
        item["inverse_fold_dir"] = str(inverse_fold_dir)
    ctx.runtime["inversefold"] = results


def _prepare_refold_motif_scaffolding(ctx: PipelineContext) -> None:
    motif_items = _restore_motif_scaffolding_items(ctx)
    if not motif_items:
        raise RuntimeError("motif_scaffolding preprocess outputs not found; run preprocess first")

    results = {}
    for item in motif_items:
        motif_name = item["motif_name"]
        motif_output_dir = Path(item["motif_output_dir"])
        refold_dir = motif_output_dir / "refold"
        refold_input_json = refold_dir / "esmfold_inputs.json"
        results[motif_name] = ctx.refold_model.run(
            action="make_esmfold_json_multi_process",
            backbone_dir=str(Path(item["inverse_fold_dir"]) / "backbones"),
            output_dir=str(refold_input_json),
        )
        item["refold_input_json"] = str(refold_input_json)
    ctx.runtime["refold_prepare"] = results


def _run_refold_motif_scaffolding(ctx: PipelineContext) -> None:
    motif_items = _restore_motif_scaffolding_items(ctx)
    if not motif_items:
        raise RuntimeError("motif_scaffolding preprocess outputs not found; run preprocess first")

    results = {}
    for item in motif_items:
        motif_name = item["motif_name"]
        refold_output_dir = Path(item["motif_output_dir"]) / "refold" / "esmfold_out"
        results[motif_name] = ctx.refold_model.run(
            action="run_esmfold",
            sequences_file_json=str(item["refold_input_json"]),
            output_dir=str(refold_output_dir),
        )
        item["refold_output_dir"] = str(refold_output_dir)
    ctx.runtime["refold"] = results


def _generate_motif_scaffolding_summaries(ctx: PipelineContext) -> None:
    script_path = (
        Path(__file__).resolve().parent.parent
        / "evaluation"
        / "motif_scaffolding"
        / "scripts"
        / "write_summaries.py"
    )
    if not script_path.exists():
        print(f"[unified] Warning: motif summary script not found: {script_path}")
        return

    motif_cfg = _get_cfg_value(ctx.cfg, "motif_scaffolding", {}) or {}
    python_path = None
    if hasattr(motif_cfg, "get"):
        python_path = motif_cfg.get("python_path", None)
    if not python_path:
        python_path = sys.executable

    test_cases_path = None
    if hasattr(motif_cfg, "get"):
        configured_test_cases = motif_cfg.get("test_cases_csv", None)
    else:
        configured_test_cases = getattr(motif_cfg, "test_cases_csv", None)
    if configured_test_cases:
        candidate = Path(str(configured_test_cases))
        if candidate.exists():
            test_cases_path = candidate
        else:
            print(f"[unified] Warning: motif_scaffolding.test_cases_csv not found: {candidate}")

    cmd = [str(python_path), str(script_path), str(ctx.pipeline_dir)]
    if test_cases_path is not None:
        cmd.extend(["--test-cases", str(test_cases_path)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[unified] Warning: motif summary generation failed: {result.stderr}")


def _plugin_motif_scaffolding_eval(ctx: PipelineContext) -> None:
    motif_items = _restore_motif_scaffolding_items(ctx)
    if not motif_items:
        raise RuntimeError("motif_scaffolding preprocess outputs not found; run preprocess first")

    eval_results = {}
    for item in motif_items:
        motif_name = item["motif_name"]
        motif_output_dir = Path(item["motif_output_dir"])
        eval_results[motif_name] = ctx.evaluation_model.run(
            task="motif_scaffolding",
            input_dir=item["input_dir"],
            output_dir=str(motif_output_dir),
            metadata_file=item["metadata_file"],
            motif_name=motif_name,
            refold_dir=str(Path(item["refold_output_dir"])),
            skip_pipeline_stages=True,
        )
    ctx.runtime["evaluation_motif_scaffolding"] = eval_results
    _generate_motif_scaffolding_summaries(ctx)


TASK_SPECS: dict[str, TaskSpec] = {
    "protein": TaskSpec(
        preprocess_stage=_preprocess_protein,
        inversefold_stage=_inversefold_proteinmpnn,
        refold_prepare_stage=_prepare_refold_esm,
        refold_stage=_run_refold_esm,
        evaluation_plugins=[_plugin_protein_eval],
    ),
    "pbp": TaskSpec(
        preprocess_stage=_preprocess_protein,
        inversefold_stage=_inversefold_pbp,
        refold_prepare_stage=_prepare_refold_af3_pbp_dual,
        refold_stage=_run_refold_af3_pbp_dual,
        evaluation_plugins=[_plugin_pbp_eval],
    ),
    "lbp": TaskSpec(
        preprocess_stage=_preprocess_ligand,
        inversefold_stage=_inversefold_lbp,
        refold_prepare_stage=_prepare_refold_chai1,
        refold_stage=_run_refold_chai1,
        evaluation_plugins=[_plugin_lbp_eval],
    ),
    "interface": TaskSpec(
        preprocess_stage=_preprocess_interface,
        inversefold_stage=_inversefold_interface,
        refold_prepare_stage=_prepare_refold_chai1,
        refold_stage=_run_refold_chai1,
        evaluation_plugins=[_plugin_interface_eval],
    ),
    "nuc": TaskSpec(
        preprocess_stage=_preprocess_cif,
        inversefold_stage=_inversefold_passthrough_formatted_designs,
        refold_prepare_stage=_prepare_refold_af3_from_inverse_fold,
        refold_stage=_run_refold_af3,
        evaluation_plugins=[_plugin_nuc_eval],
    ),
    "nbl": TaskSpec(
        preprocess_stage=_preprocess_ligand,
        inversefold_stage=_inversefold_odesign_to_inverse_fold,
        refold_prepare_stage=_prepare_refold_af3_from_inverse_fold,
        refold_stage=_run_refold_af3,
        evaluation_plugins=[_plugin_nbl_eval],
    ),
    "pbn": TaskSpec(
        preprocess_stage=_preprocess_cif,
        inversefold_stage=_inversefold_odesign_to_inverse_fold,
        refold_prepare_stage=_prepare_refold_af3_from_inverse_fold,
        refold_stage=_run_refold_af3,
        evaluation_plugins=[_plugin_pbn_eval],
    ),
    "pbl": TaskSpec(
        preprocess_stage=_preprocess_ligand,
        inversefold_stage=None,
        refold_prepare_stage=_pbl_prepare_evaluation_inputs,
        refold_stage=None,
        evaluation_plugins=[_plugin_pbl_eval],
    ),
    "ame": TaskSpec(
        preprocess_stage=_preprocess_ame,
        inversefold_stage=_inversefold_ame,
        refold_prepare_stage=_prepare_refold_ame,
        refold_stage=_run_refold_chai1,
        evaluation_plugins=[_plugin_ame_eval, _plugin_ame_statistics],
    ),
    "motif_scaffolding": TaskSpec(
        preprocess_stage=_preprocess_motif_scaffolding,
        inversefold_stage=_inversefold_motif_scaffolding,
        refold_prepare_stage=_prepare_refold_motif_scaffolding,
        refold_stage=_run_refold_motif_scaffolding,
        evaluation_plugins=[_plugin_motif_scaffolding_eval],
    ),
}


def run_unified_pipeline(cfg: object, task_name: str) -> PipelineContext:
    """Execute a task using unified stages + task-specific evaluation plugins."""
    if task_name not in TASK_SPECS:
        known = ", ".join(sorted(TASK_SPECS.keys()))
        raise ValueError(f"Unknown task_name '{task_name}'. Supported: {known}")
    spec = TASK_SPECS[task_name]

    gpu_list = _gpus_to_list(getattr(cfg, "gpus"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)

    pipeline_dir = _resolve_pipeline_dir(cfg)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    design_dir = _resolve_design_dir(cfg)
    origin_cwd = get_original_cwd()

    if not os.path.exists(design_dir):
        raise FileNotFoundError(f"design_dir not found: {design_dir}")

    inversefold_model = None
    if spec.inversefold_stage not in (None, _inversefold_passthrough_formatted_designs):
        from inversefold.inversefold_api import InverseFold

        inversefold_model = InverseFold(cfg)

    ctx = PipelineContext(
        cfg=cfg,
        task_name=task_name,
        pipeline_dir=pipeline_dir,
        design_dir=design_dir,
        origin_cwd=origin_cwd,
        gpu_list=gpu_list,
        preprocess_model=Preprocess(cfg),
        inversefold_model=inversefold_model,
        refold_model=ReFold(cfg),
        evaluation_model=Evaluation(cfg),
        runtime={},
    )

    print("=" * 80)
    print(f"[unified] Task: {task_name}")
    print(f"[unified] Pipeline dir: {pipeline_dir}")
    print(f"[unified] Design dir: {design_dir}")
    print("=" * 80)

    pipeline_t0 = perf_counter()
    pipeline_success = False
    pipeline_error = None

    try:
        _run_stage(ctx, "preprocess", spec.preprocess_stage)
        _run_stage(ctx, "inversefold", spec.inversefold_stage)
        _run_stage(ctx, "refold_prepare", spec.refold_prepare_stage)
        _run_stage(ctx, "refold", spec.refold_stage)

        if _stage_enabled(cfg, "evaluation"):
            print("[unified] Running evaluation plugins")
            for plugin in spec.evaluation_plugins:
                plugin_name = plugin.__name__
                t0 = perf_counter()
                error_msg = None
                try:
                    plugin(ctx)
                except Exception as exc:
                    error_msg = str(exc)
                    raise
                finally:
                    elapsed = perf_counter() - t0
                    timings = ctx.runtime.setdefault("timings", {})
                    key = f"evaluation:{plugin_name}"
                    timings[key] = {
                        "elapsed_seconds": round(elapsed, 3),
                        "success": error_msg is None,
                    }
                    if error_msg is not None:
                        timings[key]["error"] = error_msg
                    print(
                        f"[unified] Evaluation plugin '{plugin_name}' finished in {elapsed:.2f}s"
                        f"{' (failed)' if error_msg is not None else ''}"
                    )
        else:
            print("[unified] Skip stage 'evaluation' (disabled by config)")

        pipeline_success = True
    except Exception as exc:
        pipeline_error = str(exc)
        raise
    finally:
        total_elapsed = perf_counter() - pipeline_t0
        ctx.runtime.setdefault("timings", {})["pipeline_total"] = {
            "elapsed_seconds": round(total_elapsed, 3),
            "success": pipeline_success,
        }
        if pipeline_error is not None:
            ctx.runtime["timings"]["pipeline_total"]["error"] = pipeline_error

        summary_path = _write_runtime_summary(
            ctx=ctx,
            total_elapsed=total_elapsed,
            success=pipeline_success,
            error_msg=pipeline_error,
        )
        print(f"[unified] Total runtime: {total_elapsed:.2f}s")
        print(f"[unified] Runtime summary written to: {summary_path}")
        if pipeline_success:
            print("[unified] Pipeline completed")
        else:
            print("[unified] Pipeline failed")

    return ctx
