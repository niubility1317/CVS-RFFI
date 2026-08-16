"""Frozen CCOC Hard9+K1 matrix and method-lock validation.

This module owns only the immutable experiment identity and its sealed-package
joins. It never opens query truth, scores predictions, or changes the CCOC
scientific implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from cvsrffi.stage2_d92_e0_full_only_target125 import (
    CONTEXT_SHA256,
    GROUND_COMPONENT_DIR,
    GROUND_MANIFEST_PATH,
    GROUND_MANIFEST_SHA256,
    SCENES,
    SOURCE_D92_OUTPUT_ROOT,
)


MATRIX_SCHEMA = "cvs.phase2.d92_ccoc_hard9_k1.matrix.v1"
METHOD_LOCK_SCHEMA = "cvs.phase2.d92_ccoc_hard9_k1.method_lock.v1"
JOB_RECEIPT_SCHEMA = "cvs.phase2.d92_ccoc_hard9_k1.job_receipt.v1"
SHARD_SUMMARY_SCHEMA = "cvs.phase2.d92_ccoc_hard9_k1.shard_summary.v1"
SYSTEMIC_FAILURE_SCHEMA = "cvs.phase2.d92_ccoc_hard9_k1.systemic_failure.v1"

ARM_ID = "E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"
CANDIDATE_ID = "d92_e0_full_cross_class_offblock_consensus"
ARM_ORDER = (ARM_ID,)
ARM_CANDIDATE_IDS = {ARM_ID: CANDIDATE_ID}
ARM_ROLES = {ARM_ID: "primary"}
PRIMARY_ARM = ARM_ID
REGISTERED_MODE = "ccoc_full"
STATE_POSTPROCESS_MODE = "ccoc_full"
CLAIM_SCOPE = "DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN"
SHARD_COUNT = 8
SMOKE_OUTER_KEY = "rx_7_7__seed_713104__k_5__new_20"
LIVENESS_OUTER_KEY = "rx_20_1__seed_713106__k_1__new_20"
G0_OUTER_KEY = "rx_7_7__seed_713106__k_10__new_5"
EXCLUDED_OUTER_KEYS = (G0_OUTER_KEY,)

HARD9_K1_ROWS = (
    {
        "outer_key": "rx_7_7__seed_713104__k_5__new_20",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": "rx_7_7__seed_713103__k_10__new_5",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": "rx_8_8__seed_713103__k_5__new_20",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": "rx_8_8__seed_713103__k_10__new_5",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": "rx_8_8__seed_713106__k_5__new_20",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": "rx_7_14__seed_713104__k_10__new_10",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": "rx_3_19__seed_713102__k_10__new_5",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": "rx_7_7__seed_713105__k_10__new_20",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": "rx_7_7__seed_713104__k_10__new_5",
        "role": "performance",
        "hard_score": None,
    },
    {
        "outer_key": LIVENESS_OUTER_KEY,
        "role": "liveness",
        "hard_score": None,
    },
)
HARD9_K1_V1_ROWS = HARD9_K1_ROWS

QUERY_CONTRACT = {
    "decision": "per_sample_all_registered_classes",
    "truth_access": False,
    "fit_access": False,
    "update_access": False,
    "selection_access": False,
    "role_oracle_access": False,
    "class_quota_access": False,
    "global_reassignment": False,
}
QUERY_ZERO_FIELDS = (
    "query_truth_access",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)
FIT_GATE = {
    "k_gt_2_total": 2,
    "k_gt_2_actual": 1,
    "postprocess_fit": 0,
    "k1_alias": "K1_K2_EXACT_D92_FULL_ALIAS",
    "k1_total": 3,
    "k1_actual": 3,
}
RESOURCE_GATE = {
    "registration_wall_p90_max_ns": 150_000_000,
    "registration_wall_ratio_max": 1.5,
    "candidate_peak_hard_max_bytes": 1024 * 1024,
    "registration_wall_p90_target_max_ns": 120_000_000,
    "registration_wall_ratio_target_max": 1.25,
    "candidate_peak_target_max_bytes": 512 * 1024,
    "query_macs_equal": True,
    "state_bytes_equal": True,
}
STOP_RULE = {
    "same_normalized_exception_fingerprint_distinct_outer_count": 2,
    "pre_prediction_only": True,
    "shared_run_root_ledger": True,
    "fresh_run_retry_authorized": False,
}
PACKAGE_LAYOUT = {
    "before_enrollment": (
        ("offline", "predictor", "before", "enrollment_only"),
        ("offline", "seals", "before_enrollment.seal.json"),
    ),
    "before_apply": (
        ("offline", "predictor", "before", "apply_only_staging"),
        ("apply_seals", "before_apply.seal.json"),
    ),
    "after_enrollment": (
        ("offline", "predictor", "after", "enrollment_only"),
        ("offline", "seals", "after_enrollment.seal.json"),
    ),
    "after_apply": (
        ("offline", "predictor", "after", "apply_only_staging"),
        ("apply_seals", "after_apply.seal.json"),
    ),
}

HISTORICAL_BASELINE_PATH = (
    "E:/type10-7/local_artifacts/d92_e0_full_only_target125_20260812_v1/"
    "analysis/paired_rows.csv"
)
HISTORICAL_BASELINE_SHA256 = (
    "6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a"
)
HISTORICAL_PER_OLD_CLASS_PATH = (
    "E:/type10-7/local_artifacts/d92_e0_full_only_target125_20260812_v1/"
    "analysis/per_old_class_rows.csv"
)
HISTORICAL_PER_OLD_CLASS_SHA256 = (
    "c0fc1e02b66b01d06da68bdd824594f3281e601d72b32726fa1e97a1e49788e6"
)
RAW_SCORE_ROOT = (
    "E:/type10-7/local_artifacts/d92_e0_full_only_target125_20260812_v1/"
    "output/jobs"
)
RAW_SCORE_SHA = {
    "rx_7_7__seed_713104__k_5__new_20": (
        "492044d89de05fbee79bfd6ca493c51778e2f2b18536038067c329acedd7cee9"
    ),
    "rx_7_7__seed_713103__k_10__new_5": (
        "f8f593fe5b26983ae16a7903f3943cd07fb9e0e958beea3c142a724119f7c93b"
    ),
    "rx_8_8__seed_713103__k_5__new_20": (
        "00b217da83ffce70655360ce243ad88e37ad1e1a221980488cbb04655b091306"
    ),
    "rx_8_8__seed_713103__k_10__new_5": (
        "69ab6c617db8f657c4a21d044049984c5913b5dd0af7a76456877564f031bd32"
    ),
    "rx_8_8__seed_713106__k_5__new_20": (
        "9a42a6306669811cda5b058fa342619abf4ef20c01d499fb682d6c4700d5a360"
    ),
    "rx_7_14__seed_713104__k_10__new_10": (
        "953e9bccfad63e5e5ca7b7b87e5f48d458318b02c069d4db8c47a5d083087dd0"
    ),
    "rx_3_19__seed_713102__k_10__new_5": (
        "6488c4f516e41703cd529d6e4837d0ef0e1fe4eae008fec3beee4cf56cee7bc3"
    ),
    "rx_7_7__seed_713105__k_10__new_20": (
        "bee2068990f890cb4233834f1f4ccfb1cfb6d8ed67094d66031c0aa00323712d"
    ),
    "rx_7_7__seed_713104__k_10__new_5": (
        "01384fedde246cb017773f69516700bd3bb7a15459b7863d417cc9d2ecc602c1"
    ),
    LIVENESS_OUTER_KEY: (
        "bf60d1231127c51b9a9dbe06c9c78bbad7bfd34d0b2ffc5c7809dc94d47677f2"
    ),
}

# These ten digests were sealed by the same-outer E0 raw-score artifacts whose
# paths and file hashes are already frozen in the method lock above.  Matrix
# preparation copies only this opaque metadata; it never opens query truth.
PREREGISTERED_TRUTH_SIDECAR_SHA256 = {
    "rx_7_7__seed_713104__k_5__new_20": (
        "0ea2f8471e3632545cda52f3e0879fc276237f263885ba8a14d74b45b4b84237"
    ),
    "rx_7_7__seed_713103__k_10__new_5": (
        "83e143cac2104bd610e8ed83968a7dff2bff5bab1ba84da479ee7e2fa13481ce"
    ),
    "rx_8_8__seed_713103__k_5__new_20": (
        "7c1340020d7240e50771e104f623c2035de709f87ab15a041701b88d13eac3ea"
    ),
    "rx_8_8__seed_713103__k_10__new_5": (
        "aa3aa1eb05fd4b7781b59d48e8173cbc8304c4ff390e110e1f359cf2b5049f0b"
    ),
    "rx_8_8__seed_713106__k_5__new_20": (
        "f8e74dd1491f8e0a4fd70c0ea9ac55bfab1d185a098a25bcb7757336a3e84924"
    ),
    "rx_7_14__seed_713104__k_10__new_10": (
        "e8ae7fd3ef3369bc96b72c4c18834259cd24ac4056fdc015f2c6734f3b4f08b2"
    ),
    "rx_3_19__seed_713102__k_10__new_5": (
        "0d27fa7794d9f8f10cc4e50771b2944f2f4d948958a342e1fbdca6288af53176"
    ),
    "rx_7_7__seed_713105__k_10__new_20": (
        "75ce467e4dbe35fd2ff40475f2bac606d7c7cf4c9c30dbe0ec674fb5c4967190"
    ),
    "rx_7_7__seed_713104__k_10__new_5": (
        "e1dec138a60795619248a6b352614aff4291b3d08088c1723fe77acee0a689eb"
    ),
    LIVENESS_OUTER_KEY: (
        "b6fc53dc3a02b0867084a1146e4f23fc40ca543b726da3cb54db587f59ec621d"
    ),
}
E0_RESOURCE_OUTPUT_ROOT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "d92_e0_full_only_target125_20260812_v1/output"
)


def _e0_resource_row(
    outer_key: str,
    fit_audit_sha256: str,
    *,
    clear: tuple[int, int, int, int],
    low_elev: tuple[int, int, int, int],
    rain: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Freeze the same-outer E0 fit-audit resource projection by scene."""

    def scene_record(values: tuple[int, int, int, int]) -> dict[str, int]:
        return {
            "registration_wall_time_ns": values[0],
            "registration_incremental_peak_working_set_bytes": values[1],
            "query_macs": values[2],
            "state_bytes": values[3],
        }

    return {
        "fit_audit": {
            "path": (
                f"{E0_RESOURCE_OUTPUT_ROOT}/jobs/{outer_key}/E0_FULL_ONLY/"
                "diag/after/fit_audit.json"
            ),
            "sha256": fit_audit_sha256,
        },
        "scenes": {
            "leo_clear_weak": scene_record(clear),
            "leo_low_elev_weak": scene_record(low_elev),
            "leo_rain_weak": scene_record(rain),
        },
    }


E0_RESOURCE_ROWS = {
    "rx_7_7__seed_713104__k_5__new_20": _e0_resource_row(
        "rx_7_7__seed_713104__k_5__new_20",
        "e47b9f0f015d121bf594a899c6591a33fbcaef19e8e6926bc4a9a96fff3c4712",
        clear=(95_671_833, 2_289_664, 7_488, 18_498),
        low_elev=(94_819_902, 1_650_688, 7_488, 18_498),
        rain=(93_753_108, 536_576, 7_488, 18_498),
    ),
    "rx_7_7__seed_713103__k_10__new_5": _e0_resource_row(
        "rx_7_7__seed_713103__k_10__new_5",
        "4cd5d435e7c1704afcfbfe0f4f97df2ef6587c8e086398145a95c7f530471371",
        clear=(67_552_260, 1_323_008, 3_168, 8_583),
        low_elev=(62_632_550, 1_036_288, 3_168, 8_583),
        rain=(66_908_410, 282_624, 3_168, 8_583),
    ),
    "rx_8_8__seed_713103__k_5__new_20": _e0_resource_row(
        "rx_8_8__seed_713103__k_5__new_20",
        "e6d2b3341dc7bb4ca75fe2236a5bdcc9d9f2822b4daa84ac8d15b4449c9841cc",
        clear=(103_676_788, 2_306_048, 7_488, 18_498),
        low_elev=(101_870_949, 1_646_592, 7_488, 18_498),
        rain=(101_377_578, 81_920, 7_488, 18_498),
    ),
    "rx_8_8__seed_713103__k_10__new_5": _e0_resource_row(
        "rx_8_8__seed_713103__k_10__new_5",
        "4da713d052e2a60f26baf776b95550084d9de92170c720a7dc7d94c7885e1305",
        clear=(66_023_869, 1_323_008, 3_168, 8_583),
        low_elev=(68_134_836, 1_040_384, 3_168, 8_583),
        rain=(68_161_877, 299_008, 3_168, 8_583),
    ),
    "rx_8_8__seed_713106__k_5__new_20": _e0_resource_row(
        "rx_8_8__seed_713106__k_5__new_20",
        "49a046b109a38c32c2dcc09acbbe56383c7cf6c1135246cb41d7bf679fc869c4",
        clear=(107_504_136, 2_301_952, 7_488, 18_498),
        low_elev=(107_694_729, 1_646_592, 7_488, 18_498),
        rain=(106_661_273, 159_744, 7_488, 18_498),
    ),
    "rx_7_14__seed_713104__k_10__new_10": _e0_resource_row(
        "rx_7_14__seed_713104__k_10__new_10",
        "e397c13e51be450d0d3a34b2d8a5ee8f61d80cf51e0547aca9df9be84e7b0c57",
        clear=(69_537_319, 2_162_688, 4_608, 11_888),
        low_elev=(73_730_329, 1_253_376, 4_608, 11_888),
        rain=(68_828_579, 593_920, 4_608, 11_888),
    ),
    "rx_3_19__seed_713102__k_10__new_5": _e0_resource_row(
        "rx_3_19__seed_713102__k_10__new_5",
        "069b7a647dc5a774827b4d21f03bfaab72cffdff2c8de252933abb6f9024d8df",
        clear=(57_259_742, 1_327_104, 3_168, 8_583),
        low_elev=(57_753_489, 1_183_744, 3_168, 8_583),
        rain=(57_241_044, 434_176, 3_168, 8_583),
    ),
    "rx_7_7__seed_713105__k_10__new_20": _e0_resource_row(
        "rx_7_7__seed_713105__k_10__new_20",
        "d221da20b4382a70e4f416d5d7cee971a94f16536641903b5afc62ac833d43e7",
        clear=(109_990_099, 2_842_624, 7_488, 18_498),
        low_elev=(108_896_424, 1_978_368, 7_488, 18_498),
        rain=(99_478_682, 1_232_896, 7_488, 18_498),
    ),
    "rx_7_7__seed_713104__k_10__new_5": _e0_resource_row(
        "rx_7_7__seed_713104__k_10__new_5",
        "429635cd5841773dc7260af168a0be03d8e2316eb995477b2cb2084973f52efc",
        clear=(62_909_104, 1_318_912, 3_168, 8_583),
        low_elev=(64_490_445, 1_310_720, 3_168, 8_583),
        rain=(64_522_045, 401_408, 3_168, 8_583),
    ),
    LIVENESS_OUTER_KEY: _e0_resource_row(
        LIVENESS_OUTER_KEY,
        "30cf647066b2972dcd6c3187f0d5c8258491a4ae7d6d3ac5985af30749597127",
        clear=(10_020_321, 2_043_904, 7_488, 18_503),
        low_elev=(10_528_416, 991_232, 7_488, 18_503),
        rain=(9_709_379, 81_920, 7_488, 18_503),
    ),
}

SCIENTIFIC_ENTRY_COMMIT = "930c5d644c323bab94deece9a08fdfb09f565399"
RUNTIME_SOURCE_FILES = {
    "scripts/run_d92_e0d_prediction.py": {
        "git_blob": "02dc1e01684dcde8089a0243587b965d617bc3b8",
        "sha256": "0f7c1c1866c6d84409068c3db160ee6084a9838012f2ef93de1b8e4fefc3f30c",
    },
    "scripts/score_d92_be_prediction.py": {
        "git_blob": "5cac3560bc812f56c4718166e3509c76c2904894",
        "sha256": "a33ba598313ac953d3d02b57b3cdfe4409a441307d2f7b0eb86e0de749017001",
    },
    "scripts/probe_d92_registration_balanced_covariance.py": {
        "git_blob": "68c61c508b9179c7e272819565acaad81f002aa8",
        "sha256": "e173ea9172b8be2763bf109bd8f27a37a2e6fc8972614519883469d30e7b6766",
    },
    "scripts/probe_d81_ground_nuisance_cauchy_center.py": {
        "git_blob": "b20a4f215aca116edc8555307e6ec1489761ab14",
        "sha256": "36acb149c37ce37787cd85ab33f770fe265fd9d985c4e088c48615af718e5d99",
    },
    "scripts/run_d92_e0ocf_hard12v3.py": {
        "git_blob": "e41ce24db1de2b09fcb66a2b61f7e7936e7866e0",
        "sha256": "d201c662c283980c9882f1b940ac190b0cc5f4b46855314df4febe1737923693",
    },
    "cvsrffi/stage2_d92_e0d_query_evaluation.py": {
        "git_blob": "c3491e3359d5da5dd9291a44dd5169bfb295e877",
        "sha256": "0512ab85a678e24fb3d9a7df3eb73506452976e25a92bdcf57591f0af17a0399",
    },
    "cvsrffi/stage2_d92_e0d_slim.py": {
        "git_blob": "1cf2b527e514729e14685527a0dc91bf15c5c4e5",
        "sha256": "8dd9864f27ab19fc0f779f48287ec42b42d3cf6c84af587b63766df7fe3464ef",
    },
    "cvsrffi/stage2_d92_cross_class_offblock_consensus.py": {
        "git_blob": "9ed699a293bb1a4ab33d611f1d8ba8284f3c673f",
        "sha256": "e59eb726d531f1d0219f982c3e08f1590f5f5a623ff01b8ac3ba2353f28f04d2",
    },
    "cvsrffi/stage2_d92_registration_balanced_covariance.py": {
        "git_blob": "578fcae4c1d1a72e9cd9fec36b3d2073047942f5",
        "sha256": "ebea42f67df045223b5d50db6f5d6074f64cd1018a3caa6dcfa0328b2896bba0",
    },
    "cvsrffi/stage2_registration_resource_probe.py": {
        "git_blob": "e1115fda0fc7015d24a7add1ec457b0a504d5c58",
        "sha256": "184ba2a798b663571a1de08ffa067f8d37d22e28a87df718331bbf66aa02561c",
    },
    "cvsrffi/stage2_d42_unified_shrinkage_lda.py": {
        "git_blob": "ee7e4150a99fd331266654d0119ee867e38f55a8",
        "sha256": "b78715909e2bed57036be0f6a84073b09e830c830fa3953fa1b092464e401f35",
    },
    "cvsrffi/stage2_d38_strong_b3_quantized.py": {
        "git_blob": "1e45e0d383f72135437961549abef8a992209fbb",
        "sha256": "1781cc83ce353481ebb09df1dab2a5d413955098f442367a969eb583e4b50ec9",
    },
    "cvsrffi/stage2_d92_cauchy_scatter_oas.py": {
        "git_blob": "cfdd10b6892becd2a94d9874f69ec073035bb572",
        "sha256": "0f00d4b2e243a5c346e01c988e93ba18119e887ba2f47de722c5c62730a03164",
    },
    "cvsrffi/stage2_d81_query_evaluation.py": {
        "git_blob": "4723429aee9dbe984e11d50ba5a7688c3ece201a",
        "sha256": "d6e40695fa94f3d738d871e27698bc26d0eb5dbb346fd49d32269ac7cd659e80",
    },
    "cvsrffi/stage2_diag_cosine_scorer.py": {
        "git_blob": "867788adc8d355915a83c1581df81594fcdc4b42",
        "sha256": "22d5555fa3e80f3b8da508011ff7a56993c57ac6f17591cd6f35e9b5d64b9cba",
    },
    "cvsrffi/somph_predictor_bundle.py": {
        "git_blob": "a95cd5b566e92763b59ff82fc859553581c582de",
        "sha256": "1c68a9e06530a8ff9b3aadaa2743d419a0c454b0cab95c5a6d52b9c658696bb4",
    },
    "cvsrffi/stage2_d92_e0ocf_hard12.py": {
        "git_blob": "078b22d02b757c09dd651ba75ffc5b3b737ed04b",
        "sha256": "65998e84d7b95cff30d9e37976c0f6d757270fa9464574008a2e26836e7e06c5",
    },
}

_OUTER_PATTERN = re.compile(
    r"^rx_(?P<receiver>[0-9_]+)__seed_(?P<seed>[0-9]+)"
    r"__k_(?P<k>[0-9]+)__new_(?P<new>[0-9]+)$"
)


class D92CCOCHard9K1Error(ValueError):
    """Raised when the frozen CCOC Hard9+K1 identity drifts."""


D92CCOCHard9K1MatrixError = D92CCOCHard9K1Error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value.lower()) is not None


def _pure_path(value: Any) -> PurePath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise D92CCOCHard9K1Error("path identity drift")
    if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", value) or "\\" in value:
        return PureWindowsPath(value)
    return PurePosixPath(value)


def _path_matches(actual: Any, root: Any, *parts: str) -> bool:
    try:
        actual_path = _pure_path(actual)
        root_path = _pure_path(root)
    except D92CCOCHard9K1Error:
        return False
    return type(actual_path) is type(root_path) and actual_path == root_path.joinpath(*parts)


def _parse_outer(key: str) -> tuple[str, int, int, int]:
    match = _OUTER_PATTERN.fullmatch(str(key))
    if match is None:
        raise D92CCOCHard9K1Error(f"invalid outer key: {key}")
    return (
        match.group("receiver").replace("_", "-"),
        int(match.group("seed")),
        int(match.group("k")),
        int(match.group("new")),
    )


def _expected_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in HARD9_K1_ROWS:
        receiver, seed, k_shot, new_class_count = _parse_outer(row["outer_key"])
        rows.append(
            {
                "outer_key": row["outer_key"],
                "outer_role": row["role"],
                "hard_score": None,
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "new_class_count": new_class_count,
            }
        )
    return rows


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outer_count": len(rows),
        "scene_count": len(SCENES),
        "scene_row_count": len(rows) * len(SCENES),
        "receiver_counts": dict(
            sorted(Counter(str(row["receiver"]) for row in rows).items())
        ),
        "seed_counts": dict(
            sorted(Counter(str(row["seed"]) for row in rows).items())
        ),
        "slice_counts": dict(
            sorted(
                Counter(
                    f"K{int(row['k_shot'])}_new{int(row['new_class_count'])}"
                    for row in rows
                ).items()
            )
        ),
        "performance_outer_count": 9,
        "liveness_outer_count": 1,
    }


def _selection_payload() -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.d92_ccoc_hard9_k1.selection.v1",
        "selection_id": "D92-E0-FULL-CCOC-Hard9-K1-v1",
        "protocol_schema": "p2_min_v1",
        "claim_scope": CLAIM_SCOPE,
        "candidate": {"arm_id": ARM_ID, "candidate_id": CANDIDATE_ID},
        "order": "explicit_pre_registered_performance_then_k1_liveness",
        "outer_rows": [dict(row) for row in HARD9_K1_ROWS],
        "excluded_outer_keys": list(EXCLUDED_OUTER_KEYS),
        "resource_gate": dict(RESOURCE_GATE),
        "coverage": {
            "outer_count": 10,
            "performance_outer_count": 9,
            "liveness_outer_count": 1,
            "scene_count": 3,
            "scene_row_count": 30,
            "shard_count": SHARD_COUNT,
        },
    }


def canonical_selection_sha256() -> str:
    """Return the deterministic, frozen selection identity."""

    return hashlib.sha256(_canonical_bytes(_selection_payload())).hexdigest()


CANONICAL_SELECTION_SHA256 = canonical_selection_sha256()


def _expected_lock() -> dict[str, Any]:
    return {
        "schema": METHOD_LOCK_SCHEMA,
        "matrix_schema": MATRIX_SCHEMA,
        "job_receipt_schema": JOB_RECEIPT_SCHEMA,
        "shard_summary_schema": SHARD_SUMMARY_SCHEMA,
        "systemic_failure_schema": SYSTEMIC_FAILURE_SCHEMA,
        "experiment_id": "D92-E0-FULL-CCOC-Hard9-K1-v1",
        "protocol_schema": "p2_min_v1",
        "claim_scope": CLAIM_SCOPE,
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "arms": {
            ARM_ID: {
                "candidate_id": CANDIDATE_ID,
                "role": "primary",
                "registered_mode": REGISTERED_MODE,
            }
        },
        "primary_arm": ARM_ID,
        "registered_mode": REGISTERED_MODE,
        "state_postprocess_mode": STATE_POSTPROCESS_MODE,
        "smoke_outer_key": SMOKE_OUTER_KEY,
        "liveness_outer_key": LIVENESS_OUTER_KEY,
        "excluded_outer_keys": list(EXCLUDED_OUTER_KEYS),
        "query_contract": dict(QUERY_CONTRACT),
        "matrix": {
            "outer_count": 10,
            "performance_outer_count": 9,
            "liveness_outer_count": 1,
            "job_count": 10,
            "scene_count": 3,
            "scene_arm_count": 30,
            "shard_count": SHARD_COUNT,
        },
        "fit_gate": dict(FIT_GATE),
        "resource_gate": dict(RESOURCE_GATE),
        "sealed_inputs": {
            "context_sha256": CONTEXT_SHA256,
            "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT,
            "ground_component_dir": GROUND_COMPONENT_DIR,
            "ground_manifest_path": GROUND_MANIFEST_PATH,
            "ground_manifest_sha256": GROUND_MANIFEST_SHA256,
            "package_layout": {
                name: {
                    "package_relative_path": list(package_parts),
                    "seal_relative_path": list(seal_parts),
                }
                for name, (package_parts, seal_parts) in PACKAGE_LAYOUT.items()
            },
        },
        "historical_baseline": {
            "paired_rows_path": HISTORICAL_BASELINE_PATH,
            "paired_rows_sha256": HISTORICAL_BASELINE_SHA256,
            "per_old_class_rows_path": HISTORICAL_PER_OLD_CLASS_PATH,
            "per_old_class_rows_sha256": HISTORICAL_PER_OLD_CLASS_SHA256,
            "e0_raw_scores": {
                outer_key: {
                    "path": (
                        f"{RAW_SCORE_ROOT}/{outer_key}/E0_FULL_ONLY/"
                        "scorer/diag_cosine_score.json"
                    ),
                    "sha256": digest,
                }
                for outer_key, digest in RAW_SCORE_SHA.items()
            },
            "e0_resource_rows": {
                outer_key: {
                    "fit_audit": dict(resource["fit_audit"]),
                    "scenes": {
                        scene: dict(values)
                        for scene, values in resource["scenes"].items()
                    },
                }
                for outer_key, resource in E0_RESOURCE_ROWS.items()
            },
            "rerun": False,
        },
        "g0_prerequisite": {
            "outer_key": G0_OUTER_KEY,
            "status": "D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS",
            "candidate_wall_p90_ns": 70_463_259,
            "candidate_reference_ratio_p90": 1.142053,
            "candidate_peak_absolute_bytes": 729_088,
            "candidate_peak_hard_pass": True,
            "candidate_peak_target_pass": False,
            "performance_claim": False,
        },
        "stop_rule": dict(STOP_RULE),
        "fresh_run_retry": False,
        "only_promotion_candidate": ARM_ID,
        "runtime_source": {
            "scientific_entry_commit": SCIENTIFIC_ENTRY_COMMIT,
            "files": {
                path: dict(record) for path, record in RUNTIME_SOURCE_FILES.items()
            },
        },
        "runtime": {
            "output_root": (
                "/home/szu2070436088/2510044040/CV-SincNet/runs/"
                "d92_ccoc_hard9_k1_20260817_v3"
            )
        },
        "outputs": {
            "matrix_manifest": "matrix_manifest.json",
            "smoke": "smoke/smoke_receipt.json",
            "jobs": "jobs/<outer_key>/"
            "E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS",
            "shard_summary": "summaries/shard_<index>.json",
            "coordinator_stop": "coordination/stop_action.json",
        },
    }


def validate_method_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the CCOC method lock is exactly frozen."""

    if not isinstance(lock, Mapping) or _canonical_bytes(lock) != _canonical_bytes(
        _expected_lock()
    ):
        raise D92CCOCHard9K1Error("CCOC Hard9+K1 method lock identity drift")
    return dict(lock)


def _read_method_lock(config_path: str | Path) -> tuple[Path, dict[str, Any], str]:
    source = Path(config_path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise D92CCOCHard9K1Error("method lock must be a regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise D92CCOCHard9K1Error("method lock JSON drift") from error
    if not isinstance(payload, dict):
        raise D92CCOCHard9K1Error("method lock must be an object")
    validate_method_lock(payload)
    return source, payload, _sha256_file(source)


def _package_entries(
    source_job_root: PurePath,
    *,
    require_files: bool,
) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for name, (package_parts, seal_parts) in PACKAGE_LAYOUT.items():
        package_root = source_job_root.joinpath(*package_parts)
        seal_path = source_job_root.joinpath(*seal_parts)
        concrete_package = Path(str(package_root))
        concrete_seal = Path(str(seal_path))
        if require_files and (
            not concrete_package.is_dir()
            or concrete_package.is_symlink()
            or not concrete_seal.is_file()
            or concrete_seal.is_symlink()
        ):
            raise D92CCOCHard9K1Error(f"sealed source package missing: {name}")
        entries[name] = {
            "package_root": str(package_root),
            "detached_seal_path": str(seal_path),
            "expected_seal_sha256": (
                _sha256_file(concrete_seal) if require_files else "0" * 64
            ),
        }
    return entries


def _preregistered_truth_sidecar_sha256(outer_key: str) -> str:
    """Return opaque truth identity without touching the truth sidecar."""

    digest = PREREGISTERED_TRUTH_SIDECAR_SHA256.get(str(outer_key))
    if not _is_sha256(digest):
        raise D92CCOCHard9K1Error("pre-registered truth sidecar hash missing")
    return str(digest).lower()


_MANIFEST_KEYS = {
    "schema",
    "status",
    "claim_scope",
    "protocol_schema",
    "selection_sha256",
    "method_lock",
    "method_lock_sha256",
    "context_sha256",
    "source_d92_output_root",
    "ground_component_dir",
    "ground_manifest_path",
    "ground_manifest_sha256",
    "output_root",
    "shard_count",
    "outer_count",
    "performance_outer_count",
    "liveness_outer_count",
    "job_count",
    "scene_count",
    "scene_arm_count",
    "arms",
    "candidate_ids",
    "primary_arm",
    "smoke_outer_key",
    "liveness_outer_key",
    "arm_roles",
    "coverage",
    "selected_rows",
    "jobs",
}
_JOB_KEYS = {
    "index",
    "outer_index",
    "arm_position",
    "planned_shard_index",
    "job_id",
    "outer_key",
    "outer_role",
    "hard_score",
    "receiver",
    "seed",
    "k_shot",
    "new_class_count",
    "arm_id",
    "candidate",
    "role",
    "primary",
    "scenarios",
    "source_job_root",
    "packages",
    "truth_sidecar",
    "truth_sidecar_sha256",
    "e0_resource",
    "method_lock_sha256",
    "output_root",
}


def validate_hard9_k1_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_method_lock_sha256: str | None = None,
    require_package_hashes: bool = False,
) -> dict[str, Any]:
    """Validate the final CCOC matrix schema without opening query truth."""

    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise D92CCOCHard9K1Error("manifest allowed-key drift")
    expected = {
        "schema": MATRIX_SCHEMA,
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "claim_scope": CLAIM_SCOPE,
        "protocol_schema": "p2_min_v1",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "context_sha256": CONTEXT_SHA256,
        "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT,
        "ground_component_dir": GROUND_COMPONENT_DIR,
        "ground_manifest_path": GROUND_MANIFEST_PATH,
        "ground_manifest_sha256": GROUND_MANIFEST_SHA256,
        "shard_count": SHARD_COUNT,
        "outer_count": 10,
        "performance_outer_count": 9,
        "liveness_outer_count": 1,
        "job_count": 10,
        "scene_count": 3,
        "scene_arm_count": 30,
        "arms": [ARM_ID],
        "candidate_ids": ARM_CANDIDATE_IDS,
        "primary_arm": ARM_ID,
        "smoke_outer_key": SMOKE_OUTER_KEY,
        "liveness_outer_key": LIVENESS_OUTER_KEY,
        "arm_roles": ARM_ROLES,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise D92CCOCHard9K1Error("manifest identity/count drift")
    method_lock_sha = manifest.get("method_lock_sha256")
    if not _is_sha256(method_lock_sha) or (
        expected_method_lock_sha256 is not None
        and str(method_lock_sha).lower()
        != str(expected_method_lock_sha256).lower()
    ):
        raise D92CCOCHard9K1Error("method-lock SHA drift")
    for field in (
        "method_lock",
        "source_d92_output_root",
        "ground_component_dir",
        "ground_manifest_path",
        "output_root",
    ):
        _pure_path(manifest.get(field))
    rows = _expected_rows()
    if manifest.get("selected_rows") != rows or manifest.get("coverage") != _coverage(
        rows
    ):
        raise D92CCOCHard9K1Error("selected-row/coverage drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(rows):
        raise D92CCOCHard9K1Error("job-count drift")
    seen_job_ids: set[str] = set()
    for index, row in enumerate(rows):
        job = jobs[index]
        expected_job = {
            "index": index,
            "outer_index": index,
            "arm_position": 0,
            "planned_shard_index": index % SHARD_COUNT,
            "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}",
            **row,
            "arm_id": ARM_ID,
            "candidate": CANDIDATE_ID,
            "role": "primary",
            "primary": True,
            "scenarios": list(SCENES),
            "e0_resource": {
                "fit_audit": dict(E0_RESOURCE_ROWS[row["outer_key"]]["fit_audit"]),
                "scenes": {
                    scene: dict(values)
                    for scene, values in E0_RESOURCE_ROWS[row["outer_key"]][
                        "scenes"
                    ].items()
                },
            },
            "method_lock_sha256": method_lock_sha,
        }
        if (
            not isinstance(job, Mapping)
            or set(job) != _JOB_KEYS
            or any(job.get(key) != value for key, value in expected_job.items())
        ):
            raise D92CCOCHard9K1Error("canonical job identity drift")
        if row["outer_key"] in EXCLUDED_OUTER_KEYS:
            raise D92CCOCHard9K1Error("G0 outer exclusion drift")
        if not _path_matches(
            job.get("source_job_root"),
            SOURCE_D92_OUTPUT_ROOT,
            "jobs",
            row["outer_key"],
        ) or not _path_matches(
            job.get("output_root"),
            manifest["output_root"],
            "jobs",
            row["outer_key"],
            ARM_ID,
        ):
            raise D92CCOCHard9K1Error("job path drift")
        source_job_root = _pure_path(job["source_job_root"])
        truth_sidecar_sha256 = job.get("truth_sidecar_sha256")
        if (
            _pure_path(job.get("truth_sidecar"))
            != source_job_root.joinpath("offline", "scorer", "truth_sidecar.json")
            or not _is_sha256(truth_sidecar_sha256)
            or (
                require_package_hashes
                and str(truth_sidecar_sha256).lower() == "0" * 64
            )
        ):
            raise D92CCOCHard9K1Error("truth sidecar hash drift")
        packages = job.get("packages")
        if not isinstance(packages, Mapping) or set(packages) != set(PACKAGE_LAYOUT):
            raise D92CCOCHard9K1Error("package identity drift")
        for name, package in packages.items():
            package_parts, seal_parts = PACKAGE_LAYOUT[name]
            if (
                not isinstance(package, Mapping)
                or set(package)
                != {
                    "package_root",
                    "detached_seal_path",
                    "expected_seal_sha256",
                }
                or _pure_path(package.get("package_root"))
                != source_job_root.joinpath(*package_parts)
                or _pure_path(package.get("detached_seal_path"))
                != source_job_root.joinpath(*seal_parts)
                or not _is_sha256(package.get("expected_seal_sha256"))
                or (
                    require_package_hashes
                    and str(package.get("expected_seal_sha256")).lower()
                    == "0" * 64
                )
            ):
                raise D92CCOCHard9K1Error("package path/hash drift")
        job_id = str(job["job_id"])
        if job_id in seen_job_ids:
            raise D92CCOCHard9K1Error("duplicate job identity")
        seen_job_ids.add(job_id)
    return dict(manifest)


def build_hard9_k1_manifest(
    config_path: str | Path,
    *,
    require_package_files: bool,
) -> dict[str, Any]:
    """Build the exact ten-job manifest from the frozen CCOC method lock."""

    lock_path, lock, method_lock_sha = _read_method_lock(config_path)
    runtime = lock["runtime"]
    if not isinstance(runtime, Mapping):
        raise D92CCOCHard9K1Error("method lock runtime identity drift")
    output_root = runtime.get("output_root")
    _pure_path(output_root)
    source_root = PurePosixPath(SOURCE_D92_OUTPUT_ROOT)
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(_expected_rows()):
        source_job_root = source_root.joinpath("jobs", row["outer_key"])
        jobs.append(
            {
                "index": index,
                "outer_index": index,
                "arm_position": 0,
                "planned_shard_index": index % SHARD_COUNT,
                "job_id": f"{row['outer_key']}__arm_{ARM_ID.lower()}",
                **row,
                "arm_id": ARM_ID,
                "candidate": CANDIDATE_ID,
                "role": "primary",
                "primary": True,
                "scenarios": list(SCENES),
                "source_job_root": str(source_job_root),
                "packages": _package_entries(
                    source_job_root,
                    require_files=require_package_files,
                ),
                "truth_sidecar": str(
                    source_job_root.joinpath(
                        "offline", "scorer", "truth_sidecar.json"
                    )
                ),
                "truth_sidecar_sha256": _preregistered_truth_sidecar_sha256(
                    row["outer_key"]
                ),
                "e0_resource": {
                    "fit_audit": dict(
                        E0_RESOURCE_ROWS[row["outer_key"]]["fit_audit"]
                    ),
                    "scenes": {
                        scene: dict(values)
                        for scene, values in E0_RESOURCE_ROWS[row["outer_key"]][
                            "scenes"
                        ].items()
                    },
                },
                "method_lock_sha256": method_lock_sha,
                "output_root": str(
                    _pure_path(output_root).joinpath(
                        "jobs", row["outer_key"], ARM_ID
                    )
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema": MATRIX_SCHEMA,
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "claim_scope": CLAIM_SCOPE,
        "protocol_schema": "p2_min_v1",
        "selection_sha256": CANONICAL_SELECTION_SHA256,
        "method_lock": str(lock_path),
        "method_lock_sha256": method_lock_sha,
        "context_sha256": CONTEXT_SHA256,
        "source_d92_output_root": SOURCE_D92_OUTPUT_ROOT,
        "ground_component_dir": GROUND_COMPONENT_DIR,
        "ground_manifest_path": GROUND_MANIFEST_PATH,
        "ground_manifest_sha256": GROUND_MANIFEST_SHA256,
        "output_root": str(_pure_path(output_root)),
        "shard_count": SHARD_COUNT,
        "outer_count": 10,
        "performance_outer_count": 9,
        "liveness_outer_count": 1,
        "job_count": 10,
        "scene_count": len(SCENES),
        "scene_arm_count": 30,
        "arms": [ARM_ID],
        "candidate_ids": dict(ARM_CANDIDATE_IDS),
        "primary_arm": ARM_ID,
        "smoke_outer_key": SMOKE_OUTER_KEY,
        "liveness_outer_key": LIVENESS_OUTER_KEY,
        "arm_roles": dict(ARM_ROLES),
        "coverage": _coverage(_expected_rows()),
        "selected_rows": _expected_rows(),
        "jobs": jobs,
    }
    validate_hard9_k1_manifest(
        manifest,
        expected_method_lock_sha256=method_lock_sha,
        require_package_hashes=require_package_files,
    )
    return manifest


build_ccoc_hard9_k1_manifest = build_hard9_k1_manifest
validate_manifest = validate_hard9_k1_manifest


__all__ = [
    "ARM_ID",
    "ARM_ORDER",
    "ARM_CANDIDATE_IDS",
    "ARM_ROLES",
    "CANDIDATE_ID",
    "CANONICAL_SELECTION_SHA256",
    "CLAIM_SCOPE",
    "D92CCOCHard9K1Error",
    "D92CCOCHard9K1MatrixError",
    "EXCLUDED_OUTER_KEYS",
    "FIT_GATE",
    "G0_OUTER_KEY",
    "HARD9_K1_ROWS",
    "HARD9_K1_V1_ROWS",
    "JOB_RECEIPT_SCHEMA",
    "LIVENESS_OUTER_KEY",
    "MATRIX_SCHEMA",
    "METHOD_LOCK_SCHEMA",
    "PRIMARY_ARM",
    "QUERY_CONTRACT",
    "QUERY_ZERO_FIELDS",
    "REGISTERED_MODE",
    "RESOURCE_GATE",
    "RUNTIME_SOURCE_FILES",
    "SCENES",
    "SCIENTIFIC_ENTRY_COMMIT",
    "SHARD_COUNT",
    "SHARD_SUMMARY_SCHEMA",
    "SMOKE_OUTER_KEY",
    "STATE_POSTPROCESS_MODE",
    "STOP_RULE",
    "SYSTEMIC_FAILURE_SCHEMA",
    "build_ccoc_hard9_k1_manifest",
    "build_hard9_k1_manifest",
    "canonical_selection_sha256",
    "validate_hard9_k1_manifest",
    "validate_manifest",
    "validate_method_lock",
]
