from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import json
import math
import os
from pathlib import Path

from model_interfaces import ModelInterface
from data.schemas import LogLineSchema
from enums import SplitRunType
from utils import data_utils, config_utils, metric_utils
import constants




class RunnerInterface(ABC):
    def __init__(
        self,
        model_interfaces: Dict[str, ModelInterface],
        dataloader: Any,
        split_run_type: SplitRunType,
        rank: int = 0
    ) -> None:
        self._config = config_utils.load_config()

        self.rank = rank
        self.model_name = self._config[constants.FIELD_MODEL_TO_USE]
        self.dataset_name = self._config.get('dataset_to_use', "")
        self.timestamp = data_utils.get_timestamp()
        self.model_interfaces = model_interfaces
        self.dataloader = dataloader
        self.split_run_type = split_run_type
        self.log_folder_dir = self.create_log_foler()
        
        # Print config to log file when run starts
        self._log_config_at_start()

    def get_session_identifier(self):
        return f"{self.model_name}_{self.timestamp}_{self.dataset_name}"

    def get_runner_identifier(self):
        return f"{self.model_name}_{self.timestamp}_{self.dataset_name}_{self.split_run_type.value}"

    def create_log_foler(self) -> str:
        if self.rank != self._config.get('read_write_gpu_rank'):
            return None
        session_folder_dir =  f"./logs/{self.get_session_identifier()}"
        os.makedirs(session_folder_dir, exist_ok=True)

        log_folder_dir = f"{session_folder_dir}/{self.get_runner_identifier()}"
        os.makedirs(log_folder_dir, exist_ok=False)
        return log_folder_dir

    def _log_config_at_start(self) -> None:
        """Log the entire config dictionary to the combined log file when run starts"""
        if self.log_folder_dir is None:
            return
            
        config_log_lines = [
            LogLineSchema(
                data=[("RUN_START", "Configuration")],
                new_line_in_between=False,
                blank_line_after=True,
            )
        ]
        
        # Add each config item
        for key, value in self._config.items():
            config_log_lines.append(
                LogLineSchema(
                    data=[(key, str(value))],
                    new_line_in_between=False,
                    blank_line_after=False,
                )
            )
        
        config_log_lines.append(
            LogLineSchema(
                data=[("RUN_START_END", "Configuration")],
                new_line_in_between=False,
                blank_line_after=True,
            )
        )
        
        data_utils.write_to_epoch_log(
            self.get_combined_log_file_path(), config_log_lines, mode="w+"
        )

    def get_combined_log_file_path(self) -> str:
        return f"{self.log_folder_dir}/{self.get_runner_identifier()}.txt"

    def get_epoch_log_file_path(self, epoch: int) -> str:
        return f"{self.log_folder_dir}/{epoch}_{self.get_runner_identifier()}.txt"

    def epoch_analysis(
        self,
        epoch: int,
        probabilities: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        other_values: Dict[str, Any] = {},
    ) -> dict[str, float]:
        epoch_log_lines: List[LogLineSchema] = []

        epoch_log_lines.append(
            LogLineSchema(
                data=[("Run Type", self.split_run_type)],
                new_line_in_between=False,
                blank_line_after=True,
            )
        )

        assert predictions.keys() == targets.keys()
        metrics_per_output: Dict[str, Dict[str, Any]] = {}
        for k in predictions.keys():
            epoch_log_lines.append(
                LogLineSchema(
                    data=[("Metrics For", k)],
                    new_line_in_between=False,
                    blank_line_after=False,
                )
            )

            metrics_dict = metric_utils.get_metrics(
                probabilities.get(k), predictions[k], targets[k]
            )
            metrics_per_output[k] = metrics_dict
            epoch_log_lines.append(
                LogLineSchema(
                    data=list(metrics_dict.items()),
                    new_line_in_between=True,
                    blank_line_after=True,
                )
            )

        epoch_log_lines.append(
            LogLineSchema(
                data=list(other_values.items()),
                new_line_in_between=True,
                blank_line_after=True,
            )
        )

        data_utils.write_to_epoch_log(
            self.get_combined_log_file_path(), epoch_log_lines, mode="a+"
        )
        data_utils.write_to_epoch_log(
            self.get_epoch_log_file_path(epoch), epoch_log_lines, mode="w+"
        )

        if self.split_run_type == SplitRunType.TEST:
            try:
                self._save_roc_auc_curves(probabilities=probabilities, targets=targets, epoch=epoch)
            except Exception as e:
                print(f"Error saving ROC AUC curves: {e}")

            self._log_statistical_tests(predictions, targets, metrics_per_output)

        return metrics_dict

    @abstractmethod
    def run_epoch(self, epoch: int) -> None:
        """run_epoch must be implemented by the subclass"""
        raise NotImplementedError("Subclasses must implement run_epoch")

    def _save_roc_auc_curves(self, probabilities: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], epoch: int) -> None:
        os.makedirs(f"test_run_logs/{self.get_runner_identifier()}", exist_ok=True)
        for k in targets.keys():
            probs_tensor = torch.softmax(probabilities[k], dim=1)
            y_score = probs_tensor.detach().cpu().numpy()
            y_true = targets[k].detach().cpu().numpy()

            num_classes = y_score.shape[1]
            plt.figure()
            # Prepare randomized, mostly non-repeating style combinations
            linestyles = ['-', '--', '-.', ':']
            markers = ['o', '^', 'v', 'X', '*']
            style_combos = [(ls, mk) for ls in linestyles for mk in markers]
            random.shuffle(style_combos)
            # If classes exceed unique combos, we will wrap around
            for c in range(num_classes):
                y_true_bin = (y_true == c).astype(int)
                fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, c])
                curve_auc = auc(fpr, tpr)
                linestyle, marker = style_combos[c % len(style_combos)]
                markevery = max(1, len(fpr) // 25)
                plt.plot(
                    fpr,
                    tpr,
                    label=f"Class {c} (AUC={curve_auc:.3f})",
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.75,
                    markersize=5,
                    markevery=markevery,
                )

            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC AUC - {k}')
            plt.legend(loc='lower right')
            plt.tight_layout()
            save_path = f"test_run_logs/{self.get_runner_identifier()}/roc_{self.get_runner_identifier()}_{k}_{epoch}.png"
            plt.savefig(save_path)
            plt.close() 

    def _log_statistical_tests(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        metrics_per_output: Dict[str, Dict[str, Any]],
    ) -> None:
        if self.log_folder_dir is None:
            return

        from scipy import stats

        results: Dict[str, Dict[str, Any]] = {}
        for key in predictions.keys():
            pred_tensor = predictions[key]
            target_tensor = targets[key]

            if isinstance(pred_tensor, torch.Tensor):
                pred_tensor = pred_tensor.detach().cpu().flatten()
            if isinstance(target_tensor, torch.Tensor):
                target_tensor = target_tensor.detach().cpu().flatten()

            if pred_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Predictions and targets for key '{key}' have mismatched shapes: "
                    f"{pred_tensor.shape} vs {target_tensor.shape}."
                )

            num_samples = int(pred_tensor.shape[0])
            entry: Dict[str, Any] = {"num_samples": num_samples}

            if num_samples < 2:
                entry["paired_t_test"] = {"statistic": None, "p_value": None}
                entry["wilcoxon"] = {"statistic": None, "p_value": None}
                entry["note"] = "Insufficient samples for statistical tests (needs at least 2)."
                results[key] = entry
                continue

            pred_np = pred_tensor.numpy().astype(float)
            target_np = target_tensor.numpy().astype(float)

            try:
                ttest_res = stats.ttest_rel(
                    pred_np, target_np, nan_policy="propagate"
                )
                t_statistic = getattr(ttest_res, "statistic", ttest_res[0])
                t_pvalue = getattr(ttest_res, "pvalue", ttest_res[1])
                if math.isnan(t_statistic):
                    t_statistic = None
                if math.isnan(t_pvalue):
                    t_pvalue = None
            except Exception as exc:  # pragma: no cover - defensive
                t_statistic = None
                t_pvalue = None
                entry["paired_t_test_error"] = str(exc)

            try:
                wilcoxon_res = stats.wilcoxon(
                    pred_np, target_np, zero_method="wilcox", correction=False
                )
                wilcoxon_stat = getattr(
                    wilcoxon_res, "statistic", wilcoxon_res[0]
                )
                wilcoxon_pvalue = getattr(
                    wilcoxon_res, "pvalue", wilcoxon_res[1]
                )
                if math.isnan(wilcoxon_stat):
                    wilcoxon_stat = None
                if math.isnan(wilcoxon_pvalue):
                    wilcoxon_pvalue = None
            except ValueError as exc:
                wilcoxon_stat = None
                wilcoxon_pvalue = None
                entry["wilcoxon_note"] = (
                    "Wilcoxon test not applicable: " + str(exc)
                )

            entry["paired_t_test"] = {
                "statistic": None if t_statistic is None else float(t_statistic),
                "p_value": None if t_pvalue is None else float(t_pvalue),
            }
            entry["wilcoxon"] = {
                "statistic": None
                if wilcoxon_stat is None
                else float(wilcoxon_stat),
                "p_value": None
                if wilcoxon_pvalue is None
                else float(wilcoxon_pvalue),
            }
            results[key] = entry

        serializable_metrics = self._serialize_metrics(metrics_per_output)

        payload = {
            "runner_identifier": self.get_runner_identifier(),
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "split_run_type": self.split_run_type.value,
            "statistical_tests": results,
            "metrics": serializable_metrics,
        }

        output_dir = Path("test_run_logs") / Path("statistical_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.get_runner_identifier()}_stats.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _serialize_metrics(
        self, metrics_per_output: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        serializable: Dict[str, Dict[str, Any]] = {}
        for key, metrics in metrics_per_output.items():
            serializable[key] = {}
            for metric_name, value in metrics.items():
                serializable[key][metric_name] = self._serialize_value(value)
        return serializable

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except TypeError:  # pragma: no cover - defensive
                pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                pass
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        return str(value)