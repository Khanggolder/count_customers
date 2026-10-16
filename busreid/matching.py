from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import EmbeddingRecord, MatchDecision, MatchStatus


class MatchingEngine:
    def __init__(self, threshold: float = 0.72, ambiguity_margin: float = 0.05) -> None:
        if not -1 <= threshold <= 1:
            raise ValueError("Cosine threshold phải trong [-1, 1]")
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-12 else vector

    def match_batch(
        self,
        onboard: list[EmbeddingRecord],
        exits: list[EmbeddingRecord],
        paired_and_verified: bool = False,
    ) -> list[MatchDecision]:
        if not exits:
            return []
        eligible = [item for item in onboard if any(out.timestamp_s > item.timestamp_s for out in exits)]
        if not eligible:
            return [
                MatchDecision(item.event_id, None, MatchStatus.UNKNOWN, None, "Không có IN trước OUT")
                for item in exits
            ]
        in_matrix = np.stack([self._normalized(item.embedding) for item in eligible])
        out_matrix = np.stack([self._normalized(item.embedding) for item in exits])
        similarities = out_matrix @ in_matrix.T
        valid_time = np.asarray(
            [[out.timestamp_s > entered.timestamp_s for entered in eligible] for out in exits], dtype=bool
        )
        scores = np.where(valid_time, similarities, -1.0)
        rows, columns = linear_sum_assignment(-scores)
        assigned = {row: column for row, column in zip(rows, columns, strict=True)}
        decisions: list[MatchDecision] = []
        for out_index, out in enumerate(exits):
            if out_index not in assigned:
                decisions.append(MatchDecision(out.event_id, None, MatchStatus.UNKNOWN, None, "Không được gán one-to-one"))
                continue
            in_index = assigned[out_index]
            score = float(scores[out_index, in_index])
            candidate_scores = np.sort(scores[out_index][scores[out_index] > -0.999])[::-1]
            ambiguous = len(candidate_scores) > 1 and candidate_scores[0] - candidate_scores[1] < self.ambiguity_margin
            if score < self.threshold:
                decisions.append(MatchDecision(out.event_id, None, MatchStatus.UNKNOWN, score, "Similarity dưới ngưỡng"))
            elif ambiguous:
                decisions.append(MatchDecision(out.event_id, None, MatchStatus.UNKNOWN, score, "Hai ứng viên quá tương đồng"))
            else:
                entered = eligible[in_index]
                status = MatchStatus.CONFIRMED if paired_and_verified else MatchStatus.UNVERIFIED
                reason = "Có ground truth/nguồn paired" if paired_and_verified else "Candidate match — chưa kiểm chứng identity"
                decisions.append(
                    MatchDecision(out.event_id, entered.event_id, status, score, reason, entered.passenger_id)
                )
        return decisions

