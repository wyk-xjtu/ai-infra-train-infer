"""
投机解码（Speculative Decoding）模块

使用小模型（draft model）快速生成多个候选 token，大模型（target model）一次 forward 验证。
参考：Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (2023)

工作原理：
- Draft model（如 Qwen3-0.6B）自回归生成 K 个候选 token（速度快，质量低）
- Target model（如 Qwen3-8B）对 prompt + K 个候选进行一次 forward（得到 K+1 个 logits）
- 从左到右逐位比较：target argmax == draft token 则接受，否则拒绝后续所有
- 最终返回所有被接受的 token + target 在第一个拒绝位置采样的 token

收益：
- 接受率高时（如 >0.7），每次 target forward 产出多个 token → 总步数减少
- 接受率低时自动 fallback 到普通 decode（通过 acceptance_rate 阈值控制）
"""

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """投机解码配置"""
    enabled: bool = False
    draft_model_path: str = "./models/Qwen3-0.6B"
    num_speculative_tokens: int = 5  # 每轮投机生成的候选 token 数（K）
    temperature: float = 0.0         # draft model 采样温度（0=greedy）
    # Phase 2 新增配置
    use_batch_draft: bool = True     # 是否使用批量 draft 生成（连续 forward 多步）
    use_sampling_verification: bool = False  # 是否使用 sampling-aware 验证
    verification_temperature: float = 1.0    # 验证时的 temperature
    verification_top_p: float = 0.9          # 验证时的 top-p
    min_acceptance_prob: float = 0.1         # 概率接受阈值


class SpeculativeDecoder:
    """投机解码器

    工作流程：
    1. Draft model 快速自回归生成 K 个候选 token
    2. Target model 一次 forward 验证这 K 个 token + 生成第 K+1 个
    3. 从左到右比较，接受匹配的 token，遇到不匹配则拒绝后续所有
    4. 返回所有被接受的 token
    """

    def __init__(self, config: SpeculativeConfig, device: torch.device):
        """
        Args:
            config: 投机解码配置
            device: 目标设备（draft model 加载到此设备）
        """
        self.config = config
        self.device = device
        self._draft_model = None
        self._draft_tokenizer = None
        self._loaded = False

        # 接受率统计（滑动窗口，最近 100 次验证）
        self._acceptance_history: deque = deque(maxlen=100)
        self._total_proposed: int = 0
        self._total_accepted: int = 0

    def load_draft_model(self):
        """加载 draft model（HF AutoModelForCausalLM）

        使用 HF transformers 直接加载，不走自研 PagedAttention 路径。
        加载失败时优雅 fallback：disable speculative + warning。
        """
        if self._loaded:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(
                "Loading draft model from %s for speculative decoding...",
                self.config.draft_model_path,
            )

            # 尝试加载模型，按优先级选择 attention 实现
            model = None
            attn_implementations = ["flash_attention_2", "sdpa", "eager"]
            for impl in attn_implementations:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.config.draft_model_path,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        attn_implementation=impl,
                    )
                    logger.info(
                        "Draft model loaded with attn_implementation='%s'", impl
                    )
                    break
                except (ImportError, RuntimeError, ValueError) as e:
                    logger.debug(
                        "Draft model attn_implementation='%s' failed: %s", impl, e
                    )
                    continue

            if model is None:
                raise RuntimeError(
                    "All attention implementations failed for draft model"
                )

            model = model.to(self.device)
            model.eval()
            self._draft_model = model

            self._draft_tokenizer = AutoTokenizer.from_pretrained(
                self.config.draft_model_path, trust_remote_code=True
            )

            self._loaded = True
            logger.info(
                "Draft model loaded successfully: %s", self.config.draft_model_path
            )

        except Exception as e:
            warnings.warn(
                f"Failed to load draft model: {e}. "
                f"Disabling speculative decoding.",
                RuntimeWarning,
            )
            logger.error("Draft model loading failed: %s", e)
            self.config.enabled = False
            self._loaded = False

    @torch.inference_mode()
    def speculate(
        self,
        input_ids: torch.Tensor,  # [1, seq_len] 当前序列
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Draft model 快速生成 K 个候选 token

        Phase 2 改进：支持批量连续 forward 多步生成，减少 kernel launch 开销。
        同时返回 draft logits 供 sampling-aware 验证使用。

        Args:
            input_ids: [1, seq_len] 当前序列（prompt + 已生成 tokens）

        Returns:
            draft_tokens: [K] 候选 token ids
            draft_logits: [K, vocab_size] 各位置的 draft logits（用于 sampling 验证）
        """
        assert self._draft_model is not None, "Draft model not loaded"

        K = self.config.num_speculative_tokens
        temperature = self.config.temperature

        if self.config.use_batch_draft:
            return self._speculate_batch(input_ids, K, temperature)
        else:
            return self._speculate_sequential(input_ids, K, temperature)

    def _speculate_batch(self, input_ids: torch.Tensor, K: int, temperature: float
                         ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """批量 draft 生成：连续 forward 多步

        优化：使用 KV Cache（HF past_key_values）避免重复计算已有前缀。
        每步只需 forward 最新 token（而非整个序列），显著降低计算量。

        Args:
            input_ids: [1, seq_len] 当前序列
            K: 投机生成的 token 数
            temperature: 采样温度

        Returns:
            (draft_tokens, draft_logits)
        """
        draft_tokens = []
        draft_logits_list = []
        past_key_values = None

        # 第一步：处理完整 input_ids 以获得初始 KV Cache
        outputs = self._draft_model(
            input_ids,
            use_cache=True,
        )
        next_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        past_key_values = outputs.past_key_values

        # 采样第一个 draft token
        next_token, logits_for_record = self._sample_token(next_logits, temperature)
        draft_tokens.append(next_token.item())
        draft_logits_list.append(logits_for_record.squeeze(0))  # [vocab_size]

        # 后续步骤：只 forward 最新 token + past_key_values
        for _ in range(K - 1):
            next_input = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
            outputs = self._draft_model(
                next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            past_key_values = outputs.past_key_values

            next_token, logits_for_record = self._sample_token(next_logits, temperature)
            draft_tokens.append(next_token.item())
            draft_logits_list.append(logits_for_record.squeeze(0))

        tokens_tensor = torch.tensor(draft_tokens, dtype=torch.long, device=self.device)
        logits_tensor = torch.stack(draft_logits_list, dim=0)  # [K, vocab_size]
        return tokens_tensor, logits_tensor

    def _speculate_sequential(self, input_ids: torch.Tensor, K: int, temperature: float
                              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """逐 token 顺序生成（原始实现，作为 fallback）

        Args:
            input_ids: [1, seq_len] 当前序列
            K: 投机生成的 token 数
            temperature: 采样温度

        Returns:
            (draft_tokens, draft_logits)
        """
        draft_tokens = []
        draft_logits_list = []
        current_ids = input_ids  # [1, seq_len]

        for _ in range(K):
            outputs = self._draft_model(current_ids)
            next_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            next_token, logits_for_record = self._sample_token(next_logits, temperature)
            draft_tokens.append(next_token.item())
            draft_logits_list.append(logits_for_record.squeeze(0))

            # 追加到 input 用于下一步
            next_token_2d = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
            current_ids = torch.cat([current_ids, next_token_2d], dim=1)

        tokens_tensor = torch.tensor(draft_tokens, dtype=torch.long, device=self.device)
        logits_tensor = torch.stack(draft_logits_list, dim=0)  # [K, vocab_size]
        return tokens_tensor, logits_tensor

    def _sample_token(self, logits: torch.Tensor, temperature: float
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 logits 中采样 token

        Args:
            logits: [1, vocab_size]
            temperature: 采样温度（0=greedy）

        Returns:
            (token, logits): token [1], logits [1, vocab_size] 用于后续验证
        """
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1)  # [1]
        else:
            scaled = logits.float() / temperature
            probs = torch.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token, logits

    def verify(
        self,
        input_ids: torch.Tensor,       # [1, seq_len] 原始序列
        draft_tokens: torch.Tensor,    # [K] 候选 token
        target_logits: torch.Tensor,   # [1, K+1, vocab] target model 的 logits
        draft_logits: Optional[torch.Tensor] = None,  # [K, vocab] draft model logits
    ) -> Tuple[torch.Tensor, int]:
        """验证候选 token，返回被接受的 token 和接受数量

        Phase 2 改进：
        - 当 use_sampling_verification=True 且提供 draft_logits 时，
          使用概率接受/拒绝策略（非简单 argmax 比较）
        - 否则使用原始 greedy 验证策略

        Args:
            input_ids: [1, seq_len] 原始序列（验证前的序列）
            draft_tokens: [K] 由 draft model 生成的候选 token
            target_logits: [1, K+1, vocab_size] target model 对
                           (原序列末位, draft_token_1, ..., draft_token_K) 的 logits
            draft_logits: [K, vocab_size] draft model 各位置的 logits（可选）

        Returns:
            accepted_tokens: [num_accepted] 被接受的 token
                            （最后一个是由 target model 采样的 token）
            num_accepted: draft token 中被接受的数量（0 到 K）
        """
        if self.config.use_sampling_verification and draft_logits is not None:
            return self._verify_with_sampling(input_ids, draft_tokens, target_logits, draft_logits)
        else:
            return self._verify_greedy(input_ids, draft_tokens, target_logits)

    def _verify_greedy(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Greedy 验证策略（原始实现）

        逐个比较 target model 在对应位置的 argmax 与 draft token。
        一旦不匹配，拒绝后续所有候选。
        """
        K = draft_tokens.shape[0]

        # target_logits: [1, K+1, vocab] → [K+1, vocab]
        if target_logits.dim() == 3:
            logits = target_logits.squeeze(0)  # [K+1, vocab]
        else:
            logits = target_logits  # [K+1, vocab]

        # 逐位验证
        num_accepted = 0
        accepted_list = []

        for i in range(K):
            target_token = torch.argmax(logits[i]).item()
            draft_token = draft_tokens[i].item()

            if target_token == draft_token:
                accepted_list.append(draft_token)
                num_accepted += 1
            else:
                accepted_list.append(target_token)
                break
        else:
            # 所有 K 个 draft token 都被接受，追加 bonus token
            bonus_token = torch.argmax(logits[K]).item()
            accepted_list.append(bonus_token)

        # 更新接受率统计
        self._acceptance_history.append(num_accepted / K if K > 0 else 0.0)
        self._total_proposed += K
        self._total_accepted += num_accepted

        accepted_tokens = torch.tensor(
            accepted_list, dtype=torch.long, device=self.device
        )
        return accepted_tokens, num_accepted

    def _verify_with_sampling(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Sampling-aware 验证：使用概率接受/拒绝策略

        而非简单的 argmax 比较，使用 target 和 draft 的概率比值进行接受/拒绝：
          p_accept = min(1, p_target(x) / p_draft(x)) for each token x

        这保证了最终输出的分布严格等于 target model 的分布（无偏）。
        相比 greedy 验证，sampling 验证在 temperature > 0 时接受率更高。

        参考：Leviathan et al. 2023, Algorithm 1

        Args:
            input_ids: [1, seq_len]
            draft_tokens: [K]
            target_logits: [1, K+1, vocab_size]
            draft_logits: [K, vocab_size]

        Returns:
            (accepted_tokens, num_accepted)
        """
        K = draft_tokens.shape[0]
        temperature = self.config.verification_temperature
        top_p = self.config.verification_top_p

        # target_logits: [1, K+1, vocab] → [K+1, vocab]
        if target_logits.dim() == 3:
            t_logits = target_logits.squeeze(0)  # [K+1, vocab]
        else:
            t_logits = target_logits

        num_accepted = 0
        accepted_list = []

        for i in range(K):
            # 计算 target 和 draft 在该位置的概率分布
            target_probs = self._compute_probs(t_logits[i], temperature, top_p)
            draft_probs = self._compute_probs(draft_logits[i], temperature, top_p)

            draft_token = draft_tokens[i].item()
            p_target = target_probs[draft_token].item()
            p_draft = max(draft_probs[draft_token].item(), 1e-10)  # 防除零

            # 概率接受/拒绝
            acceptance_ratio = min(1.0, p_target / p_draft)
            rand_val = torch.rand(1, device=self.device).item()

            if rand_val < acceptance_ratio:
                # 接受该 token
                accepted_list.append(draft_token)
                num_accepted += 1
            else:
                # 拒绝：从修正分布中采样
                # 修正分布: max(0, p_target - p_draft)，归一化后采样
                residual = torch.clamp(target_probs - draft_probs, min=0)
                residual_sum = residual.sum()
                if residual_sum > 1e-10:
                    residual = residual / residual_sum
                    new_token = torch.multinomial(residual, num_samples=1).item()
                else:
                    # residual 全为 0，直接从 target 分布采样
                    new_token = torch.multinomial(target_probs, num_samples=1).item()
                accepted_list.append(new_token)
                break
        else:
            # 所有 K 个 token 都被接受，从 target 在 K+1 位置采样 bonus token
            bonus_probs = self._compute_probs(t_logits[K], temperature, top_p)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
            accepted_list.append(bonus_token)

        # 更新统计
        self._acceptance_history.append(num_accepted / K if K > 0 else 0.0)
        self._total_proposed += K
        self._total_accepted += num_accepted

        accepted_tokens = torch.tensor(
            accepted_list, dtype=torch.long, device=self.device
        )
        return accepted_tokens, num_accepted

    def _compute_probs(
        self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0
    ) -> torch.Tensor:
        """从 logits 计算概率分布（支持 temperature + top-p）

        Args:
            logits: [vocab_size]
            temperature: 温度参数
            top_p: nucleus sampling 阈值

        Returns:
            probs: [vocab_size] 概率分布
        """
        if temperature <= 0:
            # Greedy: one-hot 分布
            probs = torch.zeros_like(logits)
            probs[torch.argmax(logits)] = 1.0
            return probs

        scaled = logits.float() / temperature

        if top_p < 1.0:
            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过 top_p 的 token（保留第一个超过的）
            sorted_indices_to_remove = cumulative_probs > top_p
            # 右移一位，确保第一个超过阈值的 token 被保留
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled[indices_to_remove] = float('-inf')

        probs = torch.softmax(scaled, dim=-1)
        return probs

    def sync_kv_cache(
        self,
        accepted_tokens: torch.Tensor,
        target_kv_cache: Optional[object] = None,
    ):
        """将接受的 token 写入 target model 的 KV Cache

        在投机解码验证后，被接受的 token 需要同步到 target model 的 KV Cache 中，
        以确保下一轮推理时 KV Cache 状态正确。

        Phase 2 改进：显式的 KV Cache 同步接口，确保 target model 状态一致。

        Args:
            accepted_tokens: [num_accepted] 被接受的 token ids
            target_kv_cache: target model 的 KV Cache 对象（PagedAttention 的 cache）
                            如果为 None，则由调用方自行管理 KV Cache 更新
        """
        # KV Cache 同步由 InferenceEngine 的 step 循环负责：
        # 接受 N 个 token 后，engine 需要：
        # 1. 将 accepted_tokens 追加到序列
        # 2. 为 target model 的 KV Cache 分配 N 个新 slot
        # 3. 对 accepted_tokens 做 target model forward 以填充 KV Cache
        #    （或者利用验证 forward 的中间结果直接写入）
        #
        # 此方法提供一个明确的同步点，让调用方知道需要更新 KV Cache
        self._last_sync_tokens = accepted_tokens
        self._last_sync_count = len(accepted_tokens)

        if target_kv_cache is not None:
            # 标记需要 KV Cache 更新（实际写入由 engine 的 model forward 完成）
            logger.debug(
                "KV Cache sync requested: %d tokens accepted, "
                "target KV Cache will be updated on next forward",
                len(accepted_tokens),
            )

        return len(accepted_tokens)

    @property
    def acceptance_rate(self) -> float:
        """历史平均接受率（滑动窗口，用于自动禁用判断）

        Returns:
            最近 100 次验证的平均接受率，无历史数据时返回 1.0（乐观默认）
        """
        if not self._acceptance_history:
            return 1.0  # 无历史数据时假设接受率高，允许投机解码
        return sum(self._acceptance_history) / len(self._acceptance_history)

    @property
    def is_loaded(self) -> bool:
        """Draft model 是否已成功加载"""
        return self._loaded

    def get_stats(self) -> dict:
        """获取投机解码统计信息"""
        return {
            "enabled": self.config.enabled,
            "loaded": self._loaded,
            "acceptance_rate": self.acceptance_rate,
            "total_proposed": self._total_proposed,
            "total_accepted": self._total_accepted,
            "num_speculative_tokens": self.config.num_speculative_tokens,
            "history_window_size": len(self._acceptance_history),
        }
