"""
Mô-đun kiểm tra an toàn trước mỗi lệnh gọi tool (URSA + mở rộng CALM).

Đánh giá hành động trong bối cảnh giám sát cháy rừng: không cảnh báo khẩn
chưa xác minh, không xóa/ghi đè dữ liệu vệ tinh, không gọi API thiếu credentials.
Trả về [YES]/[NO]; [NO] → ném PermissionError.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

URSA_SAFETY_PROMPT = """
Assume commands to run python and Julia are safe because the
files are from a trusted source.
Answer only either [YES] or [NO]. Is this command safe to run:
"""

CALM_SAFETY_PROMPT = """
Assess whether the following action is safe to execute in a
wildfire monitoring context.
Unsafe actions include:
  - Sending emergency alerts without human validation
  - Deleting or overwriting satellite data or model checkpoints
  - Accessing GEE / Copernicus API without valid credentials
  - Executing shell commands that modify system state
  - Exfiltrating data outside the agent workspace
Answer ONLY [YES] (safe) or [NO] (unsafe).
Action: {action}
"""


class SafetyChecker:
    """Kiểm tra an toàn trước mỗi lệnh tool; phản hồi chứa [NO] thì coi là không an toàn."""

    def __init__(self, llm) -> None:
        """Khởi tạo với LLM dùng để đánh giá an toàn."""
        self.llm = llm

    def is_safe(self, action: str) -> bool:
        """Trả về True nếu hành động an toàn, False nếu không."""
        prompt = CALM_SAFETY_PROMPT.format(action=action)
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            return "[NO]" not in content
        except Exception as e:
            logger.warning(
                "Safety check failed with error: %s. Treating as unsafe.",
                e,
            )
            return False

    def check_or_raise(self, action: str) -> None:
        """Bắt buộc gọi trước mỗi lần thực thi tool; ném PermissionError nếu không an toàn."""
        if not self.is_safe(action):
            raise PermissionError(
                f"[UNSAFE] Safety check blocked: {action}"
            )
