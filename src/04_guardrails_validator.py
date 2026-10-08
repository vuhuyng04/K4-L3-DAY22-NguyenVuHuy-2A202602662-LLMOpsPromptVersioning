"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với các trường hợp PII và JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay output bằng FailResult.fix_value thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  Với OnFailAction.FIX, Guardrails CHỈ thay output khi validator trả về
    FailResult(fix_value=...). PassResult giữ nguyên input.

⚠️  on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

Cách dùng:
    python 04_guardrails_validator.py              # chạy cả 2 demo
    python 04_guardrails_validator.py --demo pii   # chỉ demo PII
    python 04_guardrails_validator.py --demo json  # chỉ demo JSON
"""

import re
import json
import logging
import argparse

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction

# Ẩn cảnh báo "Failed to export spans" từ telemetry của Guardrails (không ảnh hưởng kết quả)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII) bằng regex.

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối / viết liền)
    """

    # Regex patterns cho từng loại PII
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\b\d{3})[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """
        Tìm PII trong value; nếu phát hiện thì trả về FailResult kèm bản đã redact.

        Bước:
          1. Copy value → redacted_text
          2. Với mỗi loại PII: tìm mọi match bằng re.findall và thay
             bằng "[PII_TYPE_REDACTED]" trong redacted_text
          3. Có PII    → FailResult(fix_value=redacted_text); OnFailAction.FIX
                         sẽ dùng fix_value làm validated_output
          4. Không PII → PassResult() (output giữ nguyên)
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in re.findall(pattern, value):
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            types = [p[0] for p in found_pii]
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {types}")
            return FailResult(
                error_message=f"Phát hiện PII: {', '.join(types)}",
                fix_value=redacted_text,
            )

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    Không sửa được → trả về JSON dự phòng {"error": ..., "raw": ...}.
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi.

        Bước:
          1. Strip whitespace đầu/cuối
          2. Xóa markdown fences bằng re.sub
          3. Thay single quotes → double quotes
          4. Xóa trailing commas trước } hoặc ]
          5. Trả về chuỗi đã sửa (chưa re-serialize)
        """
        text = text.strip()

        # Xóa markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # Single quotes → double quotes
        text = text.replace("'", '"')

        # Xóa trailing commas trước } hoặc ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Kiểm tra value có parse được thành JSON không.

        3 nhánh:
          1. JSON hợp lệ sẵn   → PassResult()
          2. Sửa được          → FailResult(fix_value=JSON đã chuẩn hóa, indent=2)
          3. Không sửa được    → FailResult(fix_value=JSON dự phòng {"error", "raw"})
        """
        # 1) Hợp lệ sẵn
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # 2) Thử sửa rồi parse lại
        try:
            parsed = json.loads(self._repair(value))
            print("  🔧 JSON đã được sửa thành công")
            return FailResult(
                error_message="JSON lỗi, đã tự sửa",
                fix_value=json.dumps(parsed, indent=2, ensure_ascii=False),
            )
        except json.JSONDecodeError as e:
            # 3) Không sửa được → JSON dự phòng
            print(f"  ❌ Không thể sửa JSON: {e}")
            fallback = json.dumps(
                {"error": "Không thể phân tích JSON", "raw": value[:200]},
                ensure_ascii=False,
            )
            return FailResult(error_message=f"Không thể sửa JSON: {e}", fix_value=fallback)


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    """Chạy PIIDetector (on_fail=FIX) trên các câu chứa PII giả và 1 câu sạch."""
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    # on_fail truyền vào CONSTRUCTOR của validator
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    # Toàn bộ là dữ liệu giả (RULES.md §6)
    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("Phone (dots)", "Reach the hotline at 555.010.4477 after 5pm."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Card (dash)",  "Backup card: 4000-1234-5678-9010, exp 12/29."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567, SSN: 987-65-4320"),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        print(f"\n[{label}]")
        try:
            result = guard.validate(text)
            print(f"  Input:  {text}")
            print(f"  Output: {result.validated_output}")
        except Exception as e:
            print(f"  Input:  {text}")
            print(f"  ❌ Lỗi khi validate: {e}")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    """Chạy JSONFormatter (on_fail=FIX) trên JSON hợp lệ, JSON lỗi và chuỗi không phải JSON."""
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    # on_fail truyền vào CONSTRUCTOR của validator
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value", "tags": ["a", "b",],}'),
        ("Combined errors",  "```json\n{'model': 'gpt-4o-mini', 'scores': [0.9, 0.8,],}\n```"),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        print(f"\n[{label}]")
        print(f"  Input:  {text!r}")
        try:
            result = guard.validate(text)
        except Exception as e:
            print(f"  ❌ Lỗi khi validate: {e}")
            continue

        # Với OnFailAction.FIX, Guardrails coi output đã fix là "passed" →
        # tự phân loại kết quả để log rõ nhánh nào của validator đã chạy.
        output = str(result.validated_output)
        if output == text:
            status = "✅ Valid (giữ nguyên)"
        elif output.startswith('{"error"'):
            status = "⚠️  Fallback JSON (không sửa được)"
        else:
            status = "🔧 Repaired"
        print(f"  Status: {status} | validation_passed={result.validation_passed}")
        print("  Output:")
        for line in output.splitlines():
            print(f"    {line}")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main(demo: str = "all"):
    """Chạy demo được chọn: "pii", "json" hoặc "all"."""
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    if demo in ("pii", "all"):
        demo_pii_guard()
    if demo in ("json", "all"):
        demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Guardrails validators")
    parser.add_argument("--demo", choices=["pii", "json", "all"], default="all")
    main(parser.parse_args().demo)
