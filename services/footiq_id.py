import uuid


def generate_footiq_id() -> str:
    return f"FIQ-ALG-{uuid.uuid4().hex[:6].upper()}"