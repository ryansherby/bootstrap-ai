import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def get_retry_after_seconds(headers) -> float | None:
    """
    Parse Retry-After.

    Retry-After may be either:
      - a number of seconds
      - an HTTP date
    """
    if not headers:
        return None

    retry_after = headers.get("retry-after")

    if retry_after is None:
        return None

    # Most APIs return seconds.
    try:
        return max(float(retry_after), 0.0)
    except (TypeError, ValueError):
        pass

    # HTTP spec also allows an absolute HTTP date.
    try:
        retry_time = parsedate_to_datetime(retry_after)

        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        return max(
            (retry_time - now).total_seconds(),
            0.0,
        )

    except (TypeError, ValueError):
        return None


def get_backoff_seconds(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> float:
    """
    Exponential fallback:
        1, 2, 4, 8, 16, ...
    """
    return min(
        base_delay * (2 ** attempt),
        max_delay,
    )


def sleep_for_retry(
    headers,
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    retry_after = get_retry_after_seconds(headers)

    if retry_after is not None:
        delay = retry_after
    else:
        delay = get_backoff_seconds(
            attempt,
            base_delay,
            max_delay,
        )

    warnings_message = (
        f"Rate limited. Retrying in {delay:.2f} seconds "
        f"(attempt {attempt + 1})."
    )

    print(warnings_message)

    time.sleep(delay)