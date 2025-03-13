from pandas import DateOffset


def parse_time_offset(offset_str: str) -> DateOffset:
    """
    Convert a string like '1y', '6m', or '30d' into a pandas DateOffset.
    """
    unit = offset_str[-1]
    value = int(offset_str[:-1])
    if unit == 'y':
        return DateOffset(years=value)
    elif unit == 'm':
        return DateOffset(months=value)
    elif unit == 'd':
        return DateOffset(days=value)
    else:
        raise ValueError("Unsupported time offset. Use 'y', 'm', or 'd'.")