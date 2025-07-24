import datetime

# --- Helper Function to Convert Fraction to Date ---
def fraction_to_date_full(t, year=2024):
    """
    Converts a fraction of a year (0 <= t < 1) into a date (month and day)
    for a given year. Accounts for leap years.
    """
    start_date = datetime.date(year, 1, 1)
    # Determine number of days in the year
    if (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
        days_in_year = 366
    else:
        days_in_year = 365
    # Convert fraction to day offset (0-indexed)
    day_offset = int(t * days_in_year)
    event_date = start_date + datetime.timedelta(days=day_offset)
    return event_date
