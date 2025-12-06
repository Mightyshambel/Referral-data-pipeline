#!/usr/bin/env python3
"""
your_script.py
Pandas-based pipeline to produce referral_report.csv and profiling_report.csv.
Run: python scripts/your_script.py --data_dir data --out_dir out
"""

import argparse
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil import tz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("referral-pipeline")


# ---------- Config / filenames (change if input names differ) ----------
INPUT_FILES = {
    "user_referrals": "user_referrals.csv",
    "user_referral_logs": "user_referral_logs.csv",
    "user_logs": "user_logs.csv",
    "paid_transactions": "paid_transactions.csv",
    "user_referral_statuses": "user_referral_statuses.csv",
    "referral_rewards": "referral_rewards.csv",
    "lead_log": "lead_log.csv"
}

# Output filenames
OUT_REFERRAL_REPORT = "referral_report.csv"
OUT_PROFILING = "profiling_report.csv"
OUT_DIAGNOSTIC = "diagnostic_report.csv"


# ---------- Utilities ----------
def warn_missing_column(table_name, column_name, action):
    logger.warning(
        "%s missing column '%s'; skipping %s.",
        table_name,
        column_name,
        action
    )


def load_csv(path, parse_dates=None, table_name=None):
    df = pd.read_csv(path, dtype=str)
    df.columns = [str(col).strip().lower() for col in df.columns]

    if parse_dates:
        normalized_dates = [str(col).strip().lower() for col in parse_dates]
        for col in normalized_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                warn_missing_column(table_name or path, col, "date parsing")
    return df


def profile_df(df, name):
    if df is None:
        return None
    rows = []
    for c in df.columns:
        col = df[c]
        non_null = col.dropna()
        sample = non_null.unique()[:3].tolist()
        rows.append({
            "table": name,
            "column": c,
            "total_rows": len(df),
            "null_count": int(col.isna().sum()),
            "distinct_count": int(col.nunique(dropna=True)),
            "sample_values": str(sample)
        })
    return pd.DataFrame(rows)


def normalize_phone(s):
    if pd.isna(s):
        return s
    s = "".join(ch for ch in str(s) if ch.isdigit())
    return s[-12:] if len(s) > 12 else s


def ensure_dt(df, col):
    if col not in df.columns:
        return df
    try:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    except Exception:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
    return df


def assign_first_available(df, target_col, candidates):
    """
    Populate target_col using the first available column from candidates.
    Subsequent columns are used to fill null values in the target column.
    """
    available = [col for col in candidates if col in df.columns]
    if not available:
        return df
    df[target_col] = df[available[0]]
    for col in available[1:]:
        df[target_col] = df[target_col].fillna(df[col])
    return df


def to_zoneaware(dt_series, tzname):
    # convert utc-aware timestamps to target zone (tzname; e.g., 'Asia/Jakarta')
    try:
        return dt_series.dt.tz_convert(tzname)
    except Exception:
        # If not tz-aware or tzname invalid, return original
        return dt_series


# ---------- Business logic functions ----------
def compute_is_valid(df):
    # Expect these columns in df (create them if missing)
    # columns used: reward_value, referral_status, transaction_id, transaction_status,
    # transaction_type, transaction_at, referral_at, membership_expired_date,
    # is_deleted, is_reward_granted
    df['is_business_logic_valid'] = False
    df['notes'] = ""

    # normalize helper booleans/strings
    def safe_lower(col):
        return col.fillna("").astype(str).str.strip().str.lower()

    reward_val = pd.to_numeric(df.get('reward_value', pd.Series([np.nan]*len(df))), errors='coerce')
    referral_status = safe_lower(df.get('referral_status', pd.Series([""]*len(df))))
    txn_status = safe_lower(df.get('transaction_status', pd.Series([""]*len(df))))
    txn_type = safe_lower(df.get('transaction_type', pd.Series([""]*len(df))))
    is_deleted = df.get('is_deleted', pd.Series([False]*len(df)))
    is_deleted = is_deleted.replace({True: True, False: False, 'True': True, 'False': False, '1': True, '0': False})
    is_reward_granted = df.get('is_reward_granted', pd.Series([False]*len(df)))
    is_reward_granted = is_reward_granted.replace({True: True, False: False, 'True': True, 'False': False, '1': True, '0': False})

    # Ensure datetimes
    referral_at = df.get('referral_at')
    transaction_at = df.get('transaction_at')
    # membership expiry may be date-like
    mem_exp = pd.to_datetime(df.get('membership_expired_date', pd.Series([pd.NaT]*len(df))), errors='coerce')

    # Condition 1: many positive checks
    cond1 = (
        (reward_val > 0) &
        (referral_status == 'berhasil') &
        (~df['transaction_id'].isna()) &
        (txn_status == 'paid') &
        (txn_type == 'new') &
        (transaction_at.notna()) &
        (referral_at.notna())
    )

    # Additional: transaction_at >= referral_at and same month
    # Use UTC timestamps for these comparisons (they should be zone-aware)
    try:
        cond_time = (transaction_at >= referral_at) & (transaction_at.dt.month == referral_at.dt.month)
    except Exception:
        cond_time = pd.Series([False]*len(df))
    try:
        mem_after_txn = mem_exp > transaction_at.dt.date
    except Exception as exc:
        logger.warning("Failed to compare membership_expired_date and transaction_at: %s", exc)
        mem_after_txn = pd.Series([False]*len(df))
    cond1 = cond1 & cond_time & mem_after_txn & (~is_deleted) & (is_reward_granted == True)

    df.loc[cond1.fillna(False), 'is_business_logic_valid'] = True
    df.loc[cond1.fillna(False), 'notes'] = df.loc[cond1.fillna(False), 'notes'].astype(str) + "matched_cond1;"

    # Condition 2: referral_status in Menunggu or Tidak Berhasil AND reward_value is null
    cond2 = (referral_status.isin(['menunggu', 'tidak berhasil'])) & (reward_val.isna())
    df.loc[cond2.fillna(False), 'is_business_logic_valid'] = True
    df.loc[cond2.fillna(False), 'notes'] = df.loc[cond2.fillna(False), 'notes'].astype(str) + "matched_cond2;"

    # Invalid condition examples (explicitly flag as invalid)
    # E.g., reward_value > 0 but no transaction -> invalid
    cond_inv_no_txn = (reward_val > 0) & (df['transaction_id'].isna())
    df.loc[cond_inv_no_txn.fillna(False), 'notes'] = df.loc[cond_inv_no_txn.fillna(False), 'notes'].astype(str) + "invalid_no_txn;"

    # catch-alls: rows still False with no notes -> mark 'manual_review'
    df.loc[(df['is_business_logic_valid'] == False) & (df['notes'] == ""), 'notes'] = "manual_review;"
    return df


# ---------- Main pipeline ----------
def run_pipeline(data_dir: str, out_dir: str):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataframes
    parse_dates = {
        "user_referrals": ["referral_at", "created_at", "updated_at"],
        "paid_transactions": ["transaction_at", "created_at", "updated_at"],
        "user_logs": ["created_at", "updated_at"],
        "user_referral_logs": ["created_at", "updated_at"],
        "lead_log": ["created_at", "updated_at"]
    }
    dfs = {}
    profiles = []
    for key, fname in INPUT_FILES.items():
        fp = data_dir / fname
        pd_dates = parse_dates.get(key, None)
        if pd_dates is None:
            df = load_csv(str(fp), table_name=key)
        else:
            df = load_csv(str(fp), parse_dates=pd_dates, table_name=key)
        dfs[key] = df
        prof = profile_df(df, key)
        if prof is not None:
            profiles.append(prof)

    profiling_report = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    profiling_report.to_csv(out_dir / OUT_PROFILING, index=False)
    logger.info("Wrote profiling report -> %s", out_dir / OUT_PROFILING)

    # basic null-check
    if dfs.get('user_referrals') is None:
        logger.error("user_referrals missing; aborting")
        return

    # 2. Basic cleaning & type normalization
    ur = dfs['user_referrals'].copy()
    ur_columns_before = set(ur.columns)
    # normalize strings
    str_cols = ur.select_dtypes(include='object').columns
    for c in str_cols:
        ur[c] = ur[c].astype(str).str.strip().replace({'nan': None})

    # normalize phones if present
    if 'referrer_phone' in ur.columns:
        ur['referrer_phone'] = ur['referrer_phone'].apply(normalize_phone)

    # ensure datetime columns are timezone-aware UTC
    for dtc in ['referral_at', 'created_at', 'updated_at']:
        ur = ensure_dt(ur, dtc)

    # other dfs - ensure dt where relevant
    for name in ['paid_transactions', 'user_logs', 'user_referral_logs', 'lead_log']:
        df = dfs.get(name)
        if df is None:
            continue
        for c in df.columns:
            if 'date' in c.lower() or 'at' in c.lower():
                df = ensure_dt(df, c)
        dfs[name] = df

    # 3. Deduplicate (keep latest updated_at where possible)
    if 'id' in ur.columns:
        if 'updated_at' in ur.columns:
            ur = ur.sort_values('updated_at').drop_duplicates(subset=['id'], keep='last')
        else:
            warn_missing_column('user_referrals', 'updated_at', 'deduplicate with recency')
            ur = ur.drop_duplicates(subset=['id'], keep='last')
    else:
        warn_missing_column('user_referrals', 'id', 'deduplicate records')

    # 4. Merge joins to create working table
    merged = ur.copy()

    # join user_referral_logs (latest entry per user_referral_id)
    urlogs = dfs.get('user_referral_logs')
    if urlogs is not None:
        if 'user_referral_id' not in urlogs.columns:
            warn_missing_column('user_referral_logs', 'user_referral_id', 'deduplicate and join')
        elif 'id' not in merged.columns:
            warn_missing_column('user_referrals', 'id', 'join user_referral_logs')
        else:
            if 'updated_at' in urlogs.columns:
                urlogs = urlogs.sort_values('updated_at')
            else:
                warn_missing_column('user_referral_logs', 'updated_at', 'sort before deduplicate')
            urlogs = urlogs.drop_duplicates(subset=['user_referral_id'], keep='last')
            merged = merged.merge(urlogs.add_prefix('urlog_'), left_on='id', right_on='urlog_user_referral_id', how='left')

    # join paid_transactions by transaction_id
    tx = dfs.get('paid_transactions')
    if tx is not None:
        if 'transaction_id' not in tx.columns:
            warn_missing_column('paid_transactions', 'transaction_id', 'join with referrals')
        elif 'transaction_id' not in merged.columns:
            warn_missing_column('user_referrals', 'transaction_id', 'join paid transactions')
        else:
            if 'transaction_at' in tx.columns:
                tx = tx.sort_values('transaction_at')
            else:
                warn_missing_column('paid_transactions', 'transaction_at', 'sort before join')
            merged = merged.merge(tx.add_prefix('txn_'), left_on='transaction_id', right_on='txn_transaction_id', how='left')

    # join referrer user info
    ul = dfs.get('user_logs')
    if ul is not None:
        if 'user_id' not in ul.columns:
            warn_missing_column('user_logs', 'user_id', 'deduplicate and join')
        elif 'referrer_id' not in merged.columns:
            warn_missing_column('user_referrals', 'referrer_id', 'join referrer user info')
        else:
            if 'updated_at' in ul.columns:
                ul = ul.sort_values('updated_at')
            else:
                warn_missing_column('user_logs', 'updated_at', 'sort before deduplicate')
            ul = ul.drop_duplicates(subset=['user_id'], keep='last')
            merged = merged.merge(ul.add_prefix('referrer_'), left_on='referrer_id', right_on='referrer_user_id', how='left')

    # join rewards and statuses where available
    rr = dfs.get('referral_rewards')
    if rr is not None and 'id' in rr.columns:
        merged = merged.merge(rr.add_prefix('reward_'), left_on='referral_reward_id', right_on='reward_id', how='left')

    rs = dfs.get('user_referral_statuses')
    if rs is not None and 'id' in rs.columns:
        merged = merged.merge(rs.add_prefix('status_'), left_on='user_referral_status_id', right_on='status_id', how='left')

    # join lead_log if referral_source == 'Lead'
    leads = dfs.get('lead_log')
    if leads is not None:
        if 'lead_id' not in leads.columns:
            warn_missing_column('lead_log', 'lead_id', 'deduplicate and join')
        elif 'referral_source' not in merged.columns:
            warn_missing_column('user_referrals', 'referral_source', 'conditional lead join')
        elif 'referee_id' not in merged.columns:
            warn_missing_column('user_referrals', 'referee_id', 'join lead_log')
        else:
            if 'created_at' in leads.columns:
                leads = leads.sort_values('created_at')
            else:
                warn_missing_column('lead_log', 'created_at', 'sort before deduplicate')
            leads = leads.drop_duplicates(subset=['lead_id'], keep='last')
            merged = merged.merge(leads.add_prefix('lead_'), left_on='referee_id', right_on='lead_lead_id', how='left')

    # 5. Normalize and compute fields expected by business rules
    # create standardized columns used by compute_is_valid
    # reward_value: prefer reward_value from reward_ columns, fallback to referral_reward_value in ur
    reward_cols = [c for c in merged.columns if 'reward_value' in c]
    if reward_cols:
        merged['reward_value'] = pd.to_numeric(merged[reward_cols[0]], errors='coerce')
    else:
        merged['reward_value'] = pd.to_numeric(merged.get('reward_value'), errors='coerce')

    # referral_status textual
    merged = assign_first_available(merged, 'referral_status', ['status_name', 'referral_status', 'status_status'])

    # transaction columns mapping (txn_)
    merged = assign_first_available(merged, 'transaction_id', ['transaction_id', 'txn_transaction_id'])
    merged = assign_first_available(merged, 'transaction_at', ['transaction_at', 'txn_transaction_at'])
    merged = assign_first_available(merged, 'referral_at', ['referral_at', 'urlog_referral_at'])

    # membership expiration
    merged = assign_first_available(merged, 'membership_expired_date', ['membership_expired_date', 'referrer_membership_expired_date'])

    # is_deleted and is_reward_granted flags
    merged['is_deleted'] = merged.get('is_deleted', False)
    merged['is_reward_granted'] = merged.get('is_reward_granted', False)

    # Ensure referral_at and transaction_at are tz-aware datetimes
    merged = ensure_dt(merged, 'referral_at')
    merged = ensure_dt(merged, 'transaction_at')

    # If timezone info exists in referrer rows or lead rows, convert both datetimes to that zone for comparison
    tz_col = None
    if 'referrer_timezone' in merged.columns:
        tz_col = 'referrer_timezone'
    elif 'lead_timezone' in merged.columns:
        tz_col = 'lead_timezone'
    else:
        tz_col = None  # fallback to UTC comparisons

    if tz_col:
        merged['referral_at_local'] = merged.apply(
            lambda r: (r['referral_at'].tz_convert(r[tz_col]) if pd.notna(r['referral_at']) and r[tz_col] else r['referral_at']),
            axis=1
        )
        merged['transaction_at_local'] = merged.apply(
            lambda r: (r['transaction_at'].tz_convert(r[tz_col]) if pd.notna(r['transaction_at']) and r[tz_col] else r['transaction_at']),
            axis=1
        )
        # prefer local columns for logic
        merged['referral_at'] = merged['referral_at_local']
        merged['transaction_at'] = merged['transaction_at_local']

    # 6. Business logic
    merged = compute_is_valid(merged)

    # 7. Final report selection - keep only expected columns (take-home expects some schema; we'll include core fields)
    final_cols = [
        'id', 'referrer_id', 'referee_id', 'referral_at', 'transaction_id',
        'transaction_at', 'reward_value', 'referral_status', 'is_business_logic_valid', 'notes'
    ]
    final_cols_present = [c for c in final_cols if c in merged.columns]
    final = merged[final_cols_present].copy()

    # 8. Diagnostic: rows removed/dropped - we create a simple diagnostic table
    diagnostic = pd.DataFrame({
        "total_input_rows": [len(ur)],
        "total_output_rows": [len(final)],
    })
    diagnostic.to_csv(out_dir / OUT_DIAGNOSTIC, index=False)
    logger.info("Wrote diagnostic -> %s", out_dir / OUT_DIAGNOSTIC)

    # 9. Write final outputs
    final.to_csv(out_dir / OUT_REFERRAL_REPORT, index=False)
    logger.info("Wrote final report -> %s", out_dir / OUT_REFERRAL_REPORT)

    # 10. final check (assert expected count)
    if len(final) != 46:
        logger.warning("Output row count != 46 (actual=%s). Save a sample of non-matching rows to out/debug_sample.csv", len(final))
        final.head(200).to_csv(out_dir / "debug_sample.csv", index=False)
    else:
        logger.info("Output contains 46 rows as expected.")

    return {
        "final": final,
        "profiling": profiling_report,
        "diagnostic": diagnostic
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.out_dir)
